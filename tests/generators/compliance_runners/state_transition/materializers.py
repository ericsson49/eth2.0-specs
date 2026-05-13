from __future__ import annotations

from typing import Any

from eth_consensus_specs.test.helpers.withdrawals import (
    set_compounding_withdrawal_credential_with_balance,
    set_eth1_withdrawal_credential_with_balance,
)
from eth_consensus_specs.utils.ssz.ssz_impl import serialize
from tests.generators.compliance_runners.gen_base.gen_typing import (
    TestCase,
    TestCasePart,
    TestCaseResult,
)

from .abstract_cases import AbstractStateTransitionCase

GENERATOR_NAME = "operations"
SUITE_NAME = "minizinc_abstract"
MATERIALIZED_HANDLER_NAMES = (
    "withdrawal_request",
    "consolidation_request",
)
VALIDATOR_INDEX = 0
TARGET_VALIDATOR_INDEX = 1
SOURCE_ADDRESS = b"\x22" * 20
TARGET_ADDRESS = b"\x33" * 20


class UnsupportedProfileError(ValueError):
    pass


def materialize_case(
    spec,
    state,
    abstract_case: AbstractStateTransitionCase,
    *,
    fork_name: str,
    preset_name: str,
) -> TestCaseResult:
    if abstract_case.handler_name not in MATERIALIZED_HANDLER_NAMES:
        raise UnsupportedProfileError(
            f"Unsupported state-transition handler: {abstract_case.handler_name}"
        )

    test_case = TestCase(
        fork_name=fork_name,
        preset_name=preset_name,
        runner_name=GENERATOR_NAME,
        handler_name=abstract_case.handler_name,
        suite_name=SUITE_NAME,
        case_name=abstract_case.case_name,
    )
    if abstract_case.handler_name == "withdrawal_request":
        return materialize_withdrawal_request(spec, state, test_case, abstract_case.profile)
    return materialize_consolidation_request(spec, state, test_case, abstract_case.profile)


def materialize_withdrawal_request(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    validator_index = VALIDATOR_INDEX
    prepare_state_for_profile(spec, state, validator_index, profile)
    withdrawal_request = build_withdrawal_request(spec, state, validator_index, profile)

    pre_state = state.copy()
    spec.process_withdrawal_request(state, withdrawal_request)

    meta = {
        "description": "MiniZinc-generated abstract validator-state profile",
        "profile": profile,
    }
    case_parts = [
        TestCasePart(("pre", "ssz", serialize(pre_state))),
        TestCasePart(("withdrawal_request", "ssz", serialize(withdrawal_request))),
        TestCasePart(("post", "ssz", serialize(state))),
    ]
    return TestCaseResult(test_case=test_case, meta=meta, case_parts=case_parts)


def materialize_consolidation_request(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    source_index = VALIDATOR_INDEX
    target_index = TARGET_VALIDATOR_INDEX
    prepare_state_for_profile(spec, state, source_index, profile)
    prepare_target_for_consolidation(spec, state, target_index)
    consolidation_request = build_consolidation_request(spec, state, source_index, target_index)

    pre_state = state.copy()
    spec.process_consolidation_request(state, consolidation_request)

    meta = {
        "description": "MiniZinc-generated abstract validator-state profile",
        "profile": profile,
    }
    case_parts = [
        TestCasePart(("pre", "ssz", serialize(pre_state))),
        TestCasePart(("consolidation_request", "ssz", serialize(consolidation_request))),
        TestCasePart(("post", "ssz", serialize(state))),
    ]
    return TestCaseResult(test_case=test_case, meta=meta, case_parts=case_parts)


def prepare_state_for_profile(spec, state, validator_index: int, profile: dict[str, Any]) -> None:
    current_epoch = spec.Epoch(max(2, spec.config.SHARD_COMMITTEE_PERIOD + 1))
    state.slot = spec.compute_start_slot_at_epoch(current_epoch)

    validator = state.validators[validator_index]
    validator.slashed = profile["slashed"]
    validator.activation_eligibility_epoch = profile_epoch(
        spec,
        current_epoch,
        "LT" if profile["activation_eligibility_epoch_set"] else "FAR_FUTURE",
    )
    if not profile["activation_epoch_set"]:
        validator.activation_epoch = spec.FAR_FUTURE_EPOCH
    elif profile["shard_committee_period_lte_current_epoch"]:
        validator.activation_epoch = spec.Epoch(current_epoch - spec.config.SHARD_COMMITTEE_PERIOD)
    else:
        validator.activation_epoch = profile_epoch(
            spec,
            current_epoch,
            profile["activation_epoch_to_current_epoch"],
        )
    validator.exit_epoch = profile_epoch(
        spec,
        current_epoch,
        profile["exit_epoch_to_current_epoch"] if profile["exit_epoch_set"] else "FAR_FUTURE",
    )
    validator.withdrawable_epoch = profile_epoch(
        spec,
        current_epoch,
        profile["withdrawable_epoch_to_current_epoch"]
        if profile["withdrawable_epoch_set"]
        else "FAR_FUTURE",
    )

    effective_balance = choose_effective_balance(spec, profile)
    balance = choose_balance(spec, effective_balance, profile)

    withdrawal_credential_type = profile["withdrawal_credential_type"]
    if withdrawal_credential_type == "ETH1":
        set_eth1_withdrawal_credential_with_balance(
            spec,
            state,
            validator_index,
            effective_balance=effective_balance,
            balance=balance,
            address=SOURCE_ADDRESS,
        )
    elif withdrawal_credential_type == "COMP":
        set_compounding_withdrawal_credential_with_balance(
            spec,
            state,
            validator_index,
            effective_balance=effective_balance,
            balance=balance,
            address=SOURCE_ADDRESS,
        )
    elif withdrawal_credential_type == "UNKNOWN":
        validator.withdrawal_credentials = b"\xff" + b"\x00" * 11 + SOURCE_ADDRESS
        validator.effective_balance = effective_balance
        state.balances[validator_index] = balance
    else:
        validator.effective_balance = effective_balance
        state.balances[validator_index] = balance

    state.pending_partial_withdrawals = spec.List[
        spec.PendingPartialWithdrawal, spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
    ]()
    if profile["has_pending_withdrawal_request"]:
        pending_amount = min(spec.Gwei(1), state.balances[validator_index])
        state.pending_partial_withdrawals.append(
            spec.PendingPartialWithdrawal(
                validator_index=validator_index,
                amount=pending_amount,
                withdrawable_epoch=spec.Epoch(current_epoch + 1),
            )
        )


def build_withdrawal_request(spec, state, validator_index: int, profile: dict[str, Any]):
    amount = spec.FULL_EXIT_REQUEST_AMOUNT
    if profile["withdrawal_credential_type"] == "COMP" and not profile["exit_epoch_set"]:
        amount = spec.Gwei(1)
    return spec.WithdrawalRequest(
        source_address=SOURCE_ADDRESS,
        validator_pubkey=state.validators[validator_index].pubkey,
        amount=amount,
    )


def prepare_target_for_consolidation(spec, state, target_index: int) -> None:
    current_epoch = spec.get_current_epoch(state)
    target = state.validators[target_index]
    target.slashed = False
    target.activation_eligibility_epoch = spec.Epoch(0)
    target.activation_epoch = spec.Epoch(0)
    target.exit_epoch = spec.FAR_FUTURE_EPOCH
    target.withdrawable_epoch = spec.FAR_FUTURE_EPOCH
    assert spec.is_active_validator(target, current_epoch)
    set_compounding_withdrawal_credential_with_balance(
        spec,
        state,
        target_index,
        effective_balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
        balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
        address=TARGET_ADDRESS,
    )


def build_consolidation_request(spec, state, source_index: int, target_index: int):
    return spec.ConsolidationRequest(
        source_address=SOURCE_ADDRESS,
        source_pubkey=state.validators[source_index].pubkey,
        target_pubkey=state.validators[target_index].pubkey,
    )


def profile_epoch(spec, current_epoch, relation: str):
    if relation == "FAR_FUTURE":
        return spec.FAR_FUTURE_EPOCH
    if relation == "<":
        return spec.Epoch(current_epoch - 1)
    if relation == "=":
        return current_epoch
    if relation == ">":
        return spec.Epoch(current_epoch + 1)
    if relation == "LT":
        return spec.Epoch(current_epoch - 1)
    raise ValueError(f"Unsupported epoch relation: {relation}")


def choose_effective_balance(spec, profile: dict[str, Any]):
    relation_to_min = profile["effective_balance_to_min_activation_balance"]
    relation_to_max = profile["effective_balance_to_max_effective_balance"]

    if relation_to_max == "=":
        return spec.MAX_EFFECTIVE_BALANCE_ELECTRA
    if relation_to_min == "<":
        return spec.Gwei(max(0, int(spec.MIN_ACTIVATION_BALANCE) - int(spec.EFFECTIVE_BALANCE_INCREMENT)))
    if relation_to_min == "=":
        return spec.MIN_ACTIVATION_BALANCE
    return spec.Gwei(int(spec.MIN_ACTIVATION_BALANCE) + int(spec.EFFECTIVE_BALANCE_INCREMENT))


def choose_balance(spec, effective_balance, profile: dict[str, Any]):
    if profile["balance_is_zero"]:
        return spec.Gwei(0)

    relation = profile["balance_to_effective_balance"]
    if relation == "<":
        return spec.Gwei(max(0, int(effective_balance) - 1))
    if relation == "=":
        return effective_balance
    if relation == ">":
        return spec.Gwei(int(effective_balance) + int(spec.EFFECTIVE_BALANCE_INCREMENT))
    raise ValueError(f"Unsupported balance relation: {relation}")
