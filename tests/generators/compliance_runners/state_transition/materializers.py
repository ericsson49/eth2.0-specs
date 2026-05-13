from __future__ import annotations

from typing import Any

from eth_consensus_specs.test.helpers.bls_to_execution_changes import get_signed_address_change
from eth_consensus_specs.test.helpers.deposits import (
    prepare_pending_deposit,
    prepare_state_and_deposit,
)
from eth_consensus_specs.test.helpers.keys import privkeys, pubkeys
from eth_consensus_specs.test.helpers.voluntary_exits import sign_voluntary_exit
from eth_consensus_specs.test.helpers.withdrawals import (
    set_compounding_withdrawal_credential_with_balance,
    set_eth1_withdrawal_credential_with_balance,
)
from eth_consensus_specs.utils import bls
from eth_consensus_specs.utils.ssz.ssz_impl import serialize
from tests.generators.compliance_runners.gen_base.gen_typing import (
    TestCase,
    TestCasePart,
    TestCaseResult,
)

from .abstract_cases import AbstractStateTransitionCase

OPERATIONS_RUNNER_NAME = "operations"
EPOCH_PROCESSING_RUNNER_NAME = "epoch_processing"
SUITE_NAME = "minizinc_abstract"
MATERIALIZED_HANDLER_NAMES = (
    "deposit",
    "bls_to_execution_change",
    "deposit_request",
    "voluntary_exit",
    "withdrawal_request",
    "consolidation_request",
    "pending_deposits",
    "pending_consolidations",
    "effective_balance_updates",
)
VALIDATOR_INDEX = 0
TARGET_VALIDATOR_INDEX = 1
HELPER_VALIDATOR_INDEX = 2
SOURCE_ADDRESS = b"\x22" * 20
TARGET_ADDRESS = b"\x33" * 20
HELPER_ADDRESS = b"\x77" * 20
DEPOSIT_PUBKEY = b"\x44" * 48
DEPOSIT_WITHDRAWAL_CREDENTIALS = b"\x01" + b"\x00" * 11 + b"\x55" * 20
DEPOSIT_SIGNATURE = b"\x66" * 96


class UnsupportedProfileError(ValueError):
    pass


def materialize_case(
    spec,
    state,
    abstract_case: AbstractStateTransitionCase,
    *,
    fork_name: str,
    preset_name: str,
    invalid_operation: bool = False,
) -> TestCaseResult:
    if abstract_case.handler_name not in MATERIALIZED_HANDLER_NAMES:
        raise UnsupportedProfileError(
            f"Unsupported state-transition handler: {abstract_case.handler_name}"
        )

    test_case = TestCase(
        fork_name=fork_name,
        preset_name=preset_name,
        runner_name=runner_name_for_handler(abstract_case.handler_name),
        handler_name=abstract_case.handler_name,
        suite_name=SUITE_NAME,
        case_name=abstract_case.case_name,
    )
    if abstract_case.handler_name == "deposit":
        return materialize_deposit(
            spec,
            state,
            test_case,
            abstract_case.profile,
            invalid_operation=invalid_operation,
        )
    if abstract_case.handler_name == "bls_to_execution_change":
        return materialize_bls_to_execution_change(
            spec,
            state,
            test_case,
            abstract_case.profile,
            invalid_operation=invalid_operation,
        )
    if abstract_case.handler_name == "deposit_request":
        return materialize_deposit_request(
            spec,
            state,
            test_case,
            abstract_case.profile,
            invalid_operation=invalid_operation,
        )
    if abstract_case.handler_name == "voluntary_exit":
        return materialize_voluntary_exit(
            spec,
            state,
            test_case,
            abstract_case.profile,
            invalid_operation=invalid_operation,
        )
    if abstract_case.handler_name == "withdrawal_request":
        return materialize_withdrawal_request(
            spec,
            state,
            test_case,
            abstract_case.profile,
            invalid_operation=invalid_operation,
        )
    if abstract_case.handler_name == "consolidation_request":
        return materialize_consolidation_request(
            spec,
            state,
            test_case,
            abstract_case.profile,
            invalid_operation=invalid_operation,
        )
    if abstract_case.handler_name == "pending_deposits":
        return materialize_pending_deposits(
            spec,
            state,
            test_case,
            abstract_case.profile,
        )
    if abstract_case.handler_name == "pending_consolidations":
        return materialize_pending_consolidations(
            spec,
            state,
            test_case,
            abstract_case.profile,
        )
    return materialize_effective_balance_updates(
        spec,
        state,
        test_case,
        abstract_case.profile,
    )


def materialize_pending_consolidations(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_pending_consolidations(spec, state, profile)

    pre_state = state.copy()
    case_parts = [TestCasePart(("pre", "ssz", serialize(pre_state)))]
    try:
        spec.process_pending_consolidations(state)
    except AssertionError:
        return TestCaseResult(
            test_case=test_case,
            meta=operation_meta(profile, operation_valid=False, post_state_changed=None),
            case_parts=case_parts,
        )

    post_state_changed = pre_state != state
    case_parts.append(TestCasePart(("post", "ssz", serialize(state))))
    return TestCaseResult(
        test_case=test_case,
        meta=operation_meta(
            profile,
            operation_valid=True,
            post_state_changed=post_state_changed,
        ),
        case_parts=case_parts,
    )


def materialize_effective_balance_updates(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_effective_balance_updates(spec, state, profile)

    pre_state = state.copy()
    case_parts = [TestCasePart(("pre", "ssz", serialize(pre_state)))]
    try:
        spec.process_effective_balance_updates(state)
    except AssertionError:
        return TestCaseResult(
            test_case=test_case,
            meta=operation_meta(profile, operation_valid=False, post_state_changed=None),
            case_parts=case_parts,
        )

    post_state_changed = pre_state != state
    case_parts.append(TestCasePart(("post", "ssz", serialize(state))))
    return TestCaseResult(
        test_case=test_case,
        meta=operation_meta(
            profile,
            operation_valid=True,
            post_state_changed=post_state_changed,
        ),
        case_parts=case_parts,
    )


def runner_name_for_handler(handler_name: str) -> str:
    if handler_name in (
        "pending_deposits",
        "pending_consolidations",
        "effective_balance_updates",
    ):
        return EPOCH_PROCESSING_RUNNER_NAME
    return OPERATIONS_RUNNER_NAME


def materialize_deposit(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
    *,
    invalid_operation: bool,
) -> TestCaseResult:
    deposit = prepare_state_for_deposit(spec, state, profile)
    if invalid_operation:
        deposit.proof[0] = b"\xff" * 32

    pre_state = state.copy()
    case_parts = operation_case_parts(pre_state, "deposit", deposit)
    try:
        with_bls_setting(profile, lambda: spec.process_deposit(state, deposit))
    except AssertionError:
        return TestCaseResult(
            test_case=test_case,
            meta=operation_meta(profile, operation_valid=False, post_state_changed=None),
            case_parts=case_parts,
        )

    post_state_changed = pre_state != state
    case_parts.append(TestCasePart(("post", "ssz", serialize(state))))
    return TestCaseResult(
        test_case=test_case,
        meta=operation_meta(
            profile,
            operation_valid=True,
            post_state_changed=post_state_changed,
        ),
        case_parts=case_parts,
    )


def materialize_bls_to_execution_change(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
    *,
    invalid_operation: bool,
) -> TestCaseResult:
    signed_address_change = prepare_state_for_bls_to_execution_change(spec, state, profile)
    if invalid_operation:
        signed_address_change.message.validator_index = spec.ValidatorIndex(len(state.validators))

    pre_state = state.copy()
    case_parts = operation_case_parts(pre_state, "address_change", signed_address_change)
    try:
        with_bls_setting(
            profile,
            lambda: spec.process_bls_to_execution_change(state, signed_address_change),
        )
    except AssertionError:
        return TestCaseResult(
            test_case=test_case,
            meta=operation_meta(profile, operation_valid=False, post_state_changed=None),
            case_parts=case_parts,
        )

    post_state_changed = pre_state != state
    case_parts.append(TestCasePart(("post", "ssz", serialize(state))))
    return TestCaseResult(
        test_case=test_case,
        meta=operation_meta(
            profile,
            operation_valid=True,
            post_state_changed=post_state_changed,
        ),
        case_parts=case_parts,
    )


def materialize_deposit_request(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
    *,
    invalid_operation: bool,
) -> TestCaseResult:
    prepare_state_for_deposit_request(spec, state, profile)
    deposit_request = build_deposit_request(spec, profile)
    if invalid_operation:
        deposit_request.amount = spec.Gwei(0)

    pre_state = state.copy()
    case_parts = operation_case_parts(pre_state, "deposit_request", deposit_request)
    try:
        spec.process_deposit_request(state, deposit_request)
    except AssertionError:
        return TestCaseResult(
            test_case=test_case,
            meta=operation_meta(profile, operation_valid=False, post_state_changed=None),
            case_parts=case_parts,
        )

    post_state_changed = pre_state != state
    case_parts.append(TestCasePart(("post", "ssz", serialize(state))))
    return TestCaseResult(
        test_case=test_case,
        meta=operation_meta(
            profile,
            operation_valid=True,
            post_state_changed=post_state_changed,
        ),
        case_parts=case_parts,
    )


def materialize_voluntary_exit(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
    *,
    invalid_operation: bool,
) -> TestCaseResult:
    validator_index = VALIDATOR_INDEX
    prepare_state_for_voluntary_exit(spec, state, validator_index, profile)
    signed_voluntary_exit = build_signed_voluntary_exit(spec, state, validator_index, profile)
    if invalid_operation:
        signed_voluntary_exit.signature = b"\xff" * 96

    pre_state = state.copy()
    case_parts = operation_case_parts(pre_state, "voluntary_exit", signed_voluntary_exit)
    try:
        spec.process_voluntary_exit(state, signed_voluntary_exit)
    except AssertionError:
        return TestCaseResult(
            test_case=test_case,
            meta=operation_meta(profile, operation_valid=False, post_state_changed=None),
            case_parts=case_parts,
        )

    post_state_changed = pre_state != state
    case_parts.append(TestCasePart(("post", "ssz", serialize(state))))
    return TestCaseResult(
        test_case=test_case,
        meta=operation_meta(
            profile,
            operation_valid=True,
            post_state_changed=post_state_changed,
        ),
        case_parts=case_parts,
    )


def materialize_withdrawal_request(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
    *,
    invalid_operation: bool,
) -> TestCaseResult:
    validator_index = VALIDATOR_INDEX
    prepare_state_for_profile(spec, state, validator_index, profile)
    withdrawal_request = build_withdrawal_request(spec, state, validator_index, profile)
    apply_withdrawal_intent(spec, state, validator_index, withdrawal_request, profile)
    if invalid_operation:
        withdrawal_request.source_address = invalid_source_address()

    pre_state = state.copy()
    case_parts = operation_case_parts(pre_state, "withdrawal_request", withdrawal_request)
    try:
        spec.process_withdrawal_request(state, withdrawal_request)
    except AssertionError:
        return TestCaseResult(
            test_case=test_case,
            meta=operation_meta(profile, operation_valid=False, post_state_changed=None),
            case_parts=case_parts,
        )

    post_state_changed = pre_state != state
    case_parts.append(TestCasePart(("post", "ssz", serialize(state))))
    return TestCaseResult(
        test_case=test_case,
        meta=operation_meta(
            profile,
            operation_valid=True,
            post_state_changed=post_state_changed,
        ),
        case_parts=case_parts,
    )


def materialize_consolidation_request(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
    *,
    invalid_operation: bool,
) -> TestCaseResult:
    source_index = VALIDATOR_INDEX
    target_index = TARGET_VALIDATOR_INDEX
    prepare_state_for_profile(spec, state, source_index, profile)
    prepare_target_for_consolidation(spec, state, target_index)
    consolidation_request = build_consolidation_request(spec, state, source_index, target_index)
    apply_consolidation_intent(
        spec,
        state,
        source_index,
        target_index,
        consolidation_request,
        profile,
    )
    if invalid_operation:
        consolidation_request.source_address = invalid_source_address()

    pre_state = state.copy()
    case_parts = operation_case_parts(pre_state, "consolidation_request", consolidation_request)
    try:
        spec.process_consolidation_request(state, consolidation_request)
    except AssertionError:
        return TestCaseResult(
            test_case=test_case,
            meta=operation_meta(profile, operation_valid=False, post_state_changed=None),
            case_parts=case_parts,
        )

    post_state_changed = pre_state != state
    case_parts.append(TestCasePart(("post", "ssz", serialize(state))))
    return TestCaseResult(
        test_case=test_case,
        meta=operation_meta(
            profile,
            operation_valid=True,
            post_state_changed=post_state_changed,
        ),
        case_parts=case_parts,
    )


def materialize_pending_deposits(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_pending_deposits(spec, state, profile)

    pre_state = state.copy()
    case_parts = [TestCasePart(("pre", "ssz", serialize(pre_state)))]
    try:
        spec.process_pending_deposits(state)
    except AssertionError:
        return TestCaseResult(
            test_case=test_case,
            meta=operation_meta(profile, operation_valid=False, post_state_changed=None),
            case_parts=case_parts,
        )

    post_state_changed = pre_state != state
    case_parts.append(TestCasePart(("post", "ssz", serialize(state))))
    return TestCaseResult(
        test_case=test_case,
        meta=operation_meta(
            profile,
            operation_valid=True,
            post_state_changed=post_state_changed,
        ),
        case_parts=case_parts,
    )


def operation_case_parts(pre_state, operation_name: str, operation) -> list[TestCasePart]:
    return [
        TestCasePart(("pre", "ssz", serialize(pre_state))),
        TestCasePart((operation_name, "ssz", serialize(operation))),
    ]


def operation_meta(
    profile: dict[str, Any],
    *,
    operation_valid: bool,
    post_state_changed: bool | None,
) -> dict[str, Any]:
    return {
        "description": "MiniZinc-generated abstract validator-state profile",
        "profile": profile,
        "operation_valid": operation_valid,
        "post_state_changed": post_state_changed,
        "coverage_tags": profile.get("coverage_tags", []),
        "bls_setting": profile.get("bls_setting", 0),
    }


def with_bls_setting(profile: dict[str, Any], fn):
    old_bls_active = bls.bls_active
    bls.bls_active = bool(profile.get("bls_setting", 0))
    try:
        return fn()
    finally:
        bls.bls_active = old_bls_active


def prepare_state_for_deposit(spec, state, profile: dict[str, Any]):
    state.pending_deposits = spec.List[spec.PendingDeposit, spec.PENDING_DEPOSITS_LIMIT]()
    intent = profile.get("guide_intent")
    if intent in (None, "new_validator"):
        return prepare_state_and_deposit(
            spec,
            state,
            validator_index=len(state.validators),
            amount=spec.MIN_ACTIVATION_BALANCE,
            withdrawal_credentials=DEPOSIT_WITHDRAWAL_CREDENTIALS,
            signed=True,
        )
    if intent == "top_up_existing_validator":
        return prepare_state_and_deposit(
            spec,
            state,
            validator_index=VALIDATOR_INDEX,
            amount=spec.EFFECTIVE_BALANCE_INCREMENT,
            withdrawal_credentials=state.validators[VALIDATOR_INDEX].withdrawal_credentials,
            signed=True,
        )
    if intent == "invalid_proof":
        deposit = prepare_state_and_deposit(
            spec,
            state,
            validator_index=len(state.validators),
            amount=spec.MIN_ACTIVATION_BALANCE,
            withdrawal_credentials=DEPOSIT_WITHDRAWAL_CREDENTIALS,
            signed=True,
        )
        state.eth1_data.deposit_root = b"\xff" * 32
        return deposit
    if intent == "invalid_signature":
        profile["bls_setting"] = 1
        return prepare_state_and_deposit(
            spec,
            state,
            validator_index=len(state.validators),
            amount=spec.MIN_ACTIVATION_BALANCE,
            withdrawal_credentials=DEPOSIT_WITHDRAWAL_CREDENTIALS,
            signed=False,
        )
    raise ValueError(f"Unsupported deposit guide intent: {intent}")


def prepare_state_for_bls_to_execution_change(spec, state, profile: dict[str, Any]):
    validator_index = VALIDATOR_INDEX
    withdrawal_pubkey = pubkeys[-1 - validator_index]
    validator = state.validators[validator_index]
    validator.withdrawal_credentials = spec.BLS_WITHDRAWAL_PREFIX + spec.hash(withdrawal_pubkey)[1:]

    intent = profile.get("guide_intent")
    if intent in (None, "success"):
        return get_signed_address_change(
            spec,
            state,
            validator_index=validator_index,
            withdrawal_pubkey=withdrawal_pubkey,
            to_execution_address=SOURCE_ADDRESS,
        )
    if intent == "out_of_range":
        return get_signed_address_change(
            spec,
            state,
            validator_index=len(state.validators),
            withdrawal_pubkey=withdrawal_pubkey,
            to_execution_address=SOURCE_ADDRESS,
        )
    if intent == "not_bls_credentials":
        validator.withdrawal_credentials = spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b"\x00" * 11 + SOURCE_ADDRESS
        return get_signed_address_change(
            spec,
            state,
            validator_index=validator_index,
            withdrawal_pubkey=withdrawal_pubkey,
            to_execution_address=SOURCE_ADDRESS,
        )
    if intent == "pubkey_mismatch":
        return get_signed_address_change(
            spec,
            state,
            validator_index=validator_index,
            withdrawal_pubkey=pubkeys[-2 - validator_index],
            to_execution_address=SOURCE_ADDRESS,
        )
    if intent == "bad_signature":
        profile["bls_setting"] = 1
        signed_address_change = get_signed_address_change(
            spec,
            state,
            validator_index=validator_index,
            withdrawal_pubkey=withdrawal_pubkey,
            to_execution_address=SOURCE_ADDRESS,
        )
        signed_address_change.signature = spec.BLSSignature(b"\x42" * 96)
        return signed_address_change
    raise ValueError(f"Unsupported BLS to execution change guide intent: {intent}")


def invalid_source_address() -> bytes:
    return b"\xff" * 20


def prepare_state_for_deposit_request(spec, state, profile: dict[str, Any]) -> None:
    state.pending_deposits = spec.List[spec.PendingDeposit, spec.PENDING_DEPOSITS_LIMIT]()
    intent = profile.get("guide_intent")
    if intent == "start_index_set" or (
        intent is None and profile["activation_eligibility_epoch_set"]
    ):
        state.deposit_requests_start_index = spec.uint64(0)
    else:
        state.deposit_requests_start_index = spec.UNSET_DEPOSIT_REQUESTS_START_INDEX


def build_deposit_request(spec, profile: dict[str, Any]):
    amount = spec.MIN_ACTIVATION_BALANCE
    if profile["effective_balance_to_min_activation_balance"] == "<":
        amount = spec.Gwei(max(1, int(spec.MIN_ACTIVATION_BALANCE) - int(spec.EFFECTIVE_BALANCE_INCREMENT)))
    elif profile["effective_balance_to_max_effective_balance"] == "=":
        amount = spec.MAX_EFFECTIVE_BALANCE_ELECTRA

    index = spec.uint64(0)
    if profile.get("guide_intent") == "start_index_set" or profile["activation_eligibility_epoch_set"]:
        index = spec.uint64(1)

    return spec.DepositRequest(
        pubkey=DEPOSIT_PUBKEY,
        withdrawal_credentials=DEPOSIT_WITHDRAWAL_CREDENTIALS,
        amount=amount,
        signature=DEPOSIT_SIGNATURE,
        index=index,
    )


def prepare_state_for_voluntary_exit(
    spec,
    state,
    validator_index: int,
    profile: dict[str, Any],
) -> None:
    current_epoch = spec.Epoch(max(2, spec.config.SHARD_COMMITTEE_PERIOD + 1))
    state.slot = spec.compute_start_slot_at_epoch(current_epoch)
    state.pending_partial_withdrawals = spec.List[
        spec.PendingPartialWithdrawal, spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
    ]()

    validator = state.validators[validator_index]
    validator.slashed = False
    validator.activation_eligibility_epoch = spec.Epoch(0)
    validator.activation_epoch = spec.Epoch(0)
    validator.exit_epoch = spec.FAR_FUTURE_EPOCH
    validator.withdrawable_epoch = spec.FAR_FUTURE_EPOCH
    set_eth1_withdrawal_credential_with_balance(
        spec,
        state,
        validator_index,
        effective_balance=spec.MIN_ACTIVATION_BALANCE,
        balance=spec.MIN_ACTIVATION_BALANCE,
        address=SOURCE_ADDRESS,
    )

    intent = profile.get("guide_intent")
    if intent in (None, "success", "future_epoch"):
        return
    if intent == "inactive":
        validator.activation_epoch = spec.FAR_FUTURE_EPOCH
    elif intent == "already_exited":
        validator.exit_epoch = spec.Epoch(current_epoch + 1)
    elif intent == "not_active_long_enough":
        validator.activation_epoch = current_epoch
    elif intent == "pending_withdrawal":
        state.pending_partial_withdrawals.append(
            spec.PendingPartialWithdrawal(
                validator_index=validator_index,
                amount=spec.Gwei(1),
                withdrawable_epoch=spec.Epoch(current_epoch + 1),
            )
        )
    else:
        raise ValueError(f"Unsupported voluntary exit guide intent: {intent}")


def build_signed_voluntary_exit(spec, state, validator_index: int, profile: dict[str, Any]):
    current_epoch = spec.get_current_epoch(state)
    voluntary_exit_epoch = current_epoch
    if profile.get("guide_intent") == "future_epoch":
        voluntary_exit_epoch = spec.Epoch(current_epoch + 1)

    voluntary_exit = spec.VoluntaryExit(
        epoch=voluntary_exit_epoch,
        validator_index=validator_index,
    )
    privkey = privkeys[validator_index]
    return sign_voluntary_exit(spec, state, voluntary_exit, privkey)


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


def apply_withdrawal_intent(spec, state, validator_index: int, withdrawal_request, profile) -> None:
    intent = profile.get("guide_intent")
    if intent is None:
        return

    prepare_withdrawal_source(spec, state, validator_index, compounding=intent != "success_full_exit")
    withdrawal_request.source_address = SOURCE_ADDRESS
    withdrawal_request.validator_pubkey = state.validators[validator_index].pubkey
    withdrawal_request.amount = spec.FULL_EXIT_REQUEST_AMOUNT

    if intent in ("success_partial_withdrawal", "queue_full"):
        withdrawal_request.amount = spec.Gwei(1)

    if intent == "queue_full":
        fill_pending_partial_withdrawals(spec, state, validator_index)
    elif intent == "pubkey_missing":
        withdrawal_request.validator_pubkey = b"\xff" * 48
    elif intent == "bad_source_address":
        withdrawal_request.source_address = invalid_source_address()
    elif intent == "source_inactive":
        state.validators[validator_index].activation_epoch = spec.FAR_FUTURE_EPOCH
    elif intent == "source_exiting":
        state.validators[validator_index].exit_epoch = spec.Epoch(spec.get_current_epoch(state) + 1)
    elif intent == "not_active_long_enough":
        state.validators[validator_index].activation_epoch = spec.get_current_epoch(state)
    elif intent == "full_exit_with_pending_withdrawal":
        state.pending_partial_withdrawals.append(
            spec.PendingPartialWithdrawal(
                validator_index=validator_index,
                amount=spec.Gwei(1),
                withdrawable_epoch=spec.Epoch(spec.get_current_epoch(state) + 1),
            )
        )
    elif intent == "partial_conditions_not_met":
        set_eth1_withdrawal_credential_with_balance(
            spec,
            state,
            validator_index,
            effective_balance=spec.MIN_ACTIVATION_BALANCE,
            balance=spec.MIN_ACTIVATION_BALANCE,
            address=SOURCE_ADDRESS,
        )
        withdrawal_request.amount = spec.Gwei(1)
    elif intent in ("success_full_exit", "success_partial_withdrawal"):
        return
    else:
        raise ValueError(f"Unsupported withdrawal guide intent: {intent}")


def prepare_withdrawal_source(spec, state, validator_index: int, *, compounding: bool) -> None:
    current_epoch = spec.get_current_epoch(state)
    validator = state.validators[validator_index]
    validator.slashed = False
    validator.activation_eligibility_epoch = spec.Epoch(0)
    validator.activation_epoch = spec.Epoch(
        max(0, int(current_epoch) - int(spec.config.SHARD_COMMITTEE_PERIOD))
    )
    validator.exit_epoch = spec.FAR_FUTURE_EPOCH
    validator.withdrawable_epoch = spec.FAR_FUTURE_EPOCH
    if compounding:
        set_compounding_withdrawal_credential_with_balance(
            spec,
            state,
            validator_index,
            effective_balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
            balance=spec.Gwei(int(spec.MIN_ACTIVATION_BALANCE) + int(spec.EFFECTIVE_BALANCE_INCREMENT)),
            address=SOURCE_ADDRESS,
        )
    else:
        set_eth1_withdrawal_credential_with_balance(
            spec,
            state,
            validator_index,
            effective_balance=spec.MIN_ACTIVATION_BALANCE,
            balance=spec.MIN_ACTIVATION_BALANCE,
            address=SOURCE_ADDRESS,
        )
    state.pending_partial_withdrawals = spec.List[
        spec.PendingPartialWithdrawal, spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
    ]()


def fill_pending_partial_withdrawals(spec, state, validator_index: int) -> None:
    state.pending_partial_withdrawals = spec.List[
        spec.PendingPartialWithdrawal, spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
    ]()
    for _ in range(spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT):
        state.pending_partial_withdrawals.append(
            spec.PendingPartialWithdrawal(
                validator_index=validator_index,
                amount=spec.Gwei(1),
                withdrawable_epoch=spec.Epoch(spec.get_current_epoch(state) + 1),
            )
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


def apply_consolidation_intent(
    spec,
    state,
    source_index: int,
    target_index: int,
    consolidation_request,
    profile,
) -> None:
    intent = profile.get("guide_intent")
    if intent is None:
        return

    prepare_consolidation_source(spec, state, source_index)
    prepare_target_for_consolidation(spec, state, target_index)
    state.pending_consolidations = spec.List[
        spec.PendingConsolidation, spec.PENDING_CONSOLIDATIONS_LIMIT
    ]()
    state.pending_partial_withdrawals = spec.List[
        spec.PendingPartialWithdrawal, spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
    ]()
    consolidation_request.source_address = SOURCE_ADDRESS
    consolidation_request.source_pubkey = state.validators[source_index].pubkey
    consolidation_request.target_pubkey = state.validators[target_index].pubkey

    if intent == "switch_to_compounding_success":
        prepare_switch_to_compounding_source(spec, state, source_index)
        consolidation_request.target_pubkey = state.validators[source_index].pubkey
    elif intent == "switch_pubkey_missing":
        prepare_switch_to_compounding_source(spec, state, source_index)
        consolidation_request.source_pubkey = b"\xff" * 48
        consolidation_request.target_pubkey = b"\xff" * 48
    elif intent == "switch_bad_source_address":
        prepare_switch_to_compounding_source(spec, state, source_index)
        consolidation_request.target_pubkey = state.validators[source_index].pubkey
        consolidation_request.source_address = invalid_source_address()
    elif intent == "switch_source_inactive":
        prepare_switch_to_compounding_source(spec, state, source_index)
        consolidation_request.target_pubkey = state.validators[source_index].pubkey
        state.validators[source_index].activation_epoch = spec.FAR_FUTURE_EPOCH
    elif intent == "switch_source_exiting":
        prepare_switch_to_compounding_source(spec, state, source_index)
        consolidation_request.target_pubkey = state.validators[source_index].pubkey
        state.validators[source_index].exit_epoch = spec.Epoch(spec.get_current_epoch(state) + 1)
    elif intent == "source_equals_target":
        consolidation_request.target_pubkey = state.validators[source_index].pubkey
    elif intent == "queue_full":
        fill_pending_consolidations(spec, state, source_index, target_index)
    elif intent == "churn_too_low":
        set_compounding_withdrawal_credential_with_balance(
            spec,
            state,
            source_index,
            effective_balance=spec.MIN_ACTIVATION_BALANCE,
            balance=spec.MIN_ACTIVATION_BALANCE,
            address=SOURCE_ADDRESS,
        )
        set_compounding_withdrawal_credential_with_balance(
            spec,
            state,
            target_index,
            effective_balance=spec.MIN_ACTIVATION_BALANCE,
            balance=spec.MIN_ACTIVATION_BALANCE,
            address=TARGET_ADDRESS,
        )
    elif intent == "source_missing":
        consolidation_request.source_pubkey = b"\xff" * 48
    elif intent == "target_missing":
        consolidation_request.target_pubkey = b"\xff" * 48
    elif intent == "bad_source_address":
        consolidation_request.source_address = invalid_source_address()
    elif intent == "target_not_compounding":
        set_eth1_withdrawal_credential_with_balance(
            spec,
            state,
            target_index,
            effective_balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
            balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
            address=TARGET_ADDRESS,
        )
    elif intent == "source_inactive":
        prepare_churn_helper(spec, state)
        state.validators[source_index].activation_epoch = spec.FAR_FUTURE_EPOCH
    elif intent == "target_inactive":
        prepare_churn_helper(spec, state)
        state.validators[target_index].activation_epoch = spec.FAR_FUTURE_EPOCH
    elif intent == "source_exiting":
        state.validators[source_index].exit_epoch = spec.Epoch(spec.get_current_epoch(state) + 1)
    elif intent == "target_exiting":
        state.validators[target_index].exit_epoch = spec.Epoch(spec.get_current_epoch(state) + 1)
    elif intent == "source_not_active_long_enough":
        state.validators[source_index].activation_epoch = spec.get_current_epoch(state)
    elif intent == "source_pending_withdrawal":
        state.pending_partial_withdrawals.append(
            spec.PendingPartialWithdrawal(
                validator_index=source_index,
                amount=spec.Gwei(1),
                withdrawable_epoch=spec.Epoch(spec.get_current_epoch(state) + 1),
            )
        )
    elif intent == "success":
        return
    else:
        raise ValueError(f"Unsupported consolidation guide intent: {intent}")


def prepare_consolidation_source(spec, state, source_index: int) -> None:
    current_epoch = spec.get_current_epoch(state)
    source = state.validators[source_index]
    source.slashed = False
    source.activation_eligibility_epoch = spec.Epoch(0)
    source.activation_epoch = spec.Epoch(
        max(0, int(current_epoch) - int(spec.config.SHARD_COMMITTEE_PERIOD))
    )
    source.exit_epoch = spec.FAR_FUTURE_EPOCH
    source.withdrawable_epoch = spec.FAR_FUTURE_EPOCH
    set_compounding_withdrawal_credential_with_balance(
        spec,
        state,
        source_index,
        effective_balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
        balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
        address=SOURCE_ADDRESS,
    )


def prepare_switch_to_compounding_source(spec, state, source_index: int) -> None:
    current_epoch = spec.get_current_epoch(state)
    source = state.validators[source_index]
    source.slashed = False
    source.activation_eligibility_epoch = spec.Epoch(0)
    source.activation_epoch = spec.Epoch(
        max(0, int(current_epoch) - int(spec.config.SHARD_COMMITTEE_PERIOD))
    )
    source.exit_epoch = spec.FAR_FUTURE_EPOCH
    source.withdrawable_epoch = spec.FAR_FUTURE_EPOCH
    set_eth1_withdrawal_credential_with_balance(
        spec,
        state,
        source_index,
        effective_balance=spec.MIN_ACTIVATION_BALANCE,
        balance=spec.MIN_ACTIVATION_BALANCE,
        address=SOURCE_ADDRESS,
    )


def prepare_churn_helper(spec, state) -> None:
    current_epoch = spec.get_current_epoch(state)
    helper = state.validators[HELPER_VALIDATOR_INDEX]
    helper.slashed = False
    helper.activation_eligibility_epoch = spec.Epoch(0)
    helper.activation_epoch = spec.Epoch(
        max(0, int(current_epoch) - int(spec.config.SHARD_COMMITTEE_PERIOD))
    )
    helper.exit_epoch = spec.FAR_FUTURE_EPOCH
    helper.withdrawable_epoch = spec.FAR_FUTURE_EPOCH
    set_compounding_withdrawal_credential_with_balance(
        spec,
        state,
        HELPER_VALIDATOR_INDEX,
        effective_balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
        balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
        address=HELPER_ADDRESS,
    )


def fill_pending_consolidations(spec, state, source_index: int, target_index: int) -> None:
    state.pending_consolidations = spec.List[
        spec.PendingConsolidation, spec.PENDING_CONSOLIDATIONS_LIMIT
    ]()
    for _ in range(spec.PENDING_CONSOLIDATIONS_LIMIT):
        state.pending_consolidations.append(
            spec.PendingConsolidation(source_index=source_index, target_index=target_index)
        )


def prepare_state_for_pending_deposits(spec, state, profile: dict[str, Any]) -> None:
    state.deposit_requests_start_index = state.eth1_deposit_index
    state.deposit_balance_to_consume = spec.Gwei(0)
    state.pending_deposits = spec.List[spec.PendingDeposit, spec.PENDING_DEPOSITS_LIMIT]()

    intent = profile.get("guide_intent")
    if intent in (None, "success_top_up"):
        add_pending_deposit(spec, state, VALIDATOR_INDEX, spec.EFFECTIVE_BALANCE_INCREMENT)
    elif intent == "not_finalized":
        add_pending_deposit(
            spec,
            state,
            VALIDATOR_INDEX,
            spec.EFFECTIVE_BALANCE_INCREMENT,
            slot=spec.Slot(1),
        )
        state.finalized_checkpoint.epoch = spec.GENESIS_EPOCH
    elif intent == "churn_limit_reached":
        add_pending_deposit(
            spec,
            state,
            VALIDATOR_INDEX,
            spec.Gwei(spec.get_activation_exit_churn_limit(state) + 1),
        )
    elif intent == "exited_validator_postponed":
        add_pending_deposit(spec, state, VALIDATOR_INDEX, spec.EFFECTIVE_BALANCE_INCREMENT)
        spec.initiate_validator_exit(state, spec.ValidatorIndex(VALIDATOR_INDEX))
    elif intent == "withdrawable_validator":
        add_pending_deposit(
            spec,
            state,
            VALIDATOR_INDEX,
            spec.Gwei(spec.get_activation_exit_churn_limit(state) + 1),
        )
        spec.initiate_validator_exit(state, spec.ValidatorIndex(VALIDATOR_INDEX))
        state.slot = spec.compute_start_slot_at_epoch(
            spec.Epoch(state.validators[VALIDATOR_INDEX].withdrawable_epoch + 1)
        )
    elif intent == "eth1_bridge_blocks_request":
        state.deposit_requests_start_index = spec.uint64(state.eth1_deposit_index + 1)
        add_pending_deposit(
            spec,
            state,
            VALIDATOR_INDEX,
            spec.EFFECTIVE_BALANCE_INCREMENT,
            slot=spec.Slot(1),
        )
        state.finalized_checkpoint.epoch = spec.Epoch(1)
    elif intent == "max_per_epoch_reached":
        amount = spec.EFFECTIVE_BALANCE_INCREMENT
        state.deposit_balance_to_consume = spec.Gwei(
            amount * (spec.MAX_PENDING_DEPOSITS_PER_EPOCH + 1)
        )
        for validator_index in range(spec.MAX_PENDING_DEPOSITS_PER_EPOCH + 1):
            add_pending_deposit(spec, state, validator_index, amount)
    elif intent == "new_validator":
        add_pending_deposit(
            spec,
            state,
            len(state.validators),
            spec.EFFECTIVE_BALANCE_INCREMENT,
            withdrawal_credentials=DEPOSIT_WITHDRAWAL_CREDENTIALS,
        )
    else:
        raise ValueError(f"Unsupported pending deposits guide intent: {intent}")


def add_pending_deposit(
    spec,
    state,
    validator_index: int,
    amount,
    *,
    slot=None,
    withdrawal_credentials=None,
) -> None:
    if withdrawal_credentials is None:
        withdrawal_credentials = state.validators[validator_index].withdrawal_credentials
    state.pending_deposits.append(
        prepare_pending_deposit(
            spec,
            validator_index=validator_index,
            amount=amount,
            withdrawal_credentials=withdrawal_credentials,
            signed=True,
            slot=slot,
        )
    )


def prepare_state_for_pending_consolidations(spec, state, profile: dict[str, Any]) -> None:
    state.pending_consolidations = spec.List[
        spec.PendingConsolidation, spec.PENDING_CONSOLIDATIONS_LIMIT
    ]()
    state.pending_deposits = spec.List[spec.PendingDeposit, spec.PENDING_DEPOSITS_LIMIT]()
    prepare_pending_consolidation_pair(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)

    intent = profile.get("guide_intent")
    if intent == "empty_queue":
        return
    if intent in (None, "success"):
        add_pending_consolidation(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
    elif intent == "not_withdrawable":
        add_pending_consolidation(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
        state.validators[VALIDATOR_INDEX].withdrawable_epoch = spec.Epoch(
            spec.get_current_epoch(state) + 2
        )
    elif intent == "slashed_source_skipped":
        prepare_pending_consolidation_pair(spec, state, HELPER_VALIDATOR_INDEX, 3)
        add_pending_consolidation(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
        add_pending_consolidation(spec, state, HELPER_VALIDATOR_INDEX, 3)
        state.validators[VALIDATOR_INDEX].slashed = True
    elif intent == "source_balance_less_than_effective_balance":
        add_pending_consolidation(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
        state.balances[VALIDATOR_INDEX] = spec.Gwei(
            state.validators[VALIDATOR_INDEX].effective_balance
            - spec.EFFECTIVE_BALANCE_INCREMENT // 8
        )
    elif intent == "source_balance_greater_than_effective_balance":
        add_pending_consolidation(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
        state.balances[VALIDATOR_INDEX] = spec.Gwei(
            state.validators[VALIDATOR_INDEX].effective_balance
            + spec.EFFECTIVE_BALANCE_INCREMENT // 8
        )
    elif intent == "blocked_after_processed":
        prepare_pending_consolidation_pair(spec, state, HELPER_VALIDATOR_INDEX, 3)
        add_pending_consolidation(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
        add_pending_consolidation(spec, state, HELPER_VALIDATOR_INDEX, 3)
        state.validators[HELPER_VALIDATOR_INDEX].withdrawable_epoch = spec.Epoch(
            spec.get_current_epoch(state) + 2
        )
    else:
        raise ValueError(f"Unsupported pending consolidations guide intent: {intent}")


def prepare_pending_consolidation_pair(spec, state, source_index: int, target_index: int) -> None:
    current_epoch = spec.get_current_epoch(state)
    source = state.validators[source_index]
    target = state.validators[target_index]
    source.slashed = False
    source.activation_eligibility_epoch = spec.Epoch(0)
    source.activation_epoch = spec.Epoch(0)
    target.activation_eligibility_epoch = spec.Epoch(0)
    target.activation_epoch = spec.Epoch(0)
    target.exit_epoch = spec.FAR_FUTURE_EPOCH
    target.withdrawable_epoch = spec.FAR_FUTURE_EPOCH
    set_compounding_withdrawal_credential_with_balance(
        spec,
        state,
        source_index,
        effective_balance=spec.MIN_ACTIVATION_BALANCE,
        balance=spec.MIN_ACTIVATION_BALANCE,
        address=SOURCE_ADDRESS,
    )
    set_compounding_withdrawal_credential_with_balance(
        spec,
        state,
        target_index,
        effective_balance=spec.MIN_ACTIVATION_BALANCE,
        balance=spec.MIN_ACTIVATION_BALANCE,
        address=TARGET_ADDRESS,
    )
    source.exit_epoch = spec.Epoch(0)
    source.withdrawable_epoch = current_epoch


def add_pending_consolidation(spec, state, source_index: int, target_index: int) -> None:
    state.pending_consolidations.append(
        spec.PendingConsolidation(source_index=source_index, target_index=target_index)
    )


def prepare_state_for_effective_balance_updates(spec, state, profile: dict[str, Any]) -> None:
    index = VALIDATOR_INDEX
    validator = state.validators[index]
    validator.slashed = False
    validator.activation_eligibility_epoch = spec.Epoch(0)
    validator.activation_epoch = spec.Epoch(0)
    validator.exit_epoch = spec.FAR_FUTURE_EPOCH
    validator.withdrawable_epoch = spec.FAR_FUTURE_EPOCH

    increment = spec.EFFECTIVE_BALANCE_INCREMENT
    hysteresis_increment = increment // spec.HYSTERESIS_QUOTIENT
    downward_threshold = hysteresis_increment * spec.HYSTERESIS_DOWNWARD_MULTIPLIER
    upward_threshold = hysteresis_increment * spec.HYSTERESIS_UPWARD_MULTIPLIER
    intent = profile.get("guide_intent")

    if intent in (None, "no_change_at_threshold"):
        set_eth1_withdrawal_credential_with_balance(
            spec,
            state,
            index,
            effective_balance=spec.MIN_ACTIVATION_BALANCE,
            balance=spec.Gwei(spec.MIN_ACTIVATION_BALANCE + upward_threshold),
            address=SOURCE_ADDRESS,
        )
    elif intent == "step_down":
        set_eth1_withdrawal_credential_with_balance(
            spec,
            state,
            index,
            effective_balance=spec.MIN_ACTIVATION_BALANCE,
            balance=spec.Gwei(spec.MIN_ACTIVATION_BALANCE - downward_threshold - 1),
            address=SOURCE_ADDRESS,
        )
    elif intent == "step_up":
        set_compounding_withdrawal_credential_with_balance(
            spec,
            state,
            index,
            effective_balance=spec.MIN_ACTIVATION_BALANCE,
            balance=spec.Gwei(spec.MIN_ACTIVATION_BALANCE + upward_threshold + 1),
            address=SOURCE_ADDRESS,
        )
    elif intent == "cap_at_max":
        set_compounding_withdrawal_credential_with_balance(
            spec,
            state,
            index,
            effective_balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA - increment,
            balance=spec.Gwei(spec.MAX_EFFECTIVE_BALANCE_ELECTRA + increment),
            address=SOURCE_ADDRESS,
        )
    else:
        raise ValueError(f"Unsupported effective balance updates guide intent: {intent}")


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
