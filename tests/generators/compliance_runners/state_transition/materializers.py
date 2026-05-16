from __future__ import annotations

from typing import Any

from eth_consensus_specs.test.helpers.attestations import get_valid_attestation
from eth_consensus_specs.test.helpers.attester_slashings import (
    get_valid_attester_slashing_by_indices,
)
from eth_consensus_specs.test.helpers.bls_to_execution_changes import get_signed_address_change
from eth_consensus_specs.test.helpers.deposits import (
    prepare_pending_deposit,
    prepare_state_and_deposit,
)
from eth_consensus_specs.test.helpers.keys import privkeys, pubkeys
from eth_consensus_specs.test.helpers.proposer_slashings import get_valid_proposer_slashing
from eth_consensus_specs.test.helpers.state import transition_to
from eth_consensus_specs.test.helpers.sync_committee import (
    compute_aggregate_sync_committee_signature,
    compute_committee_indices,
)
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
    "proposer_slashing",
    "attester_slashing",
    "attestation",
    "deposit",
    "bls_to_execution_change",
    "deposit_request",
    "voluntary_exit",
    "withdrawal_request",
    "consolidation_request",
    "pending_deposits",
    "pending_consolidations",
    "effective_balance_updates",
    "registry_updates",
    "slashings",
    "justification_and_finalization",
    "inactivity_updates",
    "rewards_and_penalties",
    "participation_flag_updates",
    "slashings_reset",
    "randao_mixes_reset",
    "eth1_data_reset",
    "historical_summaries_update",
    "sync_committee_updates",
    "sync_aggregate",
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
    if abstract_case.handler_name == "proposer_slashing":
        return materialize_proposer_slashing(
            spec,
            state,
            test_case,
            abstract_case.profile,
            invalid_operation=invalid_operation,
        )
    if abstract_case.handler_name == "attester_slashing":
        return materialize_attester_slashing(
            spec,
            state,
            test_case,
            abstract_case.profile,
            invalid_operation=invalid_operation,
        )
    if abstract_case.handler_name == "attestation":
        return materialize_attestation(
            spec,
            state,
            test_case,
            abstract_case.profile,
            invalid_operation=invalid_operation,
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
    if abstract_case.handler_name == "effective_balance_updates":
        return materialize_effective_balance_updates(
            spec,
            state,
            test_case,
            abstract_case.profile,
        )
    if abstract_case.handler_name == "registry_updates":
        return materialize_registry_updates(spec, state, test_case, abstract_case.profile)
    if abstract_case.handler_name == "slashings":
        return materialize_slashings(spec, state, test_case, abstract_case.profile)
    if abstract_case.handler_name == "justification_and_finalization":
        return materialize_justification_and_finalization(
            spec, state, test_case, abstract_case.profile
        )
    if abstract_case.handler_name == "inactivity_updates":
        return materialize_inactivity_updates(spec, state, test_case, abstract_case.profile)
    if abstract_case.handler_name == "rewards_and_penalties":
        return materialize_rewards_and_penalties(spec, state, test_case, abstract_case.profile)
    if abstract_case.handler_name == "participation_flag_updates":
        return materialize_participation_flag_updates(
            spec, state, test_case, abstract_case.profile
        )
    if abstract_case.handler_name == "slashings_reset":
        return materialize_slashings_reset(spec, state, test_case, abstract_case.profile)
    if abstract_case.handler_name == "randao_mixes_reset":
        return materialize_randao_mixes_reset(spec, state, test_case, abstract_case.profile)
    if abstract_case.handler_name == "eth1_data_reset":
        return materialize_eth1_data_reset(spec, state, test_case, abstract_case.profile)
    if abstract_case.handler_name == "historical_summaries_update":
        return materialize_historical_summaries_update(
            spec, state, test_case, abstract_case.profile
        )
    if abstract_case.handler_name == "sync_committee_updates":
        return materialize_sync_committee_updates(spec, state, test_case, abstract_case.profile)
    return materialize_sync_aggregate(
        spec,
        state,
        test_case,
        abstract_case.profile,
        invalid_operation=invalid_operation,
    )


def materialize_proposer_slashing(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
    *,
    invalid_operation: bool,
) -> TestCaseResult:
    proposer_slashing = prepare_state_for_proposer_slashing(spec, state, profile)
    if invalid_operation:
        proposer_slashing.signed_header_2.message.slot += 1

    pre_state = state.copy()
    case_parts = operation_case_parts(pre_state, "proposer_slashing", proposer_slashing)
    try:
        with_bls_setting(profile, lambda: spec.process_proposer_slashing(state, proposer_slashing))
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


def materialize_attester_slashing(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
    *,
    invalid_operation: bool,
) -> TestCaseResult:
    attester_slashing = prepare_state_for_attester_slashing(spec, state, profile)
    if invalid_operation:
        attester_slashing.attestation_2.data = attester_slashing.attestation_1.data.copy()

    pre_state = state.copy()
    case_parts = operation_case_parts(pre_state, "attester_slashing", attester_slashing)
    try:
        with_bls_setting(profile, lambda: spec.process_attester_slashing(state, attester_slashing))
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


def materialize_attestation(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
    *,
    invalid_operation: bool,
) -> TestCaseResult:
    attestation = prepare_state_for_attestation(spec, state, profile)
    if invalid_operation:
        attestation.data.slot = state.slot

    pre_state = state.copy()
    case_parts = operation_case_parts(pre_state, "attestation", attestation)
    try:
        with_bls_setting(profile, lambda: spec.process_attestation(state, attestation))
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


def materialize_registry_updates(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_registry_updates(spec, state, profile)

    pre_state = state.copy()
    case_parts = [TestCasePart(("pre", "ssz", serialize(pre_state)))]
    try:
        spec.process_registry_updates(state)
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


def materialize_slashings(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_slashings(spec, state, profile)

    pre_state = state.copy()
    case_parts = [TestCasePart(("pre", "ssz", serialize(pre_state)))]
    try:
        spec.process_slashings(state)
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


def materialize_justification_and_finalization(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_justification_and_finalization(spec, state, profile)

    pre_state = state.copy()
    case_parts = [TestCasePart(("pre", "ssz", serialize(pre_state)))]
    try:
        spec.process_justification_and_finalization(state)
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


def materialize_inactivity_updates(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_inactivity_updates(spec, state, profile)
    return materialize_epoch_processor(
        spec.process_inactivity_updates,
        state,
        test_case,
        profile,
    )


def materialize_rewards_and_penalties(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_rewards_and_penalties(spec, state, profile)
    return materialize_epoch_processor(
        spec.process_rewards_and_penalties,
        state,
        test_case,
        profile,
    )


def materialize_participation_flag_updates(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_participation_flag_updates(spec, state, profile)
    return materialize_epoch_processor(
        spec.process_participation_flag_updates,
        state,
        test_case,
        profile,
    )


def materialize_slashings_reset(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_slashings_reset(spec, state, profile)
    return materialize_epoch_processor(
        spec.process_slashings_reset,
        state,
        test_case,
        profile,
    )


def materialize_randao_mixes_reset(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_randao_mixes_reset(spec, state, profile)
    return materialize_epoch_processor(
        spec.process_randao_mixes_reset,
        state,
        test_case,
        profile,
    )


def materialize_eth1_data_reset(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_eth1_data_reset(spec, state, profile)
    return materialize_epoch_processor(
        spec.process_eth1_data_reset,
        state,
        test_case,
        profile,
    )


def materialize_historical_summaries_update(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_historical_summaries_update(spec, state, profile)
    return materialize_epoch_processor(
        spec.process_historical_summaries_update,
        state,
        test_case,
        profile,
    )


def materialize_sync_committee_updates(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
) -> TestCaseResult:
    prepare_state_for_sync_committee_updates(spec, state, profile)
    return materialize_epoch_processor(
        spec.process_sync_committee_updates,
        state,
        test_case,
        profile,
    )


def materialize_epoch_processor(process_fn, state, test_case: TestCase, profile: dict[str, Any]):
    pre_state = state.copy()
    case_parts = [TestCasePart(("pre", "ssz", serialize(pre_state)))]
    try:
        process_fn(state)
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
        "justification_and_finalization",
        "registry_updates",
        "slashings",
        "inactivity_updates",
        "rewards_and_penalties",
        "participation_flag_updates",
        "slashings_reset",
        "randao_mixes_reset",
        "eth1_data_reset",
        "historical_summaries_update",
        "sync_committee_updates",
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
    if "input_profiles" in profile and not has_input_profile_constraints(profile, "validator_state"):
        compounding = (
            input_profile_shape(
                profile,
                "withdrawal_request_input",
                "request_kind",
            )
            == "PARTIAL_WITHDRAWAL_REQUEST"
        )
        prepare_withdrawal_source(spec, state, validator_index, compounding=compounding)
    if "input_profiles" in profile:
        apply_queue_profile_for_validator(spec, state, validator_index, profile)
    withdrawal_request = build_withdrawal_request(spec, state, validator_index, profile)
    apply_withdrawal_profile(spec, state, validator_index, withdrawal_request, profile)
    if not profile.get("profile_driven"):
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
    if "input_profiles" in profile:
        apply_queue_profile_for_validator(spec, state, source_index, profile)
    consolidation_request = build_consolidation_request(spec, state, source_index, target_index)
    apply_consolidation_profile(
        spec,
        state,
        source_index,
        target_index,
        consolidation_request,
        profile,
    )
    if not profile.get("profile_driven"):
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


def materialize_sync_aggregate(
    spec,
    state,
    test_case: TestCase,
    profile: dict[str, Any],
    *,
    invalid_operation: bool,
) -> TestCaseResult:
    sync_aggregate = prepare_state_for_sync_aggregate(spec, state, profile)
    if invalid_operation:
        profile["bls_setting"] = 1
        sync_aggregate.sync_committee_signature = spec.BLSSignature(b"\x42" * 96)

    pre_state = state.copy()
    case_parts = operation_case_parts(pre_state, "sync_aggregate", sync_aggregate)
    try:
        with_bls_setting(profile, lambda: spec.process_sync_aggregate(state, sync_aggregate))
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


def prepare_state_for_proposer_slashing(spec, state, profile: dict[str, Any]):
    transition_to(spec, state, spec.compute_start_slot_at_epoch(spec.Epoch(2)))
    proposer_index = VALIDATOR_INDEX
    prepare_slashable_validator(spec, state, proposer_index)
    proposer_slashing = get_valid_proposer_slashing(
        spec,
        state,
        slashed_index=proposer_index,
        signed_1=True,
        signed_2=True,
    )
    if "input_profiles" in profile:
        if input_profile_shape(profile, "proposer_slashing_input", "branch_target") is not None:
            apply_proposer_slashing_intent(
                spec,
                state,
                proposer_index,
                proposer_slashing,
                profile,
            )
            return proposer_slashing
        proposer_profile = profile.get("input_profiles", {}).get("proposer_slashing_input", {})
        if proposer_profile.get("header_relation") == "SAME_HEADER":
            proposer_slashing.signed_header_2.message = proposer_slashing.signed_header_1.message.copy()
        if proposer_profile.get("proposer_relation") == "DIFFERENT_PROPOSER":
            proposer_slashing.signed_header_2.message.proposer_index = spec.ValidatorIndex(
                proposer_index + 1
            )
        if proposer_profile.get("proposer_status") == "PROPOSER_ALREADY_SLASHED":
            state.validators[proposer_index].slashed = True
        apply_operation_signature_profile(spec, profile, proposer_slashing.signed_header_2)
        return proposer_slashing

    intent = profile.get("guide_intent")
    if intent in (None, "success"):
        return proposer_slashing
    if intent == "same_header":
        proposer_slashing.signed_header_2.message = proposer_slashing.signed_header_1.message.copy()
    elif intent == "proposer_mismatch":
        proposer_slashing.signed_header_2.message.proposer_index = spec.ValidatorIndex(
            proposer_index + 1
        )
    elif intent == "already_slashed":
        state.validators[proposer_index].slashed = True
    elif intent == "bad_signature":
        profile["bls_setting"] = 1
        proposer_slashing.signed_header_2.signature = spec.BLSSignature(b"\x42" * 96)
    else:
        raise ValueError(f"Unsupported proposer slashing guide intent: {intent}")
    return proposer_slashing


def apply_proposer_slashing_intent(
    spec,
    state,
    proposer_index: int,
    proposer_slashing,
    profile: dict[str, Any],
) -> None:
    intent = profile.get("guide_intent")
    if intent in (None, "success"):
        return
    if intent == "same_header":
        proposer_slashing.signed_header_2.message = proposer_slashing.signed_header_1.message.copy()
    elif intent == "proposer_mismatch":
        proposer_slashing.signed_header_2.message.proposer_index = spec.ValidatorIndex(
            proposer_index + 1
        )
    elif intent == "already_slashed":
        state.validators[proposer_index].slashed = True
    elif intent == "bad_signature":
        profile["bls_setting"] = 1
        proposer_slashing.signed_header_2.signature = spec.BLSSignature(b"\x42" * 96)
    else:
        raise ValueError(f"Unsupported proposer slashing guide intent: {intent}")


def prepare_state_for_attester_slashing(spec, state, profile: dict[str, Any]):
    transition_to(spec, state, spec.compute_start_slot_at_epoch(spec.Epoch(2)))
    prepare_slashable_validator(spec, state, VALIDATOR_INDEX)
    prepare_slashable_validator(spec, state, TARGET_VALIDATOR_INDEX)

    intent = profile.get("guide_intent")
    attester_profile = profile.get("input_profiles", {}).get("attester_slashing_input", {})
    indices_1 = [spec.ValidatorIndex(VALIDATOR_INDEX)]
    indices_2 = [spec.ValidatorIndex(VALIDATOR_INDEX)]
    if intent == "no_overlap" or attester_profile.get("attester_overlap") == "DISJOINT":
        indices_2 = [spec.ValidatorIndex(TARGET_VALIDATOR_INDEX)]

    attester_slashing = get_valid_attester_slashing_by_indices(
        spec,
        state,
        indices_1=indices_1,
        indices_2=indices_2,
        slot=spec.Slot(state.slot - 1),
        signed_1=True,
        signed_2=True,
    )
    if "input_profiles" in profile:
        if input_profile_shape(profile, "attester_slashing_input", "branch_target") is not None:
            apply_attester_slashing_intent(spec, state, attester_slashing, profile)
            return attester_slashing
        if attester_profile.get("attestation_data_relation") == "ATTESTATION_DATA_SAME":
            attester_slashing.attestation_2.data = attester_slashing.attestation_1.data.copy()
        if attester_profile.get("attester_status") == "ATTESTER_ALREADY_SLASHED":
            state.validators[VALIDATOR_INDEX].slashed = True
        apply_operation_signature_profile(spec, profile, attester_slashing.attestation_2)
        return attester_slashing

    if intent in (None, "success", "no_overlap"):
        return attester_slashing
    if intent == "not_slashable_data":
        attester_slashing.attestation_2.data = attester_slashing.attestation_1.data.copy()
        return attester_slashing
    if intent == "already_slashed":
        state.validators[VALIDATOR_INDEX].slashed = True
        return attester_slashing
    if intent == "bad_signature":
        profile["bls_setting"] = 1
        attester_slashing.attestation_2.signature = spec.BLSSignature(b"\x42" * 96)
        return attester_slashing
    raise ValueError(f"Unsupported attester slashing guide intent: {intent}")


def apply_attester_slashing_intent(spec, state, attester_slashing, profile: dict[str, Any]) -> None:
    intent = profile.get("guide_intent")
    if intent in (None, "success", "no_overlap"):
        return
    if intent == "not_slashable_data":
        attester_slashing.attestation_2.data = attester_slashing.attestation_1.data.copy()
    elif intent == "already_slashed":
        state.validators[VALIDATOR_INDEX].slashed = True
    elif intent == "bad_signature":
        profile["bls_setting"] = 1
        attester_slashing.attestation_2.signature = spec.BLSSignature(b"\x42" * 96)
    else:
        raise ValueError(f"Unsupported attester slashing guide intent: {intent}")


def prepare_state_for_attestation(spec, state, profile: dict[str, Any]):
    if "input_profiles" in profile:
        apply_epoch_boundary_profile(spec, state, profile)
        if spec.get_current_epoch(state) <= spec.GENESIS_EPOCH:
            transition_to(spec, state, spec.compute_start_slot_at_epoch(spec.Epoch(2)) + 1)
    else:
        transition_to(spec, state, spec.compute_start_slot_at_epoch(spec.Epoch(2)) + 1)
    slot = spec.Slot(state.slot - 1)
    attestation_profile = profile.get("input_profiles", {}).get("attestation_input", {})
    if (
        profile.get("guide_intent") == "previous_epoch_success"
        or attestation_profile.get("slot_relation") == "PREVIOUS_EPOCH"
    ):
        slot = spec.Slot(spec.compute_start_slot_at_epoch(spec.Epoch(2)) - 1)
    attestation = get_valid_attestation(
        spec,
        state,
        slot=slot,
        signed=True,
    )
    if "input_profiles" in profile:
        if input_profile_shape(profile, "attestation_input", "branch_target") is not None:
            apply_attestation_intent(spec, state, attestation, profile)
            return attestation
        if attestation_profile.get("slot_relation") == "FUTURE_SLOT":
            attestation.data.slot = state.slot
        if attestation_profile.get("target_epoch_relation") == "WRONG_TARGET_EPOCH":
            attestation.data.target.epoch = spec.Epoch(attestation.data.target.epoch + 1)
        if attestation_profile.get("committee_index_shape") == "COMMITTEE_BAD_INDEX":
            attestation.committee_bits = spec.Bitvector[spec.MAX_COMMITTEES_PER_SLOT](
                [False] * spec.MAX_COMMITTEES_PER_SLOT
            )
            attestation.committee_bits[spec.MAX_COMMITTEES_PER_SLOT - 1] = True
        if attestation_profile.get("aggregation_shape") == "AGGREGATION_EMPTY":
            for index in range(len(attestation.aggregation_bits)):
                attestation.aggregation_bits[index] = False
        apply_operation_signature_profile(spec, profile, attestation)
        return attestation

    intent = profile.get("guide_intent")
    if intent in (None, "success", "previous_epoch_success"):
        return attestation
    if intent == "future_slot":
        attestation.data.slot = state.slot
    elif intent == "wrong_target_epoch":
        attestation.data.target.epoch = spec.Epoch(attestation.data.target.epoch + 1)
    elif intent == "bad_committee_index":
        attestation.committee_bits = spec.Bitvector[spec.MAX_COMMITTEES_PER_SLOT](
            [False] * spec.MAX_COMMITTEES_PER_SLOT
        )
        attestation.committee_bits[spec.MAX_COMMITTEES_PER_SLOT - 1] = True
    elif intent == "empty_aggregation":
        for index in range(len(attestation.aggregation_bits)):
            attestation.aggregation_bits[index] = False
    elif intent == "bad_signature":
        profile["bls_setting"] = 1
        attestation.signature = spec.BLSSignature(b"\x42" * 96)
    else:
        raise ValueError(f"Unsupported attestation guide intent: {intent}")
    return attestation


def apply_attestation_intent(spec, state, attestation, profile: dict[str, Any]) -> None:
    intent = profile.get("guide_intent")
    if intent in (None, "success", "previous_epoch_success"):
        return
    if intent == "future_slot":
        attestation.data.slot = state.slot
    elif intent == "wrong_target_epoch":
        attestation.data.target.epoch = spec.Epoch(attestation.data.target.epoch + 1)
    elif intent == "bad_committee_index":
        attestation.committee_bits = spec.Bitvector[spec.MAX_COMMITTEES_PER_SLOT](
            [False] * spec.MAX_COMMITTEES_PER_SLOT
        )
        attestation.committee_bits[spec.MAX_COMMITTEES_PER_SLOT - 1] = True
    elif intent == "empty_aggregation":
        for index in range(len(attestation.aggregation_bits)):
            attestation.aggregation_bits[index] = False
    elif intent == "bad_signature":
        profile["bls_setting"] = 1
        attestation.signature = spec.BLSSignature(b"\x42" * 96)
    else:
        raise ValueError(f"Unsupported attestation guide intent: {intent}")


def prepare_slashable_validator(spec, state, validator_index: int) -> None:
    validator = state.validators[validator_index]
    validator.slashed = False
    validator.activation_eligibility_epoch = spec.Epoch(0)
    validator.activation_epoch = spec.Epoch(0)
    validator.exit_epoch = spec.FAR_FUTURE_EPOCH
    validator.withdrawable_epoch = spec.FAR_FUTURE_EPOCH
    validator.effective_balance = spec.MIN_ACTIVATION_BALANCE
    state.balances[validator_index] = spec.MIN_ACTIVATION_BALANCE


def prepare_state_for_deposit(spec, state, profile: dict[str, Any]):
    state.pending_deposits = spec.List[spec.PendingDeposit, spec.PENDING_DEPOSITS_LIMIT]()
    if "input_profiles" in profile:
        if input_profile_shape(profile, "deposit_input", "branch_target") is not None:
            return build_deposit_for_intent(spec, state, profile)
        validator_index = len(state.validators)
        if input_profile_shape(
            profile,
            "deposit_input",
            "recipient_shape",
        ) == "TOP_UP_EXISTING_VALIDATOR":
            validator_index = VALIDATOR_INDEX
        deposit = prepare_state_and_deposit(
            spec,
            state,
            validator_index=validator_index,
            amount=spec.MIN_ACTIVATION_BALANCE,
            withdrawal_credentials=DEPOSIT_WITHDRAWAL_CREDENTIALS,
            signed=operation_input_shape(profile, "signature_shape") != "SIGNATURE_INVALID",
        )
        if operation_input_shape(profile, "proof_shape") == "PROOF_INVALID":
            state.eth1_data.deposit_root = b"\xff" * 32
        if operation_input_shape(profile, "signature_shape") == "SIGNATURE_INVALID":
            profile["bls_setting"] = 1
        return deposit

    intent = profile.get("guide_intent")
    return build_deposit_for_intent(spec, state, profile, intent)


def build_deposit_for_intent(spec, state, profile: dict[str, Any], intent: str | None = None):
    if intent is None:
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
    if "input_profiles" in profile:
        if input_profile_shape(profile, "bls_to_execution_change_input", "branch_target") is not None:
            return build_bls_to_execution_change_for_intent(
                spec,
                state,
                validator_index,
                withdrawal_pubkey,
                profile,
            )
        if input_profile_shape(
            profile,
            "bls_to_execution_change_input",
            "credential_shape",
        ) == "EXECUTION_CREDENTIALS":
            validator.withdrawal_credentials = (
                spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b"\x00" * 11 + SOURCE_ADDRESS
            )
        if input_profile_shape(
            profile,
            "bls_to_execution_change_input",
            "withdrawal_pubkey_relation",
        ) == "PUBKEY_MISMATCH":
            withdrawal_pubkey = pubkeys[-2 - validator_index]
        signed_address_change = get_signed_address_change(
            spec,
            state,
            validator_index=validator_index,
            withdrawal_pubkey=withdrawal_pubkey,
            to_execution_address=SOURCE_ADDRESS,
        )
        if operation_input_shape(profile, "lookup_shape") == "LOOKUP_MISSING":
            signed_address_change.message.validator_index = spec.ValidatorIndex(len(state.validators))
        if operation_input_shape(profile, "signature_shape") == "SIGNATURE_INVALID":
            profile["bls_setting"] = 1
            signed_address_change.signature = spec.BLSSignature(b"\x42" * 96)
        return signed_address_change

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


def build_bls_to_execution_change_for_intent(
    spec,
    state,
    validator_index: int,
    withdrawal_pubkey,
    profile: dict[str, Any],
):
    validator = state.validators[validator_index]
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


def operation_input_shape(
    profile: dict[str, Any],
    dimension: str,
    default: str | None = None,
) -> str | None:
    return input_profile_shape(profile, "operation_input", dimension, default)


def input_profile_shape(
    profile: dict[str, Any],
    profile_model: str,
    dimension: str,
    default: str | None = None,
) -> str | None:
    return profile.get("input_profiles", {}).get(profile_model, {}).get(dimension, default)


def has_input_profile_constraints(profile: dict[str, Any], profile_model: str) -> bool:
    return profile_model in profile.get("input_profile_constraints", {})


def apply_operation_signature_profile(spec, profile: dict[str, Any], signed_message) -> None:
    if operation_input_shape(profile, "signature_shape") == "SIGNATURE_INVALID":
        profile["bls_setting"] = 1
        signed_message.signature = spec.BLSSignature(b"\x42" * 96)


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
    if "input_profiles" in profile:
        prepare_state_for_profile(spec, state, validator_index, profile)
        if not has_input_profile_constraints(profile, "validator_state"):
            prepare_withdrawal_source(spec, state, validator_index, compounding=False)
        apply_queue_profile_for_validator(spec, state, validator_index, profile)
        if input_profile_shape(profile, "voluntary_exit_input", "branch_target") is not None:
            apply_voluntary_exit_intent(spec, state, validator_index, profile)
        return

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


def apply_voluntary_exit_intent(spec, state, validator_index: int, profile: dict[str, Any]) -> None:
    current_epoch = spec.get_current_epoch(state)
    validator = state.validators[validator_index]
    state.pending_partial_withdrawals = spec.List[
        spec.PendingPartialWithdrawal, spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
    ]()
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
    if (
        profile.get("guide_intent") == "future_epoch"
        or input_profile_shape(
            profile,
            "voluntary_exit_input",
            "exit_epoch_relation",
        ) == "EXIT_EPOCH_FUTURE"
    ):
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
    if input_profile_shape(
        profile,
        "withdrawal_request_input",
        "request_kind",
    ) == "PARTIAL_WITHDRAWAL_REQUEST":
        amount = spec.Gwei(1)
    elif profile["withdrawal_credential_type"] == "COMP" and not profile["exit_epoch_set"]:
        amount = spec.Gwei(1)
    return spec.WithdrawalRequest(
        source_address=SOURCE_ADDRESS,
        validator_pubkey=state.validators[validator_index].pubkey,
        amount=amount,
    )


def withdrawal_branch_target(profile: dict[str, Any]) -> str | None:
    return input_profile_shape(profile, "withdrawal_request_input", "branch_target")


def apply_withdrawal_profile(spec, state, validator_index: int, withdrawal_request, profile) -> None:
    if "input_profiles" not in profile:
        return
    if apply_withdrawal_branch_target(spec, state, validator_index, withdrawal_request, profile):
        return

    if input_profile_shape(
        profile,
        "withdrawal_request_input",
        "request_kind",
    ) == "PARTIAL_WITHDRAWAL_REQUEST":
        prepare_withdrawal_source(spec, state, validator_index, compounding=True)
        withdrawal_request.source_address = SOURCE_ADDRESS
        withdrawal_request.validator_pubkey = state.validators[validator_index].pubkey
        withdrawal_request.amount = spec.Gwei(1)
    if operation_input_shape(profile, "lookup_shape") == "LOOKUP_MISSING":
        withdrawal_request.validator_pubkey = b"\xff" * 48
    if operation_input_shape(profile, "source_address_shape") == "SOURCE_ADDRESS_INVALID":
        withdrawal_request.source_address = invalid_source_address()
    queue_profile = profile.get("input_profiles", {}).get("queue", {})
    if queue_profile.get("pending_request") == "REQUEST_WITHDRAWAL":
        state.pending_partial_withdrawals.append(
            spec.PendingPartialWithdrawal(
                validator_index=spec.ValidatorIndex(validator_index),
                amount=spec.Gwei(1),
                withdrawable_epoch=spec.Epoch(spec.get_current_epoch(state) + 1),
            )
        )


def apply_withdrawal_branch_target(
    spec,
    state,
    validator_index: int,
    withdrawal_request,
    profile: dict[str, Any],
) -> bool:
    branch_target = withdrawal_branch_target(profile)
    if branch_target is None:
        return False

    partial_request_targets = {
        "WITHDRAWAL_QUEUE_FULL_PARTIAL",
        "WITHDRAWAL_PARTIAL_CONDITIONS_NOT_MET",
        "WITHDRAWAL_PARTIAL_SUCCESS",
    }
    is_partial_request = branch_target in partial_request_targets
    prepare_withdrawal_source(spec, state, validator_index, compounding=is_partial_request)
    withdrawal_request.source_address = SOURCE_ADDRESS
    withdrawal_request.validator_pubkey = state.validators[validator_index].pubkey
    withdrawal_request.amount = (
        spec.Gwei(1) if is_partial_request else spec.FULL_EXIT_REQUEST_AMOUNT
    )

    current_epoch = spec.get_current_epoch(state)
    validator = state.validators[validator_index]

    if branch_target == "WITHDRAWAL_QUEUE_FULL_PARTIAL":
        fill_pending_partial_withdrawals(spec, state, validator_index)
    elif branch_target == "WITHDRAWAL_PUBKEY_MISSING":
        withdrawal_request.validator_pubkey = b"\xff" * 48
    elif branch_target == "WITHDRAWAL_BAD_SOURCE_ADDRESS":
        withdrawal_request.source_address = invalid_source_address()
    elif branch_target == "WITHDRAWAL_SOURCE_INACTIVE":
        validator.activation_epoch = spec.FAR_FUTURE_EPOCH
    elif branch_target == "WITHDRAWAL_SOURCE_EXITING":
        validator.exit_epoch = spec.Epoch(current_epoch + 1)
    elif branch_target == "WITHDRAWAL_NOT_ACTIVE_LONG_ENOUGH":
        validator.activation_epoch = current_epoch
    elif branch_target == "WITHDRAWAL_FULL_EXIT_PENDING_WITHDRAWAL":
        state.pending_partial_withdrawals.append(
            spec.PendingPartialWithdrawal(
                validator_index=validator_index,
                amount=spec.Gwei(1),
                withdrawable_epoch=spec.Epoch(current_epoch + 1),
            )
        )
    elif branch_target == "WITHDRAWAL_FULL_EXIT_SUCCESS":
        return True
    elif branch_target == "WITHDRAWAL_PARTIAL_CONDITIONS_NOT_MET":
        set_eth1_withdrawal_credential_with_balance(
            spec,
            state,
            validator_index,
            effective_balance=spec.MIN_ACTIVATION_BALANCE,
            balance=spec.MIN_ACTIVATION_BALANCE,
            address=SOURCE_ADDRESS,
        )
    elif branch_target == "WITHDRAWAL_PARTIAL_SUCCESS":
        return True
    else:
        raise ValueError(f"Unsupported withdrawal branch target: {branch_target}")
    return True


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


def apply_queue_profile_for_validator(
    spec,
    state,
    validator_index: int,
    profile: dict[str, Any],
) -> None:
    queue_profile = profile.get("input_profiles", {}).get("queue", {})
    current_epoch = spec.get_current_epoch(state)
    partial_withdrawals_shape = queue_profile.get("pending_partial_withdrawals")
    if partial_withdrawals_shape == "EMPTY":
        state.pending_partial_withdrawals = spec.List[
            spec.PendingPartialWithdrawal, spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
        ]()
    elif partial_withdrawals_shape == "NONEMPTY":
        state.pending_partial_withdrawals = spec.List[
            spec.PendingPartialWithdrawal, spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
        ]()
        state.pending_partial_withdrawals.append(
            spec.PendingPartialWithdrawal(
                validator_index=spec.ValidatorIndex(validator_index),
                amount=spec.Gwei(1),
                withdrawable_epoch=spec.Epoch(current_epoch + 1),
            )
        )
    elif partial_withdrawals_shape == "FULL":
        fill_pending_partial_withdrawals(spec, state, validator_index)
    elif partial_withdrawals_shape is not None:
        raise ValueError(
            f"Unsupported pending partial withdrawals profile shape: {partial_withdrawals_shape}"
        )

    consolidations_shape = queue_profile.get("pending_consolidations")
    if consolidations_shape == "EMPTY":
        state.pending_consolidations = spec.List[
            spec.PendingConsolidation, spec.PENDING_CONSOLIDATIONS_LIMIT
        ]()
    elif consolidations_shape == "NONEMPTY":
        state.pending_consolidations = spec.List[
            spec.PendingConsolidation, spec.PENDING_CONSOLIDATIONS_LIMIT
        ]()
        add_pending_consolidation(spec, state, validator_index, TARGET_VALIDATOR_INDEX)
    elif consolidations_shape == "FULL":
        fill_pending_consolidations(spec, state, validator_index, TARGET_VALIDATOR_INDEX)
    elif consolidations_shape is not None:
        raise ValueError(f"Unsupported pending consolidations profile shape: {consolidations_shape}")


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


def apply_consolidation_profile(
    spec,
    state,
    source_index: int,
    target_index: int,
    consolidation_request,
    profile,
) -> None:
    if "input_profiles" not in profile:
        return
    if apply_consolidation_branch_target(
        spec,
        state,
        source_index,
        target_index,
        consolidation_request,
        profile,
    ):
        return

    request_profile = profile.get("input_profiles", {}).get("consolidation_request_input", {})
    if request_profile.get("request_kind") == "SWITCH_TO_COMPOUNDING_REQUEST":
        prepare_switch_to_compounding_source(spec, state, source_index)
        consolidation_request.source_pubkey = state.validators[source_index].pubkey
        consolidation_request.target_pubkey = state.validators[source_index].pubkey
        consolidation_request.source_address = SOURCE_ADDRESS
    elif request_profile.get("request_kind") == "CONSOLIDATION_REQUEST":
        prepare_consolidation_source(spec, state, source_index)
        prepare_target_for_consolidation(spec, state, target_index)
        consolidation_request.source_pubkey = state.validators[source_index].pubkey
        consolidation_request.target_pubkey = state.validators[target_index].pubkey
        consolidation_request.source_address = SOURCE_ADDRESS

    if operation_input_shape(profile, "lookup_shape") == "LOOKUP_MISSING":
        consolidation_request.source_pubkey = b"\xff" * 48
    if operation_input_shape(profile, "source_address_shape") == "SOURCE_ADDRESS_INVALID":
        consolidation_request.source_address = invalid_source_address()
    if operation_input_shape(profile, "source_target_relation") == "SOURCE_TARGET_SAME":
        consolidation_request.target_pubkey = state.validators[source_index].pubkey
    if request_profile.get("target_lookup_shape") == "TARGET_MISSING":
        consolidation_request.target_pubkey = b"\xff" * 48

    source_activity_shape = request_profile.get("source_activity_shape")
    if source_activity_shape == "SOURCE_INACTIVE":
        state.validators[source_index].activation_epoch = spec.FAR_FUTURE_EPOCH
    elif source_activity_shape == "SOURCE_EXITING":
        state.validators[source_index].exit_epoch = spec.Epoch(spec.get_current_epoch(state) + 1)
    elif source_activity_shape == "SOURCE_NOT_ACTIVE_LONG_ENOUGH":
        state.validators[source_index].activation_epoch = spec.get_current_epoch(state)

    target_activity_shape = request_profile.get("target_activity_shape")
    if target_activity_shape == "TARGET_INACTIVE":
        state.validators[target_index].activation_epoch = spec.FAR_FUTURE_EPOCH
    elif target_activity_shape == "TARGET_EXITING":
        state.validators[target_index].exit_epoch = spec.Epoch(spec.get_current_epoch(state) + 1)

    if request_profile.get("target_credential_shape") == "TARGET_ETH1":
        set_eth1_withdrawal_credential_with_balance(
            spec,
            state,
            target_index,
            effective_balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
            balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
            address=TARGET_ADDRESS,
        )

    if request_profile.get("churn_shape") == "CHURN_TOO_LOW":
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

    queue_profile = profile.get("input_profiles", {}).get("queue", {})
    if queue_profile.get("pending_request") == "REQUEST_CONSOLIDATION":
        state.pending_consolidations.append(
            spec.PendingConsolidation(
                source_index=spec.ValidatorIndex(source_index),
                target_index=spec.ValidatorIndex(target_index),
            )
        )
    elif queue_profile.get("pending_request") == "REQUEST_WITHDRAWAL":
        state.pending_partial_withdrawals.append(
            spec.PendingPartialWithdrawal(
                validator_index=spec.ValidatorIndex(source_index),
                amount=spec.Gwei(1),
                withdrawable_epoch=spec.Epoch(spec.get_current_epoch(state) + 1),
            )
        )


def consolidation_branch_target(profile: dict[str, Any]) -> str | None:
    return input_profile_shape(profile, "consolidation_request_input", "branch_target")


def apply_consolidation_branch_target(
    spec,
    state,
    source_index: int,
    target_index: int,
    consolidation_request,
    profile: dict[str, Any],
) -> bool:
    branch_target = consolidation_branch_target(profile)
    if branch_target is None:
        return False

    state.pending_consolidations = spec.List[
        spec.PendingConsolidation, spec.PENDING_CONSOLIDATIONS_LIMIT
    ]()
    state.pending_partial_withdrawals = spec.List[
        spec.PendingPartialWithdrawal, spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
    ]()

    if branch_target.startswith("CONSOLIDATION_SWITCH_"):
        prepare_switch_to_compounding_source(spec, state, source_index)
        consolidation_request.source_address = SOURCE_ADDRESS
        consolidation_request.source_pubkey = state.validators[source_index].pubkey
        consolidation_request.target_pubkey = state.validators[source_index].pubkey
        current_epoch = spec.get_current_epoch(state)

        if branch_target == "CONSOLIDATION_SWITCH_SUCCESS":
            return True
        if branch_target == "CONSOLIDATION_SWITCH_PUBKEY_MISSING":
            consolidation_request.source_pubkey = b"\xff" * 48
            consolidation_request.target_pubkey = b"\xff" * 48
        elif branch_target == "CONSOLIDATION_SWITCH_BAD_SOURCE_ADDRESS":
            consolidation_request.source_address = invalid_source_address()
        elif branch_target == "CONSOLIDATION_SWITCH_SOURCE_INACTIVE":
            state.validators[source_index].activation_epoch = spec.FAR_FUTURE_EPOCH
        elif branch_target == "CONSOLIDATION_SWITCH_SOURCE_EXITING":
            state.validators[source_index].exit_epoch = spec.Epoch(current_epoch + 1)
        else:
            raise ValueError(f"Unsupported consolidation branch target: {branch_target}")
        return True

    prepare_consolidation_source(spec, state, source_index)
    prepare_target_for_consolidation(spec, state, target_index)
    consolidation_request.source_address = SOURCE_ADDRESS
    consolidation_request.source_pubkey = state.validators[source_index].pubkey
    consolidation_request.target_pubkey = state.validators[target_index].pubkey
    current_epoch = spec.get_current_epoch(state)

    if branch_target == "CONSOLIDATION_SOURCE_EQUALS_TARGET":
        consolidation_request.target_pubkey = state.validators[source_index].pubkey
    elif branch_target == "CONSOLIDATION_QUEUE_FULL":
        fill_pending_consolidations(spec, state, source_index, target_index)
    elif branch_target == "CONSOLIDATION_CHURN_TOO_LOW":
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
    else:
        prepare_churn_helper(spec, state)
        if branch_target == "CONSOLIDATION_SOURCE_MISSING":
            consolidation_request.source_pubkey = b"\xff" * 48
        elif branch_target == "CONSOLIDATION_TARGET_MISSING":
            consolidation_request.target_pubkey = b"\xff" * 48
        elif branch_target == "CONSOLIDATION_BAD_SOURCE_ADDRESS":
            consolidation_request.source_address = invalid_source_address()
        elif branch_target == "CONSOLIDATION_TARGET_NOT_COMPOUNDING":
            set_eth1_withdrawal_credential_with_balance(
                spec,
                state,
                target_index,
                effective_balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
                balance=spec.MAX_EFFECTIVE_BALANCE_ELECTRA,
                address=TARGET_ADDRESS,
            )
        elif branch_target == "CONSOLIDATION_SOURCE_INACTIVE":
            state.validators[source_index].activation_epoch = spec.FAR_FUTURE_EPOCH
        elif branch_target == "CONSOLIDATION_TARGET_INACTIVE":
            state.validators[target_index].activation_epoch = spec.FAR_FUTURE_EPOCH
        elif branch_target == "CONSOLIDATION_SOURCE_EXITING":
            state.validators[source_index].exit_epoch = spec.Epoch(current_epoch + 1)
        elif branch_target == "CONSOLIDATION_TARGET_EXITING":
            state.validators[target_index].exit_epoch = spec.Epoch(current_epoch + 1)
        elif branch_target == "CONSOLIDATION_SOURCE_NOT_ACTIVE_LONG_ENOUGH":
            state.validators[source_index].activation_epoch = current_epoch
        elif branch_target == "CONSOLIDATION_SOURCE_PENDING_WITHDRAWAL":
            state.pending_partial_withdrawals.append(
                spec.PendingPartialWithdrawal(
                    validator_index=source_index,
                    amount=spec.Gwei(1),
                    withdrawable_epoch=spec.Epoch(current_epoch + 1),
                )
            )
        elif branch_target == "CONSOLIDATION_SUCCESS":
            return True
        else:
            raise ValueError(f"Unsupported consolidation branch target: {branch_target}")
    return True


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
    if profile.get("profile_driven"):
        prepare_state_for_pending_deposits_profiles(spec, state, profile)
        return

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


def prepare_state_for_pending_deposits_profiles(
    spec,
    state,
    profile: dict[str, Any],
) -> None:
    apply_epoch_boundary_profile(spec, state, profile)
    state.deposit_requests_start_index = state.eth1_deposit_index
    state.deposit_balance_to_consume = spec.Gwei(0)
    state.pending_deposits = spec.List[spec.PendingDeposit, spec.PENDING_DEPOSITS_LIMIT]()
    pending_deposits_profile = profile.get("input_profiles", {}).get("pending_deposits_input", {})

    queue_profile = profile.get("input_profiles", {}).get("queue", {})
    pending_deposits_shape = queue_profile.get("pending_deposits", "EMPTY")
    if pending_deposits_shape == "EMPTY":
        pass
    elif pending_deposits_shape == "NONEMPTY":
        add_pending_deposit(spec, state, VALIDATOR_INDEX, spec.EFFECTIVE_BALANCE_INCREMENT)
    elif pending_deposits_shape == "FULL":
        for validator_index in range(spec.MAX_PENDING_DEPOSITS_PER_EPOCH + 1):
            add_pending_deposit(spec, state, validator_index, spec.EFFECTIVE_BALANCE_INCREMENT)
    else:
        raise ValueError(f"Unsupported pending deposits profile shape: {pending_deposits_shape}")

    if pending_deposits_profile:
        state.pending_deposits = spec.List[spec.PendingDeposit, spec.PENDING_DEPOSITS_LIMIT]()
        validator_index = VALIDATOR_INDEX
        withdrawal_credentials = None
        deposit_kind = pending_deposits_profile.get("deposit_kind")
        if deposit_kind == "NEW_VALIDATOR":
            validator_index = len(state.validators)
            withdrawal_credentials = DEPOSIT_WITHDRAWAL_CREDENTIALS
        elif deposit_kind == "EXITING_VALIDATOR":
            spec.initiate_validator_exit(state, spec.ValidatorIndex(VALIDATOR_INDEX))
        elif deposit_kind == "WITHDRAWABLE_VALIDATOR":
            spec.initiate_validator_exit(state, spec.ValidatorIndex(VALIDATOR_INDEX))
            state.slot = spec.compute_start_slot_at_epoch(
                spec.Epoch(state.validators[VALIDATOR_INDEX].withdrawable_epoch + 1)
            )

        amount = spec.EFFECTIVE_BALANCE_INCREMENT
        if pending_deposits_profile.get("churn_shape") == "CHURN_LIMIT_REACHED":
            amount = spec.Gwei(spec.get_activation_exit_churn_limit(state) + 1)

        slot = None
        if pending_deposits_profile.get("finality_shape") == "NOT_FINALIZED":
            slot = spec.Slot(state.slot + 1)
        add_pending_deposit(
            spec,
            state,
            validator_index,
            amount,
            slot=slot,
            withdrawal_credentials=withdrawal_credentials,
        )

        if pending_deposits_profile.get("bridge_state") == "ETH1_BRIDGE_PENDING":
            state.deposit_requests_start_index = spec.uint64(state.eth1_deposit_index + 1)
            state.finalized_checkpoint.epoch = spec.Epoch(max(1, int(state.finalized_checkpoint.epoch)))

    pending_request_shape = queue_profile.get("pending_request", "REQUEST_NONE")
    if pending_request_shape == "REQUEST_WITHDRAWAL":
        state.pending_partial_withdrawals = spec.List[
            spec.PendingPartialWithdrawal, spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
        ]()
        state.pending_partial_withdrawals.append(
            spec.PendingPartialWithdrawal(
                validator_index=spec.ValidatorIndex(VALIDATOR_INDEX),
                amount=spec.EFFECTIVE_BALANCE_INCREMENT,
                withdrawable_epoch=spec.Epoch(spec.get_current_epoch(state) + 1),
            )
        )
    elif pending_request_shape == "REQUEST_CONSOLIDATION":
        prepare_pending_consolidation_pair(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
        state.pending_consolidations = spec.List[
            spec.PendingConsolidation, spec.PENDING_CONSOLIDATIONS_LIMIT
        ]()
        add_pending_consolidation(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
    elif pending_request_shape != "REQUEST_NONE":
        raise ValueError(f"Unsupported pending request profile shape: {pending_request_shape}")


def apply_epoch_boundary_profile(spec, state, profile: dict[str, Any]) -> None:
    epoch_boundary_shape = (
        profile.get("input_profiles", {})
        .get("epoch_boundary", {})
        .get("epoch_boundary_shape", "NORMAL")
    )
    if epoch_boundary_shape == "GENESIS":
        state.slot = spec.compute_start_slot_at_epoch(spec.GENESIS_EPOCH)
    elif epoch_boundary_shape == "NORMAL":
        state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(2))
    elif epoch_boundary_shape == "PERIOD_BOUNDARY":
        state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(spec.EPOCHS_PER_SYNC_COMMITTEE_PERIOD))
    elif epoch_boundary_shape == "NON_BOUNDARY":
        state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(2)) + 1
    else:
        raise ValueError(f"Unsupported epoch boundary profile shape: {epoch_boundary_shape}")


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
    if profile.get("profile_driven"):
        prepare_state_for_pending_consolidations_profiles(spec, state, profile)
        return

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


def prepare_state_for_pending_consolidations_profiles(
    spec,
    state,
    profile: dict[str, Any],
) -> None:
    state.pending_consolidations = spec.List[
        spec.PendingConsolidation, spec.PENDING_CONSOLIDATIONS_LIMIT
    ]()
    state.pending_deposits = spec.List[spec.PendingDeposit, spec.PENDING_DEPOSITS_LIMIT]()
    prepare_pending_consolidation_pair(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
    pending_consolidations_profile = (
        profile.get("input_profiles", {}).get("pending_consolidations_input", {})
    )

    queue_profile = profile.get("input_profiles", {}).get("queue", {})
    pending_consolidations_shape = queue_profile.get("pending_consolidations", "EMPTY")
    if pending_consolidations_shape == "EMPTY":
        pass
    elif pending_consolidations_shape == "NONEMPTY":
        add_pending_consolidation(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
    elif pending_consolidations_shape == "FULL":
        for _ in range(spec.PENDING_CONSOLIDATIONS_LIMIT):
            add_pending_consolidation(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
    else:
        raise ValueError(
            f"Unsupported pending consolidations profile shape: {pending_consolidations_shape}"
        )

    pending_request_shape = queue_profile.get("pending_request", "REQUEST_NONE")
    if pending_request_shape == "REQUEST_CONSOLIDATION" and len(state.pending_consolidations) == 0:
        add_pending_consolidation(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)
    elif pending_request_shape == "REQUEST_WITHDRAWAL":
        state.pending_partial_withdrawals = spec.List[
            spec.PendingPartialWithdrawal, spec.PENDING_PARTIAL_WITHDRAWALS_LIMIT
        ]()
        state.pending_partial_withdrawals.append(
            spec.PendingPartialWithdrawal(
                validator_index=spec.ValidatorIndex(VALIDATOR_INDEX),
                amount=spec.EFFECTIVE_BALANCE_INCREMENT,
                withdrawable_epoch=spec.Epoch(spec.get_current_epoch(state) + 1),
            )
        )
    elif pending_request_shape not in ("REQUEST_NONE", "REQUEST_CONSOLIDATION"):
        raise ValueError(f"Unsupported pending request profile shape: {pending_request_shape}")

    if pending_consolidations_profile:
        state.pending_consolidations = spec.List[
            spec.PendingConsolidation, spec.PENDING_CONSOLIDATIONS_LIMIT
        ]()
        add_pending_consolidation(spec, state, VALIDATOR_INDEX, TARGET_VALIDATOR_INDEX)

        source_shape = pending_consolidations_profile.get("source_shape")
        if source_shape == "SOURCE_NOT_WITHDRAWABLE":
            state.validators[VALIDATOR_INDEX].withdrawable_epoch = spec.Epoch(
                spec.get_current_epoch(state) + 2
            )
        elif source_shape == "SOURCE_SLASHED":
            state.validators[VALIDATOR_INDEX].slashed = True

        balance_shape = pending_consolidations_profile.get("balance_shape")
        if balance_shape == "BALANCE_LESS_THAN_EFFECTIVE_BALANCE":
            state.balances[VALIDATOR_INDEX] = spec.Gwei(
                state.validators[VALIDATOR_INDEX].effective_balance
                - spec.EFFECTIVE_BALANCE_INCREMENT // 8
            )
        elif balance_shape == "BALANCE_GREATER_THAN_EFFECTIVE_BALANCE":
            state.balances[VALIDATOR_INDEX] = spec.Gwei(
                state.validators[VALIDATOR_INDEX].effective_balance
                + spec.EFFECTIVE_BALANCE_INCREMENT // 8
            )

        if pending_consolidations_profile.get("queue_shape") == "BLOCKED_AFTER_PROCESSED":
            prepare_pending_consolidation_pair(spec, state, HELPER_VALIDATOR_INDEX, 3)
            add_pending_consolidation(spec, state, HELPER_VALIDATOR_INDEX, 3)
            state.validators[HELPER_VALIDATOR_INDEX].withdrawable_epoch = spec.Epoch(
                spec.get_current_epoch(state) + 2
            )


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


def prepare_state_for_registry_updates(spec, state, profile: dict[str, Any]) -> None:
    current_epoch = spec.Epoch(max(2, spec.get_current_epoch(state)))
    state.slot = spec.compute_start_slot_at_epoch(current_epoch)
    index = VALIDATOR_INDEX
    validator = state.validators[index]
    validator.slashed = False
    validator.activation_eligibility_epoch = spec.Epoch(0)
    validator.activation_epoch = spec.Epoch(0)
    validator.exit_epoch = spec.FAR_FUTURE_EPOCH
    validator.withdrawable_epoch = spec.FAR_FUTURE_EPOCH
    set_eth1_withdrawal_credential_with_balance(
        spec,
        state,
        index,
        effective_balance=spec.MIN_ACTIVATION_BALANCE,
        balance=spec.MIN_ACTIVATION_BALANCE,
        address=SOURCE_ADDRESS,
    )

    intent = profile.get("guide_intent")
    if intent in (None, "no_change"):
        return
    if intent == "activation_queue":
        validator.activation_eligibility_epoch = spec.FAR_FUTURE_EPOCH
        validator.activation_epoch = spec.FAR_FUTURE_EPOCH
    elif intent == "ejection":
        validator.effective_balance = spec.config.EJECTION_BALANCE
        state.balances[index] = spec.config.EJECTION_BALANCE
    elif intent == "activation":
        validator.activation_eligibility_epoch = spec.Epoch(0)
        validator.activation_epoch = spec.FAR_FUTURE_EPOCH
        state.finalized_checkpoint.epoch = current_epoch
    else:
        raise ValueError(f"Unsupported registry updates guide intent: {intent}")


def prepare_state_for_slashings(spec, state, profile: dict[str, Any]) -> None:
    current_epoch = spec.Epoch(spec.EPOCHS_PER_SLASHINGS_VECTOR // 2 + 2)
    state.slot = spec.compute_start_slot_at_epoch(current_epoch)
    state.slashings = spec.Vector[spec.Gwei, spec.EPOCHS_PER_SLASHINGS_VECTOR](
        [spec.Gwei(0)] * spec.EPOCHS_PER_SLASHINGS_VECTOR
    )

    index = VALIDATOR_INDEX
    validator = state.validators[index]
    validator.activation_eligibility_epoch = spec.Epoch(0)
    validator.activation_epoch = spec.Epoch(0)
    validator.exit_epoch = spec.Epoch(0)
    validator.withdrawable_epoch = spec.Epoch(
        current_epoch + spec.EPOCHS_PER_SLASHINGS_VECTOR // 2
    )
    validator.effective_balance = spec.MIN_ACTIVATION_BALANCE
    state.balances[index] = spec.MIN_ACTIVATION_BALANCE

    intent = profile.get("guide_intent")
    if intent in (None, "penalty_applied"):
        validator.slashed = True
        state.slashings[0] = spec.Gwei(spec.get_total_active_balance(state))
    elif intent == "no_slashed_validators":
        validator.slashed = False
        state.slashings[0] = spec.Gwei(spec.get_total_active_balance(state))
    elif intent == "wrong_withdrawable_epoch":
        validator.slashed = True
        validator.withdrawable_epoch = spec.Epoch(validator.withdrawable_epoch + 1)
        state.slashings[0] = spec.Gwei(spec.get_total_active_balance(state))
    elif intent == "zero_slashing_balance":
        validator.slashed = True
    else:
        raise ValueError(f"Unsupported slashings guide intent: {intent}")


def prepare_state_for_justification_and_finalization(
    spec,
    state,
    profile: dict[str, Any],
) -> None:
    if profile.get("profile_driven"):
        prepare_state_for_participation_profiles(spec, state, profile, for_finality=True)
        return

    intent = profile.get("guide_intent")
    if intent == "genesis_skip":
        state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(1))
        return
    if intent == "finalize_234":
        prepare_state_for_finalization_pattern(
            spec,
            state,
            current_epoch=spec.Epoch(5),
            previous_justified_epoch=spec.Epoch(2),
            current_justified_epoch=spec.Epoch(3),
            pre_shift_justification_bits=[1, 2],
            supported_epochs=[spec.Epoch(4)],
        )
        return
    if intent == "finalize_23":
        prepare_state_for_finalization_pattern(
            spec,
            state,
            current_epoch=spec.Epoch(4),
            previous_justified_epoch=spec.Epoch(2),
            current_justified_epoch=spec.Epoch(2),
            pre_shift_justification_bits=[1],
            supported_epochs=[spec.Epoch(3)],
        )
        return
    if intent == "finalize_123":
        prepare_state_for_finalization_pattern(
            spec,
            state,
            current_epoch=spec.Epoch(6),
            previous_justified_epoch=spec.Epoch(1),
            current_justified_epoch=spec.Epoch(4),
            pre_shift_justification_bits=[1],
            supported_epochs=[spec.Epoch(5), spec.Epoch(6)],
        )
        return

    current_epoch = spec.Epoch(3)
    transition_to(spec, state, spec.compute_start_slot_at_epoch(current_epoch) + 1)
    previous_epoch = spec.get_previous_epoch(state)
    current_root = spec.get_block_root(state, current_epoch)
    previous_root = spec.get_block_root(state, previous_epoch)
    old_current = spec.Checkpoint(epoch=spec.Epoch(1), root=previous_root)
    state.previous_justified_checkpoint = old_current
    state.current_justified_checkpoint = old_current
    state.justification_bits = spec.Bitvector[spec.JUSTIFICATION_BITS_LENGTH]()

    if intent in (None, "current_justified", "finalize_current"):
        set_epoch_target_participation(spec, state, current_epoch)
    elif intent == "previous_justified":
        set_epoch_target_participation(spec, state, previous_epoch)
    elif intent == "poor_support":
        state.current_epoch_participation[VALIDATOR_INDEX] = spec.ParticipationFlags(
            2**spec.TIMELY_TARGET_FLAG_INDEX
        )
    else:
        raise ValueError(f"Unsupported justification/finalization guide intent: {intent}")

    if intent == "finalize_current":
        state.justification_bits[0] = True
        state.current_justified_checkpoint = spec.Checkpoint(
            epoch=spec.Epoch(2),
            root=current_root,
        )


def set_epoch_target_participation(spec, state, epoch) -> None:
    if epoch == spec.get_current_epoch(state):
        participation = state.current_epoch_participation
    elif epoch == spec.get_previous_epoch(state):
        participation = state.previous_epoch_participation
    else:
        raise ValueError(f"Cannot set participation for epoch {epoch}")

    target_flag = spec.ParticipationFlags(2**spec.TIMELY_TARGET_FLAG_INDEX)
    for index in spec.get_active_validator_indices(state, epoch):
        participation[index] |= target_flag


def prepare_state_for_finalization_pattern(
    spec,
    state,
    *,
    current_epoch,
    previous_justified_epoch,
    current_justified_epoch,
    pre_shift_justification_bits: list[int],
    supported_epochs: list,
) -> None:
    transition_to(spec, state, spec.compute_start_slot_at_epoch(current_epoch) + 1)
    state.previous_justified_checkpoint = spec.Checkpoint(
        epoch=previous_justified_epoch,
        root=spec.get_block_root(state, previous_justified_epoch),
    )
    state.current_justified_checkpoint = spec.Checkpoint(
        epoch=current_justified_epoch,
        root=spec.get_block_root(state, current_justified_epoch),
    )
    state.justification_bits = spec.Bitvector[spec.JUSTIFICATION_BITS_LENGTH]()
    for bit_index in pre_shift_justification_bits:
        state.justification_bits[bit_index] = True
    for epoch in supported_epochs:
        set_epoch_target_participation(spec, state, epoch)


def prepare_state_for_inactivity_updates(spec, state, profile: dict[str, Any]) -> None:
    if profile.get("profile_driven"):
        prepare_state_for_participation_profiles(spec, state, profile)
        state.inactivity_scores[VALIDATOR_INDEX] = spec.uint64(5)
        return

    intent = profile.get("guide_intent")
    if intent == "genesis_skip":
        state.slot = spec.compute_start_slot_at_epoch(spec.GENESIS_EPOCH)
        state.inactivity_scores[VALIDATOR_INDEX] = spec.uint64(7)
        return

    setup_participation_epoch_state(spec, state, leak=intent == "non_participating_leak")
    state.inactivity_scores[VALIDATOR_INDEX] = spec.uint64(5)

    if intent in (None, "participating_recovery"):
        set_epoch_target_participation(spec, state, spec.get_previous_epoch(state))
    elif intent in ("non_participating_no_leak", "non_participating_leak"):
        clear_participation(spec, state)
    else:
        raise ValueError(f"Unsupported inactivity updates guide intent: {intent}")


def prepare_state_for_rewards_and_penalties(spec, state, profile: dict[str, Any]) -> None:
    if profile.get("profile_driven"):
        prepare_state_for_participation_profiles(spec, state, profile)
        state.inactivity_scores[VALIDATOR_INDEX] = spec.uint64(16)
        return

    intent = profile.get("guide_intent")
    if intent == "genesis_skip":
        state.slot = spec.compute_start_slot_at_epoch(spec.GENESIS_EPOCH)
        return

    setup_participation_epoch_state(
        spec,
        state,
        leak=intent in ("inactivity_leak_penalty", "inactivity_leak_full_participation"),
    )
    state.inactivity_scores[VALIDATOR_INDEX] = spec.uint64(16)
    if intent in (None, "full_participation_reward", "inactivity_leak_full_participation"):
        set_full_participation_flags(spec, state, previous=True, current=False)
    elif intent in ("empty_participation_penalty", "inactivity_leak_penalty"):
        clear_participation(spec, state)
    else:
        raise ValueError(f"Unsupported rewards and penalties guide intent: {intent}")


def prepare_state_for_participation_flag_updates(spec, state, profile: dict[str, Any]) -> None:
    if profile.get("profile_driven"):
        prepare_state_for_participation_profiles(spec, state, profile)
        return

    transition_to(spec, state, spec.compute_start_slot_at_epoch(spec.Epoch(2)))
    clear_participation(spec, state)
    intent = profile.get("guide_intent")
    if intent in (None, "all_zero"):
        return
    if intent == "current_filled":
        set_full_participation_flags(spec, state, previous=False, current=True)
    elif intent == "previous_filled":
        set_full_participation_flags(spec, state, previous=True, current=False)
    else:
        raise ValueError(f"Unsupported participation flag updates guide intent: {intent}")


def prepare_state_for_participation_profiles(
    spec,
    state,
    profile: dict[str, Any],
    *,
    for_finality: bool = False,
) -> None:
    participation_profile = profile.get("input_profiles", {}).get("participation", {})
    epoch_profile = profile.get("input_profiles", {}).get("epoch_boundary", {})
    epoch_boundary_shape = epoch_profile.get("epoch_boundary_shape")
    if epoch_boundary_shape == "GENESIS":
        state.slot = spec.compute_start_slot_at_epoch(spec.GENESIS_EPOCH)
        clear_participation(spec, state)
        return

    leak = bool(participation_profile.get("inactivity_leak", False))
    current_epoch = spec.Epoch(spec.MIN_EPOCHS_TO_INACTIVITY_PENALTY + 2 if leak else 3)
    transition_to(spec, state, spec.compute_start_slot_at_epoch(current_epoch) + 1)
    finalized_epoch = spec.GENESIS_EPOCH if leak else spec.Epoch(current_epoch - 1)
    state.finalized_checkpoint = spec.Checkpoint(
        epoch=finalized_epoch,
        root=spec.get_block_root(state, finalized_epoch),
    )
    state.previous_justified_checkpoint = state.finalized_checkpoint
    state.current_justified_checkpoint = state.finalized_checkpoint
    state.justification_bits = spec.Bitvector[spec.JUSTIFICATION_BITS_LENGTH]()
    clear_participation(spec, state)

    apply_participation_shape(spec, state, participation_profile)
    if for_finality:
        apply_finality_shape(spec, state, participation_profile)


def apply_participation_shape(spec, state, participation_profile: dict[str, Any]) -> None:
    participation_shape = participation_profile.get("participation_shape", "PARTICIPATION_NONE")
    previous_epoch = spec.get_previous_epoch(state)
    if participation_shape == "PARTICIPATION_NONE":
        return
    if participation_shape == "PARTICIPATION_TARGET_ONLY":
        set_epoch_target_participation(spec, state, previous_epoch)
        return
    if participation_shape == "PARTICIPATION_FULL":
        set_full_participation_flags(spec, state, previous=True, current=True)
        return
    if participation_shape == "PARTICIPATION_POOR_SUPPORT":
        state.previous_epoch_participation[VALIDATOR_INDEX] = spec.ParticipationFlags(
            2**spec.TIMELY_TARGET_FLAG_INDEX
        )
        return
    raise ValueError(f"Unsupported participation profile shape: {participation_shape}")


def apply_finality_shape(spec, state, participation_profile: dict[str, Any]) -> None:
    finality_shape = participation_profile.get("finality_shape", "FINALITY_NONE")
    current_epoch = spec.get_current_epoch(state)
    previous_epoch = spec.get_previous_epoch(state)
    if finality_shape == "FINALITY_NONE":
        return
    if finality_shape == "FINALITY_PREVIOUS_JUSTIFIED":
        state.current_justified_checkpoint = spec.Checkpoint(
            epoch=spec.Epoch(previous_epoch - 1),
            root=spec.get_block_root(state, spec.Epoch(previous_epoch - 1)),
        )
        state.justification_bits[1] = True
        set_epoch_target_participation(spec, state, previous_epoch)
        return
    if finality_shape == "FINALITY_CURRENT_JUSTIFIED":
        set_epoch_target_participation(spec, state, current_epoch)
        return
    if finality_shape == "FINALITY_FINALIZE_CURRENT":
        state.current_justified_checkpoint = spec.Checkpoint(
            epoch=spec.Epoch(current_epoch - 1),
            root=spec.get_block_root(state, spec.Epoch(current_epoch - 1)),
        )
        state.justification_bits[1] = True
        set_epoch_target_participation(spec, state, current_epoch)
        return
    raise ValueError(f"Unsupported finality profile shape: {finality_shape}")


def setup_participation_epoch_state(spec, state, *, leak: bool) -> None:
    current_epoch = spec.Epoch(spec.MIN_EPOCHS_TO_INACTIVITY_PENALTY + 2 if leak else 2)
    transition_to(spec, state, spec.compute_start_slot_at_epoch(current_epoch))
    finalized_epoch = spec.GENESIS_EPOCH if leak else spec.Epoch(current_epoch - 1)
    state.finalized_checkpoint = spec.Checkpoint(
        epoch=finalized_epoch,
        root=spec.get_block_root(state, finalized_epoch),
    )
    for index, validator in enumerate(state.validators):
        validator.slashed = False
        validator.activation_eligibility_epoch = spec.Epoch(0)
        validator.activation_epoch = spec.Epoch(0)
        validator.exit_epoch = spec.FAR_FUTURE_EPOCH
        validator.withdrawable_epoch = spec.FAR_FUTURE_EPOCH
        validator.effective_balance = spec.MIN_ACTIVATION_BALANCE
        state.balances[index] = spec.MIN_ACTIVATION_BALANCE
        state.inactivity_scores[index] = spec.uint64(0)
    clear_participation(spec, state)


def clear_participation(spec, state) -> None:
    for index in range(len(state.validators)):
        state.previous_epoch_participation[index] = spec.ParticipationFlags(0)
        state.current_epoch_participation[index] = spec.ParticipationFlags(0)


def set_full_participation_flags(spec, state, *, previous: bool, current: bool) -> None:
    full_flags = spec.ParticipationFlags(0)
    for flag_index in range(len(spec.PARTICIPATION_FLAG_WEIGHTS)):
        full_flags = spec.add_flag(full_flags, flag_index)
    for index in range(len(state.validators)):
        if previous:
            state.previous_epoch_participation[index] = full_flags
        if current:
            state.current_epoch_participation[index] = full_flags


def prepare_state_for_slashings_reset(spec, state, profile: dict[str, Any]) -> None:
    state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(2))
    reset_index = (spec.get_current_epoch(state) + 1) % spec.EPOCHS_PER_SLASHINGS_VECTOR
    state.slashings[reset_index] = spec.Gwei(0)

    intent = profile.get("guide_intent")
    if intent in (None, "reset_nonzero"):
        state.slashings[reset_index] = spec.Gwei(spec.MIN_ACTIVATION_BALANCE)
    elif intent == "already_zero":
        return
    else:
        raise ValueError(f"Unsupported slashings reset guide intent: {intent}")


def prepare_state_for_randao_mixes_reset(spec, state, profile: dict[str, Any]) -> None:
    state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(2))
    current_epoch = spec.get_current_epoch(state)
    reset_index = (current_epoch + 1) % spec.EPOCHS_PER_HISTORICAL_VECTOR
    current_mix = spec.get_randao_mix(state, current_epoch)

    intent = profile.get("guide_intent")
    if intent in (None, "reset_to_current_mix"):
        state.randao_mixes[reset_index] = b"\x56" * 32
    elif intent == "already_current_mix":
        state.randao_mixes[reset_index] = current_mix
    else:
        raise ValueError(f"Unsupported randao mixes reset guide intent: {intent}")


def prepare_state_for_eth1_data_reset(spec, state, profile: dict[str, Any]) -> None:
    intent = profile.get("guide_intent")
    if intent in (None, "period_boundary"):
        state.slot = spec.Slot(spec.EPOCHS_PER_ETH1_VOTING_PERIOD * spec.SLOTS_PER_EPOCH - 1)
    elif intent == "non_boundary":
        state.slot = spec.Slot(spec.SLOTS_PER_EPOCH - 1)
    else:
        raise ValueError(f"Unsupported eth1 data reset guide intent: {intent}")

    state.eth1_data_votes = spec.List[spec.Eth1Data, spec.EPOCHS_PER_ETH1_VOTING_PERIOD * spec.SLOTS_PER_EPOCH]()
    state.eth1_data_votes.append(
        spec.Eth1Data(
            deposit_root=b"\xaa" * 32,
            deposit_count=state.eth1_deposit_index,
            block_hash=b"\xbb" * 32,
        )
    )


def prepare_state_for_historical_summaries_update(spec, state, profile: dict[str, Any]) -> None:
    intent = profile.get("guide_intent")
    if intent in (None, "period_boundary"):
        state.slot = spec.Slot(spec.SLOTS_PER_HISTORICAL_ROOT - 1)
    elif intent == "non_boundary":
        state.slot = spec.compute_start_slot_at_epoch(spec.Epoch(2))
    else:
        raise ValueError(f"Unsupported historical summaries update guide intent: {intent}")


def prepare_state_for_sync_committee_updates(spec, state, profile: dict[str, Any]) -> None:
    intent = profile.get("guide_intent")
    if intent in (None, "period_boundary", "genesis_period_boundary"):
        target_period = 1
        if intent == "genesis_period_boundary":
            target_period = 0
        end_epoch = spec.Epoch(
            (target_period + 1) * spec.EPOCHS_PER_SYNC_COMMITTEE_PERIOD - 1
        )
        transition_to(spec, state, spec.compute_start_slot_at_epoch(end_epoch))
    elif intent == "non_boundary":
        transition_to(spec, state, spec.compute_start_slot_at_epoch(spec.Epoch(1)))
    else:
        raise ValueError(f"Unsupported sync committee updates guide intent: {intent}")


def prepare_state_for_sync_aggregate(spec, state, profile: dict[str, Any]):
    transition_to(spec, state, spec.Slot(1))
    profile["bls_setting"] = 1
    old_bls_active = bls.bls_active
    bls.bls_active = True
    try:
        state.current_sync_committee.aggregate_pubkey = spec.eth_aggregate_pubkeys(
            state.current_sync_committee.pubkeys
        )
    finally:
        bls.bls_active = old_bls_active
    committee_indices = compute_committee_indices(state)
    committee_size = len(committee_indices)
    intent = profile.get("guide_intent")

    if input_profile_shape(profile, "sync_aggregate_input", "branch_target") is not None:
        if intent in (None, "all_participate"):
            participant_count = committee_size
        elif intent == "majority_participate":
            participant_count = committee_size // 2 + 1
        elif intent == "minority_participate":
            participant_count = max(1, committee_size // 2)
        elif intent in ("none_participate", "bad_signature"):
            participant_count = 0 if intent == "none_participate" else committee_size
        else:
            raise ValueError(f"Unsupported sync aggregate guide intent: {intent}")
    elif profile.get("profile_driven"):
        participant_count = sync_participant_count_from_profile(profile, committee_size)
    elif intent in (None, "all_participate"):
        participant_count = committee_size
    elif intent == "majority_participate":
        participant_count = committee_size // 2 + 1
    elif intent == "minority_participate":
        participant_count = max(1, committee_size // 2)
    elif intent in ("none_participate", "bad_signature"):
        participant_count = 0 if intent == "none_participate" else committee_size
    else:
        raise ValueError(f"Unsupported sync aggregate guide intent: {intent}")

    committee_bits = [index < participant_count for index in range(committee_size)]
    participants = [
        validator_index
        for validator_index, participation_bit in zip(committee_indices, committee_bits)
        if participation_bit
    ]
    old_bls_active = bls.bls_active
    bls.bls_active = True
    try:
        signature = compute_aggregate_sync_committee_signature(
            spec,
            state,
            spec.Slot(0),
            participants,
        )
    finally:
        bls.bls_active = old_bls_active
    if intent == "bad_signature" or operation_input_shape(profile, "signature_shape") == "SIGNATURE_INVALID":
        signature = spec.BLSSignature(b"\x42" * 96)

    return spec.SyncAggregate(
        sync_committee_bits=committee_bits,
        sync_committee_signature=signature,
    )


def sync_participant_count_from_profile(profile: dict[str, Any], committee_size: int) -> int:
    participation_shape = (
        profile.get("input_profiles", {})
        .get("participation", {})
        .get("participation_shape", "PARTICIPATION_FULL")
    )
    if participation_shape == "PARTICIPATION_NONE":
        return 0
    if participation_shape == "PARTICIPATION_TARGET_ONLY":
        return committee_size // 2 + 1
    if participation_shape == "PARTICIPATION_POOR_SUPPORT":
        return max(1, committee_size // 2)
    if participation_shape == "PARTICIPATION_FULL":
        return committee_size
    raise ValueError(f"Unsupported sync participation profile shape: {participation_shape}")


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
