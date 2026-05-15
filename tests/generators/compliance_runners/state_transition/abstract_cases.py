from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from itertools import combinations
from typing import Any

from tests.generators.compliance_runners.py_to_mzn import Convertor, get_solutions

from .ontology import guided_operation_intents

MODEL_PACKAGE = "tests.generators.compliance_runners.state_transition.models"
VALIDATOR_STATE_MODEL = "validator_state.py"
PROFILE_MODELS = {
    "validator_state": "validator_state.py",
    "operation_input": "operation_input.py",
    "queue": "queue.py",
    "epoch_boundary": "epoch_boundary.py",
    "participation": "participation.py",
}


@dataclass(frozen=True)
class AbstractStateTransitionCase:
    """A solved abstract profile plus runner-facing labels."""

    handler_name: str
    case_name: str
    profile: dict[str, Any]


HANDLER_NAMES = (
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

GUIDED_OPERATION_INTENTS = guided_operation_intents()
DEFAULT_PROFILE_PARTITION_DIMENSIONS = (
    "withdrawal_credential_type",
    "activation_epoch_to_current_epoch",
    "exit_epoch_to_current_epoch",
    "withdrawable_epoch_to_current_epoch",
    "balance_is_zero",
    "balance_to_effective_balance",
    "effective_balance_lte_ejection_balance",
    "effective_balance_to_min_activation_balance",
    "effective_balance_to_max_effective_balance",
    "slashed",
    "exit_epoch_set",
    "has_pending_withdrawal_request",
    "has_pending_consolidation_request",
)
INPUT_PROFILE_DIMENSIONS = {
    "operation_input": (
        "signature_shape",
        "proof_shape",
        "lookup_shape",
        "source_address_shape",
        "source_target_relation",
    ),
    "queue": (
        "pending_partial_withdrawals",
        "pending_consolidations",
        "pending_deposits",
        "pending_request",
    ),
    "epoch_boundary": ("epoch_boundary_shape",),
    "participation": (
        "participation_shape",
        "finality_shape",
        "inactivity_leak",
    ),
}
HANDLER_INPUT_PROFILE_MODELS = {
    "proposer_slashing": ("operation_input",),
    "attester_slashing": ("operation_input",),
    "attestation": ("operation_input", "epoch_boundary"),
    "deposit": ("operation_input",),
    "bls_to_execution_change": ("operation_input",),
    "deposit_request": ("validator_state",),
    "voluntary_exit": ("validator_state", "queue", "epoch_boundary"),
    "withdrawal_request": ("validator_state", "operation_input", "queue"),
    "consolidation_request": ("validator_state", "operation_input", "queue"),
    "pending_deposits": ("queue", "epoch_boundary"),
    "pending_consolidations": ("queue",),
    "effective_balance_updates": ("validator_state",),
    "registry_updates": ("validator_state",),
    "slashings": ("validator_state",),
    "justification_and_finalization": ("epoch_boundary", "participation"),
    "inactivity_updates": ("epoch_boundary", "participation"),
    "rewards_and_penalties": ("epoch_boundary", "participation"),
    "participation_flag_updates": ("participation",),
    "slashings_reset": ("epoch_boundary",),
    "randao_mixes_reset": ("epoch_boundary",),
    "eth1_data_reset": ("epoch_boundary",),
    "historical_summaries_update": ("epoch_boundary",),
    "sync_committee_updates": ("epoch_boundary",),
    "sync_aggregate": ("operation_input", "participation"),
}


def load_validator_state_model() -> str:
    return load_profile_model(VALIDATOR_STATE_MODEL)


def load_profile_model(model_name: str) -> str:
    model = resources.files(MODEL_PACKAGE).joinpath(model_name)
    return model.read_text()


def transpile_validator_state_model() -> str:
    return Convertor().convert(load_validator_state_model())


def transpile_profile_model(profile_model: str) -> str:
    return Convertor().convert(load_profile_model(PROFILE_MODELS[profile_model]))


def solve_validator_state_profiles(limit: int | None = None) -> Iterable[dict[str, Any]]:
    for index, profile in enumerate(get_solutions(transpile_validator_state_model())):
        if limit is not None and index >= limit:
            return
        yield profile


def solve_profile_model(profile_model: str) -> Iterable[dict[str, Any]]:
    yield from get_solutions(transpile_profile_model(profile_model))


def enumerate_abstract_cases(limit: int | None = None) -> Iterable[AbstractStateTransitionCase]:
    for index, profile in enumerate(solve_validator_state_profiles(limit=limit)):
        handler_name = classify_handler(profile)
        yield make_abstract_case(handler_name, index, profile)


def select_abstract_cases(
    per_handler_limit: int,
    handlers: Iterable[str] = HANDLER_NAMES,
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")

    counts = {handler: 0 for handler in requested_handlers}
    for index, profile in enumerate(solve_validator_state_profiles()):
        handler_name = classify_handler(profile)
        if handler_name not in counts:
            continue
        if counts[handler_name] >= per_handler_limit:
            continue
        counts[handler_name] += 1
        yield make_abstract_case(handler_name, index, profile)
        if all(count >= per_handler_limit for count in counts.values()):
            return


def enumerate_matching_abstract_cases(
    handlers: Iterable[str],
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")

    for index, profile in enumerate(solve_validator_state_profiles()):
        handler_name = classify_handler(profile)
        if handler_name in requested_handlers:
            yield make_abstract_case(handler_name, index, profile)


def enumerate_materializable_operation_cases(
    handlers: Iterable[str],
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")

    for index, profile in enumerate(solve_validator_state_profiles()):
        for handler_name in requested_handlers:
            if is_materializable_for_handler(profile, handler_name):
                yield make_abstract_case(handler_name, index, profile)


def enumerate_profile_partition_cases(
    handlers: Iterable[str],
    *,
    dimensions: Iterable[str] | None = None,
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")

    partition_dimensions = tuple(dimensions or DEFAULT_PROFILE_PARTITION_DIMENSIONS)
    candidates = {
        handler_name: {dimension: {} for dimension in partition_dimensions}
        for handler_name in requested_handlers
    }
    for index, profile in enumerate(solve_validator_state_profiles()):
        validate_profile_dimensions(profile, partition_dimensions)
        for handler_name in requested_handlers:
            if not is_materializable_for_handler(profile, handler_name):
                continue
            for dimension in partition_dimensions:
                value = profile[dimension]
                if value in candidates[handler_name][dimension]:
                    continue
                candidates[handler_name][dimension][value] = (index, profile)

    for handler_name in requested_handlers:
        max_values = max(
            len(dimension_candidates)
            for dimension_candidates in candidates[handler_name].values()
        )
        for value_index in range(max_values):
            for dimension in partition_dimensions:
                ordered_values = sorted(
                    candidates[handler_name][dimension],
                    key=profile_partition_value_sort_key,
                )
                if value_index >= len(ordered_values):
                    continue
                value = ordered_values[value_index]
                solution_index, profile = candidates[handler_name][dimension][value]
                profile_with_tags = dict(profile)
                profile_with_tags["coverage_tags"] = [
                    f"handler:{handler_name}",
                    f"profile:{dimension}:{value}",
                ]
                yield make_abstract_case(
                    handler_name,
                    solution_index,
                    profile_with_tags,
                    case_name=profile_partition_case_name(
                        dimension,
                        value,
                        solution_index,
                    ),
                )


def enumerate_profile_interaction_cases(
    handlers: Iterable[str],
    *,
    dimensions: Iterable[str] | None = None,
    order: int = 2,
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")
    if order < 2:
        raise ValueError(f"Profile interaction order must be at least 2: {order}")

    interaction_dimensions = tuple(dimensions or DEFAULT_PROFILE_PARTITION_DIMENSIONS)
    dimension_groups = tuple(combinations(interaction_dimensions, order))
    candidates = {
        handler_name: {dimension_group: {} for dimension_group in dimension_groups}
        for handler_name in requested_handlers
    }
    for index, profile in enumerate(solve_validator_state_profiles()):
        validate_profile_dimensions(profile, interaction_dimensions)
        for handler_name in requested_handlers:
            if not is_materializable_for_handler(profile, handler_name):
                continue
            for dimension_group in dimension_groups:
                values = tuple(profile[dimension] for dimension in dimension_group)
                if values in candidates[handler_name][dimension_group]:
                    continue
                candidates[handler_name][dimension_group][values] = (index, profile)

    for handler_name in requested_handlers:
        max_values = max(
            len(group_candidates)
            for group_candidates in candidates[handler_name].values()
        )
        for value_index in range(max_values):
            for dimension_group in dimension_groups:
                ordered_values = sorted(
                    candidates[handler_name][dimension_group],
                    key=profile_interaction_value_sort_key,
                )
                if value_index >= len(ordered_values):
                    continue
                values = ordered_values[value_index]
                solution_index, profile = candidates[handler_name][dimension_group][values]
                profile_with_tags = dict(profile)
                profile_with_tags["profile_interaction"] = {
                    "dimensions": list(dimension_group),
                    "values": [str(value) for value in values],
                }
                profile_with_tags["coverage_tags"] = [
                    f"handler:{handler_name}",
                    profile_interaction_tag(dimension_group, values),
                ]
                yield make_abstract_case(
                    handler_name,
                    solution_index,
                    profile_with_tags,
                    case_name=profile_interaction_case_name(
                        dimension_group,
                        values,
                        solution_index,
                    ),
                )


def enumerate_input_profile_cases(
    handlers: Iterable[str],
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")

    base_profiles = first_materializable_profiles(requested_handlers)
    model_solutions = {
        profile_model: tuple(solve_profile_model(profile_model))
        for profile_model in PROFILE_MODELS
        if profile_model != "validator_state"
    }
    for handler_name in requested_handlers:
        if handler_name not in base_profiles:
            continue
        solution_index, base_profile = base_profiles[handler_name]
        for profile_model in HANDLER_INPUT_PROFILE_MODELS[handler_name]:
            if profile_model == "validator_state":
                yield from enumerate_validator_state_input_cases(
                    handler_name,
                    solution_index,
                    base_profile,
                )
                continue
            yield from enumerate_model_input_cases(
                handler_name,
                solution_index,
                base_profile,
                profile_model,
                model_solutions[profile_model],
            )


def first_materializable_profiles(
    requested_handlers: tuple[str, ...],
) -> dict[str, tuple[int, dict[str, Any]]]:
    base_profiles = {}
    for index, profile in enumerate(solve_validator_state_profiles()):
        for handler_name in requested_handlers:
            if handler_name in base_profiles:
                continue
            if is_materializable_for_handler(profile, handler_name):
                base_profiles[handler_name] = (index, profile)
        if len(base_profiles) == len(requested_handlers):
            break
    return base_profiles


def enumerate_validator_state_input_cases(
    handler_name: str,
    solution_index: int,
    base_profile: dict[str, Any],
) -> Iterable[AbstractStateTransitionCase]:
    for dimension in DEFAULT_PROFILE_PARTITION_DIMENSIONS:
        profile = dict(base_profile)
        intent = input_intent_for_dimension(handler_name, "validator_state", dimension, profile[dimension])
        if intent is None:
            continue
        profile["guide_intent"] = intent
        profile["input_profiles"] = {"validator_state": {dimension: profile[dimension]}}
        profile["coverage_tags"] = [
            f"handler:{handler_name}",
            f"input_profile:validator_state.{dimension}:{profile[dimension]}",
        ]
        yield make_abstract_case(
            handler_name,
            solution_index,
            profile,
            case_name=f"input_validator_state_{dimension}_{safe_case_value(profile[dimension])}",
        )


def enumerate_model_input_cases(
    handler_name: str,
    solution_index: int,
    base_profile: dict[str, Any],
    profile_model: str,
    model_profiles: tuple[dict[str, Any], ...],
) -> Iterable[AbstractStateTransitionCase]:
    seen = {dimension: set() for dimension in INPUT_PROFILE_DIMENSIONS[profile_model]}
    for model_profile in model_profiles:
        for dimension in INPUT_PROFILE_DIMENSIONS[profile_model]:
            value = model_profile[dimension]
            if value in seen[dimension]:
                continue
            intent = input_intent_for_dimension(handler_name, profile_model, dimension, value)
            if intent is None:
                continue
            seen[dimension].add(value)
            profile = dict(base_profile)
            profile["guide_intent"] = intent
            profile["input_profiles"] = {profile_model: {dimension: value}}
            profile["coverage_tags"] = [
                f"handler:{handler_name}",
                f"input_profile:{profile_model}.{dimension}:{value}",
            ]
            yield make_abstract_case(
                handler_name,
                solution_index,
                profile,
                case_name=f"input_{profile_model}_{dimension}_{safe_case_value(value)}",
            )


def enumerate_guided_operation_cases(
    handlers: Iterable[str],
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(GUIDED_OPERATION_INTENTS)
    if unknown_handlers:
        raise ValueError(f"Unsupported guided handlers: {sorted(unknown_handlers)}")

    base_profiles = {}
    for index, profile in enumerate(solve_validator_state_profiles()):
        for handler_name in requested_handlers:
            if handler_name in base_profiles:
                continue
            if is_materializable_for_handler(profile, handler_name):
                base_profiles[handler_name] = (index, profile)
        if len(base_profiles) == len(requested_handlers):
            break

    for handler_name in requested_handlers:
        if handler_name not in base_profiles:
            continue
        solution_index, base_profile = base_profiles[handler_name]
        for intent_name in GUIDED_OPERATION_INTENTS[handler_name]:
            profile = dict(base_profile)
            profile["guide_intent"] = intent_name
            profile["coverage_tags"] = [
                f"handler:{handler_name}",
                f"intent:{intent_name}",
            ]
            yield make_abstract_case(
                handler_name,
                solution_index,
                profile,
                case_name=f"guided_{intent_name}",
            )


def is_materializable_for_handler(profile: dict[str, Any], handler_name: str) -> bool:
    if handler_name == "proposer_slashing":
        return True
    if handler_name == "attester_slashing":
        return True
    if handler_name == "attestation":
        return True
    if handler_name == "deposit":
        return True
    if handler_name == "bls_to_execution_change":
        return True
    if handler_name == "deposit_request":
        return True
    if handler_name == "voluntary_exit":
        return True
    if handler_name == "withdrawal_request":
        return profile["withdrawal_credential_type"] in ("ETH1", "COMP")
    if handler_name == "consolidation_request":
        return profile["withdrawal_credential_type"] in ("ETH1", "COMP")
    if handler_name == "pending_deposits":
        return True
    if handler_name == "pending_consolidations":
        return profile["withdrawal_credential_type"] in ("ETH1", "COMP")
    if handler_name == "effective_balance_updates":
        return True
    if handler_name == "registry_updates":
        return True
    if handler_name == "slashings":
        return True
    if handler_name == "justification_and_finalization":
        return True
    if handler_name == "inactivity_updates":
        return True
    if handler_name == "rewards_and_penalties":
        return True
    if handler_name == "participation_flag_updates":
        return True
    if handler_name == "slashings_reset":
        return True
    if handler_name == "randao_mixes_reset":
        return True
    if handler_name == "eth1_data_reset":
        return True
    if handler_name == "historical_summaries_update":
        return True
    if handler_name == "sync_committee_updates":
        return True
    if handler_name == "sync_aggregate":
        return True
    raise ValueError(f"Unknown handler: {handler_name}")


def input_intent_for_dimension(
    handler_name: str,
    profile_model: str,
    dimension: str,
    value: Any,
) -> str | None:
    value = str(value)
    if profile_model == "operation_input":
        return operation_input_intent(handler_name, dimension, value)
    if profile_model == "queue":
        return queue_input_intent(handler_name, dimension, value)
    if profile_model == "epoch_boundary":
        return epoch_boundary_input_intent(handler_name, value)
    if profile_model == "participation":
        return participation_input_intent(handler_name, dimension, value)
    if profile_model == "validator_state":
        return validator_state_input_intent(handler_name, dimension, value)
    raise ValueError(f"Unknown input profile model: {profile_model}")


def operation_input_intent(handler_name: str, dimension: str, value: str) -> str | None:
    if dimension == "signature_shape" and value == "SIGNATURE_INVALID":
        return {
            "proposer_slashing": "bad_signature",
            "attester_slashing": "bad_signature",
            "attestation": "bad_signature",
            "deposit": "invalid_signature",
            "bls_to_execution_change": "bad_signature",
            "sync_aggregate": "bad_signature",
        }.get(handler_name)
    if dimension == "proof_shape" and value == "PROOF_INVALID" and handler_name == "deposit":
        return "invalid_proof"
    if dimension == "lookup_shape" and value == "LOOKUP_MISSING":
        return {
            "bls_to_execution_change": "out_of_range",
            "withdrawal_request": "pubkey_missing",
            "consolidation_request": "source_missing",
        }.get(handler_name)
    if dimension == "source_address_shape" and value == "SOURCE_ADDRESS_INVALID":
        return {
            "withdrawal_request": "bad_source_address",
            "consolidation_request": "bad_source_address",
        }.get(handler_name)
    if (
        dimension == "source_target_relation"
        and value == "SOURCE_TARGET_SAME"
        and handler_name == "consolidation_request"
    ):
        return "source_equals_target"
    return None


def queue_input_intent(handler_name: str, dimension: str, value: str) -> str | None:
    if dimension == "pending_partial_withdrawals" and value == "FULL":
        return {
            "withdrawal_request": "queue_full",
            "voluntary_exit": "pending_withdrawal",
        }.get(handler_name)
    if dimension == "pending_partial_withdrawals" and value == "NONEMPTY":
        return {
            "withdrawal_request": "full_exit_with_pending_withdrawal",
            "voluntary_exit": "pending_withdrawal",
        }.get(handler_name)
    if dimension == "pending_consolidations" and value == "FULL":
        return {"consolidation_request": "queue_full"}.get(handler_name)
    if dimension == "pending_consolidations" and value == "EMPTY":
        return {"pending_consolidations": "empty_queue"}.get(handler_name)
    if dimension == "pending_deposits" and value == "FULL":
        return {"pending_deposits": "max_per_epoch_reached"}.get(handler_name)
    if dimension == "pending_deposits" and value == "EMPTY":
        return {"pending_deposits": "eth1_bridge_blocks_request"}.get(handler_name)
    if dimension == "pending_request" and value == "REQUEST_WITHDRAWAL":
        return {
            "withdrawal_request": "full_exit_with_pending_withdrawal",
            "voluntary_exit": "pending_withdrawal",
            "consolidation_request": "source_pending_withdrawal",
        }.get(handler_name)
    if dimension == "pending_request" and value == "REQUEST_CONSOLIDATION":
        return {"pending_consolidations": "success"}.get(handler_name)
    return None


def epoch_boundary_input_intent(handler_name: str, value: str) -> str | None:
    if value == "GENESIS":
        return {
            "justification_and_finalization": "genesis_skip",
            "inactivity_updates": "genesis_skip",
            "rewards_and_penalties": "genesis_skip",
            "sync_committee_updates": "genesis_period_boundary",
        }.get(handler_name)
    if value == "PERIOD_BOUNDARY":
        return {
            "eth1_data_reset": "period_boundary",
            "historical_summaries_update": "period_boundary",
            "sync_committee_updates": "period_boundary",
        }.get(handler_name)
    if value == "NON_BOUNDARY":
        return {
            "eth1_data_reset": "non_boundary",
            "historical_summaries_update": "non_boundary",
            "sync_committee_updates": "non_boundary",
        }.get(handler_name)
    return None


def participation_input_intent(handler_name: str, dimension: str, value: str) -> str | None:
    if dimension == "participation_shape":
        if value == "PARTICIPATION_NONE":
            return {
                "inactivity_updates": "non_participating_no_leak",
                "rewards_and_penalties": "empty_participation_penalty",
                "participation_flag_updates": "all_zero",
                "sync_aggregate": "none_participate",
            }.get(handler_name)
        if value == "PARTICIPATION_FULL":
            return {
                "rewards_and_penalties": "full_participation_reward",
                "participation_flag_updates": "previous_filled",
                "sync_aggregate": "all_participate",
            }.get(handler_name)
        if value == "PARTICIPATION_TARGET_ONLY":
            return {
                "justification_and_finalization": "current_justified",
                "inactivity_updates": "participating_recovery",
                "sync_aggregate": "majority_participate",
            }.get(handler_name)
        if value == "PARTICIPATION_POOR_SUPPORT":
            return {"justification_and_finalization": "poor_support"}.get(handler_name)
    if dimension == "finality_shape":
        return {
            "FINALITY_CURRENT_JUSTIFIED": "current_justified",
            "FINALITY_PREVIOUS_JUSTIFIED": "previous_justified",
            "FINALITY_FINALIZE_CURRENT": "finalize_current",
        }.get(value) if handler_name == "justification_and_finalization" else None
    if dimension == "inactivity_leak" and value == "True":
        return {
            "inactivity_updates": "non_participating_leak",
            "rewards_and_penalties": "inactivity_leak_penalty",
        }.get(handler_name)
    return None


def validator_state_input_intent(handler_name: str, dimension: str, value: str) -> str | None:
    if handler_name == "registry_updates":
        if dimension == "activation_epoch_to_current_epoch" and value == ">":
            return "activation_queue"
        if dimension == "effective_balance_lte_ejection_balance" and value == "True":
            return "ejection"
    if handler_name == "effective_balance_updates":
        if dimension == "balance_to_effective_balance" and value == "<":
            return "step_down"
        if dimension == "balance_to_effective_balance" and value == ">":
            return "step_up"
        if dimension == "effective_balance_to_max_effective_balance" and value == "=":
            return "cap_at_max"
    if handler_name == "slashings":
        if dimension == "slashed" and value == "True":
            return "penalty_applied"
        if dimension == "slashed" and value == "False":
            return "no_slashed_validators"
        if dimension == "balance_is_zero" and value == "True":
            return "zero_slashing_balance"
    if handler_name == "deposit_request":
        if dimension == "activation_eligibility_epoch_set" and value == "True":
            return "start_index_set"
        if dimension == "activation_eligibility_epoch_set" and value == "False":
            return "start_index_unset"
    return None


def validate_profile_dimensions(
    profile: dict[str, Any],
    dimensions: tuple[str, ...],
) -> None:
    unknown_dimensions = [dimension for dimension in dimensions if dimension not in profile]
    if unknown_dimensions:
        raise ValueError(f"Unknown profile dimensions: {unknown_dimensions}")


def profile_partition_case_name(dimension: str, value: Any, solution_index: int) -> str:
    return f"profile_{dimension}_{safe_case_value(value)}_{solution_index:04d}"


def profile_interaction_case_name(
    dimensions: tuple[str, ...],
    values: tuple[Any, ...],
    solution_index: int,
) -> str:
    dimension_label = "_x_".join(dimensions)
    value_label = "_x_".join(safe_case_value(value) for value in values)
    return f"profile_pair_{dimension_label}_{value_label}_{solution_index:04d}"


def profile_interaction_tag(dimensions: tuple[str, ...], values: tuple[Any, ...]) -> str:
    labels = [
        f"{dimension}:{value}"
        for dimension, value in zip(dimensions, values, strict=True)
    ]
    return f"profile_interaction:{'|'.join(labels)}"


def safe_case_value(value: Any) -> str:
    text = str(value)
    return (
        text.replace("<", "lt")
        .replace("=", "eq")
        .replace(">", "gt")
        .replace(" ", "_")
        .replace("/", "_")
        .lower()
    )


def profile_partition_value_sort_key(value: Any) -> tuple[int, str]:
    if isinstance(value, bool):
        return (0 if value else 1, str(value))
    return (0, str(value))


def profile_interaction_value_sort_key(values: tuple[Any, ...]) -> tuple[tuple[int, str], ...]:
    return tuple(profile_partition_value_sort_key(value) for value in values)


def make_abstract_case(
    handler_name: str,
    solution_index: int,
    profile: dict[str, Any],
    case_name: str | None = None,
) -> AbstractStateTransitionCase:
    return AbstractStateTransitionCase(
        handler_name=handler_name,
        case_name=case_name or f"validator_state_profile_{solution_index:04d}",
        profile=profile,
    )


def classify_handler(profile: dict[str, Any]) -> str:
    if profile["withdrawal_credential_type"] == "BLS":
        return "deposit_request"
    if profile["has_pending_consolidation_request"]:
        return "consolidation_request"
    if profile["has_pending_withdrawal_request"]:
        return "withdrawal_request"
    if profile["withdrawal_credential_type"] in ("ETH1", "COMP"):
        return "withdrawal_request"
    return "registry_updates"
