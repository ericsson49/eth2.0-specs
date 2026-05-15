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


def load_validator_state_model() -> str:
    model = resources.files(MODEL_PACKAGE).joinpath(VALIDATOR_STATE_MODEL)
    return model.read_text()


def transpile_validator_state_model() -> str:
    return Convertor().convert(load_validator_state_model())


def solve_validator_state_profiles(limit: int | None = None) -> Iterable[dict[str, Any]]:
    for index, profile in enumerate(get_solutions(transpile_validator_state_model())):
        if limit is not None and index >= limit:
            return
        yield profile


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
