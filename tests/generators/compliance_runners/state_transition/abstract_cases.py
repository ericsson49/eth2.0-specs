from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from itertools import combinations, product
from typing import Any

from tests.generators.compliance_runners.py_to_mzn import Convertor, get_solutions

from .ontology import guided_operation_intents

MODEL_PACKAGE = "tests.generators.compliance_runners.state_transition.models"
VALIDATOR_STATE_MODEL = "validator_state.py"
PROFILE_MODELS = {
    "validator_state": "validator_state.py",
    "operation_input": "operation_input.py",
    "proposer_slashing_input": "proposer_slashing_input.py",
    "attester_slashing_input": "attester_slashing_input.py",
    "attestation_input": "attestation_input.py",
    "deposit_input": "deposit_input.py",
    "bls_to_execution_change_input": "bls_to_execution_change_input.py",
    "voluntary_exit_input": "voluntary_exit_input.py",
    "withdrawal_request_input": "withdrawal_request_input.py",
    "consolidation_request_input": "consolidation_request_input.py",
    "pending_deposits_input": "pending_deposits_input.py",
    "pending_consolidations_input": "pending_consolidations_input.py",
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
    "proposer_slashing_input": (
        "header_relation",
        "proposer_relation",
        "proposer_status",
    ),
    "attester_slashing_input": (
        "attester_overlap",
        "attestation_data_relation",
        "attester_status",
    ),
    "attestation_input": (
        "slot_relation",
        "target_epoch_relation",
        "committee_index_shape",
        "aggregation_shape",
    ),
    "deposit_input": ("recipient_shape",),
    "bls_to_execution_change_input": (
        "credential_shape",
        "withdrawal_pubkey_relation",
    ),
    "voluntary_exit_input": ("exit_epoch_relation",),
    "withdrawal_request_input": ("request_kind",),
    "consolidation_request_input": (
        "request_kind",
        "target_lookup_shape",
        "source_activity_shape",
        "target_activity_shape",
        "target_credential_shape",
        "churn_shape",
    ),
    "pending_deposits_input": (
        "deposit_kind",
        "finality_shape",
        "churn_shape",
        "bridge_state",
    ),
    "pending_consolidations_input": (
        "source_shape",
        "balance_shape",
        "queue_shape",
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
    "proposer_slashing": ("proposer_slashing_input", "operation_input"),
    "attester_slashing": ("attester_slashing_input", "operation_input"),
    "attestation": ("attestation_input", "operation_input", "epoch_boundary"),
    "deposit": ("deposit_input", "operation_input"),
    "bls_to_execution_change": ("bls_to_execution_change_input", "operation_input"),
    "deposit_request": ("validator_state",),
    "voluntary_exit": ("voluntary_exit_input", "queue", "epoch_boundary", "validator_state"),
    "withdrawal_request": ("withdrawal_request_input", "operation_input", "queue", "validator_state"),
    "consolidation_request": (
        "consolidation_request_input",
        "operation_input",
        "queue",
        "validator_state",
    ),
    "pending_deposits": ("pending_deposits_input", "queue", "epoch_boundary"),
    "pending_consolidations": ("pending_consolidations_input", "queue"),
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
PROFILE_DRIVEN_INPUT_HANDLERS = frozenset({
    "proposer_slashing",
    "attester_slashing",
    "attestation",
    "deposit",
    "bls_to_execution_change",
    "deposit_request",
    "voluntary_exit",
    "withdrawal_request",
    "consolidation_request",
    "sync_aggregate",
    "pending_deposits",
    "pending_consolidations",
    "effective_balance_updates",
    "registry_updates",
    "slashings",
    "justification_and_finalization",
    "inactivity_updates",
    "rewards_and_penalties",
    "participation_flag_updates",
})


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
    yield from cached_profile_model_solutions(profile_model)


@lru_cache
def cached_profile_model_solutions(profile_model: str) -> tuple[dict[str, Any], ...]:
    return tuple(get_solutions(transpile_profile_model(profile_model)))


@lru_cache
def cached_validator_state_profiles() -> tuple[dict[str, Any], ...]:
    return tuple(get_solutions(transpile_validator_state_model()))


def profile_constraints_compatible(
    constraints: Iterable[tuple[str, str, Any]],
) -> bool:
    """Return whether every profile-model projection has a matching solution."""
    grouped_constraints = group_profile_constraints(constraints)

    for profile_model, model_constraints in grouped_constraints.items():
        if not profile_model_constraints_compatible(profile_model, model_constraints):
            return False
    return True


def profile_model_constraints_compatible(
    profile_model: str,
    constraints: list[tuple[str, Any]],
) -> bool:
    return complete_profile_model_constraints(profile_model, constraints) is not None


def complete_profile_model_constraints(
    profile_model: str,
    constraints: list[tuple[str, Any]],
) -> dict[str, Any] | None:
    """Return a deterministic solved profile row matching a partial projection."""
    for profile in profiles_for_compatibility(profile_model):
        if all(profile[dimension] == value for dimension, value in constraints):
            return profile
    return None


def group_profile_constraints(
    constraints: Iterable[tuple[str, str, Any]],
) -> dict[str, list[tuple[str, Any]]]:
    grouped_constraints: dict[str, list[tuple[str, Any]]] = {}
    for profile_model, dimension, value in constraints:
        grouped_constraints.setdefault(profile_model, []).append((dimension, value))
    return grouped_constraints


def profiles_for_compatibility(profile_model: str) -> tuple[dict[str, Any], ...]:
    if profile_model == "validator_state":
        return cached_validator_state_profiles()
    return cached_profile_model_solutions(profile_model)


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
    *,
    order: int = 1,
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")
    if order < 1:
        raise ValueError(f"Input profile order must be at least 1: {order}")

    base_profiles = first_materializable_profiles(requested_handlers)
    for handler_name in requested_handlers:
        if handler_name not in base_profiles:
            continue
        solution_index, base_profile = base_profiles[handler_name]
        dimensions = input_profile_dimensions_for_handler(handler_name, base_profile)
        yield from enumerate_input_profile_dimension_groups(
            handler_name,
            solution_index,
            base_profile,
            dimensions,
            order=order,
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


def input_profile_dimensions_for_handler(
    handler_name: str,
    base_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    dimensions = []
    for profile_model in HANDLER_INPUT_PROFILE_MODELS[handler_name]:
        if profile_model == "validator_state":
            dimensions.extend(validator_state_input_dimensions(handler_name, base_profile))
        else:
            dimensions.extend(
                model_input_dimensions(
                    handler_name,
                    profile_model,
                    tuple(solve_profile_model(profile_model)),
                )
            )
    return dimensions


def validator_state_input_dimensions(
    handler_name: str,
    base_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    dimensions = []
    if is_profile_driven_input_handler(handler_name):
        return model_dimension_values(
            handler_name,
            "validator_state",
            DEFAULT_PROFILE_PARTITION_DIMENSIONS,
            tuple(solve_validator_state_profiles()),
        )

    for dimension in DEFAULT_PROFILE_PARTITION_DIMENSIONS:
        value = base_profile[dimension]
        intent = input_intent_for_dimension(handler_name, "validator_state", dimension, value)
        if intent is None and not is_profile_driven_input_handler(handler_name):
            continue
        dimensions.append(
            {
                "profile_model": "validator_state",
                "dimension": dimension,
                "value": value,
                "intent": intent,
            }
        )
    return dimensions


def model_input_dimensions(
    handler_name: str,
    profile_model: str,
    model_profiles: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    return model_dimension_values(
        handler_name,
        profile_model,
        INPUT_PROFILE_DIMENSIONS[profile_model],
        model_profiles,
    )


def model_dimension_values(
    handler_name: str,
    profile_model: str,
    dimension_names: tuple[str, ...],
    model_profiles: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    dimensions = []
    seen = {dimension: set() for dimension in dimension_names}
    for model_profile in model_profiles:
        for dimension in dimension_names:
            value = model_profile[dimension]
            if value in seen[dimension]:
                continue
            intent = input_intent_for_dimension(handler_name, profile_model, dimension, value)
            if intent is None and not is_profile_driven_input_handler(handler_name):
                continue
            seen[dimension].add(value)
            dimensions.append(
                {
                    "profile_model": profile_model,
                    "dimension": dimension,
                    "value": value,
                    "intent": intent,
                }
            )
    return dimensions


def is_profile_driven_input_handler(handler_name: str) -> bool:
    return handler_name in PROFILE_DRIVEN_INPUT_HANDLERS


def enumerate_input_profile_dimension_groups(
    handler_name: str,
    solution_index: int,
    base_profile: dict[str, Any],
    dimensions: list[dict[str, Any]],
    *,
    order: int,
) -> Iterable[AbstractStateTransitionCase]:
    if order == 1:
        for dimension in dimensions:
            yield make_input_profile_case(handler_name, solution_index, base_profile, (dimension,))
        return

    for dimension in dimensions:
        yield make_input_profile_case(handler_name, solution_index, base_profile, (dimension,))

    for dimension_group in combinations(dimensions, order):
        profile_model_values = [
            input_profile_dimension_values(dimension)
            for dimension in dimension_group
        ]
        for value_group in product(*profile_model_values):
            if complete_input_profiles(handler_name, base_profile, value_group) is None:
                continue
            yield make_input_profile_case(
                handler_name,
                solution_index,
                base_profile,
                value_group,
            )


def input_profile_dimension_values(dimension: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return (dimension,)


def complete_input_profiles(
    handler_name: str,
    base_profile: dict[str, Any],
    dimension_group: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]] | None:
    constraints = group_profile_constraints(
        (
            dimension["profile_model"],
            dimension["dimension"],
            dimension["value"],
        )
        for dimension in dimension_group
    )
    completed_profiles: dict[str, dict[str, Any]] = {}

    for profile_model in profile_models_to_complete(handler_name, constraints):
        model_constraints = constraints.get(profile_model, [])
        if profile_model == "validator_state":
            if is_profile_driven_input_handler(handler_name):
                completed_profile = complete_profile_model_constraints(
                    profile_model,
                    model_constraints,
                )
                if completed_profile is None:
                    return None
                completed_profiles[profile_model] = dict(completed_profile)
                continue
            if not validator_state_constraints_match_base(base_profile, model_constraints):
                return None
            completed_profiles[profile_model] = {
                dimension: base_profile[dimension]
                for dimension in DEFAULT_PROFILE_PARTITION_DIMENSIONS
            }
            continue

        completed_profile = complete_profile_model_constraints(
            profile_model,
            model_constraints,
        )
        if completed_profile is None:
            return None
        completed_profiles[profile_model] = dict(completed_profile)

    return completed_profiles


def profile_models_to_complete(
    handler_name: str,
    constraints: dict[str, list[tuple[str, Any]]],
) -> tuple[str, ...]:
    if handler_name:
        return HANDLER_INPUT_PROFILE_MODELS[handler_name]
    return tuple(constraints)


def validator_state_constraints_match_base(
    base_profile: dict[str, Any],
    constraints: list[tuple[str, Any]],
) -> bool:
    return all(base_profile[dimension] == value for dimension, value in constraints)


def make_input_profile_case(
    handler_name: str,
    solution_index: int,
    base_profile: dict[str, Any],
    dimension_group: tuple[dict[str, Any], ...],
) -> AbstractStateTransitionCase:
    profile = dict(base_profile)
    profile["guide_intent"] = input_profile_guide_intent(dimension_group)
    profile["profile_driven"] = is_profile_driven_input_handler(handler_name)
    input_profiles = complete_input_profiles(handler_name, base_profile, dimension_group)
    if input_profiles is None:
        raise ValueError(f"Cannot complete input profile case: {dimension_group}")
    if "validator_state" in input_profiles:
        profile.update(input_profiles["validator_state"])
    profile["input_profiles"] = input_profiles
    profile["input_profile_constraints"] = input_profile_constraints(dimension_group)
    profile["coverage_tags"] = [
        f"handler:{handler_name}",
        *[input_profile_tag(dimension) for dimension in dimension_group],
    ]
    return make_abstract_case(
        handler_name,
        solution_index,
        profile,
        case_name=input_profile_case_name(dimension_group),
    )


def input_profile_guide_intent(dimension_group: tuple[dict[str, Any], ...]) -> str | None:
    for dimension in dimension_group:
        if dimension["intent"] is not None:
            return dimension["intent"]
    return None


def input_profile_constraints(
    dimension_group: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    constraints = {}
    for dimension in dimension_group:
        constraints.setdefault(dimension["profile_model"], {})[
            dimension["dimension"]
        ] = dimension["value"]
    return constraints


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
    if profile_model == "proposer_slashing_input":
        return proposer_slashing_input_intent(dimension, value)
    if profile_model == "attester_slashing_input":
        return attester_slashing_input_intent(dimension, value)
    if profile_model == "attestation_input":
        return attestation_input_intent(dimension, value)
    if profile_model == "deposit_input":
        return deposit_input_intent(value)
    if profile_model == "bls_to_execution_change_input":
        return bls_to_execution_change_input_intent(dimension, value)
    if profile_model == "voluntary_exit_input":
        return voluntary_exit_input_intent(value)
    if profile_model == "withdrawal_request_input":
        return withdrawal_request_input_intent(value)
    if profile_model == "consolidation_request_input":
        return consolidation_request_input_intent(dimension, value)
    if profile_model == "pending_deposits_input":
        return pending_deposits_input_intent(dimension, value)
    if profile_model == "pending_consolidations_input":
        return pending_consolidations_input_intent(dimension, value)
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


def proposer_slashing_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "header_relation" and value == "SAME_HEADER":
        return "same_header"
    if dimension == "proposer_relation" and value == "DIFFERENT_PROPOSER":
        return "proposer_mismatch"
    if dimension == "proposer_status" and value == "PROPOSER_ALREADY_SLASHED":
        return "already_slashed"
    if (
        dimension == "header_relation"
        and value == "DIFFERENT_HEADERS"
    ):
        return "success"
    return None


def attester_slashing_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "attester_overlap" and value == "DISJOINT":
        return "no_overlap"
    if dimension == "attestation_data_relation" and value == "ATTESTATION_DATA_SAME":
        return "not_slashable_data"
    if dimension == "attester_status" and value == "ATTESTER_ALREADY_SLASHED":
        return "already_slashed"
    if dimension == "attestation_data_relation" and value == "ATTESTATION_DATA_SLASHABLE":
        return "success"
    return None


def attestation_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "slot_relation" and value == "PREVIOUS_EPOCH":
        return "previous_epoch_success"
    if dimension == "slot_relation" and value == "FUTURE_SLOT":
        return "future_slot"
    if dimension == "slot_relation" and value == "CURRENT_EPOCH":
        return "success"
    if dimension == "target_epoch_relation" and value == "WRONG_TARGET_EPOCH":
        return "wrong_target_epoch"
    if dimension == "committee_index_shape" and value == "COMMITTEE_BAD_INDEX":
        return "bad_committee_index"
    if dimension == "aggregation_shape" and value == "AGGREGATION_EMPTY":
        return "empty_aggregation"
    return None


def deposit_input_intent(value: str) -> str | None:
    if value == "NEW_VALIDATOR":
        return "new_validator"
    if value == "TOP_UP_EXISTING_VALIDATOR":
        return "top_up_existing_validator"
    return None


def bls_to_execution_change_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "credential_shape" and value == "EXECUTION_CREDENTIALS":
        return "not_bls_credentials"
    if dimension == "withdrawal_pubkey_relation" and value == "PUBKEY_MISMATCH":
        return "pubkey_mismatch"
    if dimension == "credential_shape" and value == "BLS_CREDENTIALS":
        return "success"
    return None


def voluntary_exit_input_intent(value: str) -> str | None:
    if value == "EXIT_EPOCH_FUTURE":
        return "future_epoch"
    if value == "EXIT_EPOCH_CURRENT":
        return "success"
    return None


def withdrawal_request_input_intent(value: str) -> str | None:
    if value == "FULL_EXIT_REQUEST":
        return "success_full_exit"
    if value == "PARTIAL_WITHDRAWAL_REQUEST":
        return "success_partial_withdrawal"
    return None


def consolidation_request_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "request_kind" and value == "SWITCH_TO_COMPOUNDING_REQUEST":
        return "switch_to_compounding_success"
    if dimension == "request_kind" and value == "CONSOLIDATION_REQUEST":
        return "success"
    if dimension == "target_lookup_shape" and value == "TARGET_MISSING":
        return "target_missing"
    if dimension == "source_activity_shape" and value == "SOURCE_INACTIVE":
        return "source_inactive"
    if dimension == "source_activity_shape" and value == "SOURCE_EXITING":
        return "source_exiting"
    if dimension == "source_activity_shape" and value == "SOURCE_NOT_ACTIVE_LONG_ENOUGH":
        return "source_not_active_long_enough"
    if dimension == "target_activity_shape" and value == "TARGET_INACTIVE":
        return "target_inactive"
    if dimension == "target_activity_shape" and value == "TARGET_EXITING":
        return "target_exiting"
    if dimension == "target_credential_shape" and value == "TARGET_ETH1":
        return "target_not_compounding"
    if dimension == "churn_shape" and value == "CHURN_TOO_LOW":
        return "churn_too_low"
    return None


def pending_deposits_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "deposit_kind" and value == "EXISTING_ACTIVE_VALIDATOR":
        return "success_top_up"
    if dimension == "deposit_kind" and value == "NEW_VALIDATOR":
        return "new_validator"
    if dimension == "deposit_kind" and value == "EXITING_VALIDATOR":
        return "exited_validator_postponed"
    if dimension == "deposit_kind" and value == "WITHDRAWABLE_VALIDATOR":
        return "withdrawable_validator"
    if dimension == "finality_shape" and value == "NOT_FINALIZED":
        return "not_finalized"
    if dimension == "churn_shape" and value == "CHURN_LIMIT_REACHED":
        return "churn_limit_reached"
    if dimension == "bridge_state" and value == "ETH1_BRIDGE_PENDING":
        return "eth1_bridge_blocks_request"
    return None


def pending_consolidations_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "source_shape" and value == "SOURCE_NOT_WITHDRAWABLE":
        return "not_withdrawable"
    if dimension == "source_shape" and value == "SOURCE_SLASHED":
        return "slashed_source_skipped"
    if dimension == "balance_shape" and value == "BALANCE_LESS_THAN_EFFECTIVE_BALANCE":
        return "source_balance_less_than_effective_balance"
    if dimension == "balance_shape" and value == "BALANCE_GREATER_THAN_EFFECTIVE_BALANCE":
        return "source_balance_greater_than_effective_balance"
    if dimension == "queue_shape" and value == "BLOCKED_AFTER_PROCESSED":
        return "blocked_after_processed"
    if dimension == "source_shape" and value == "SOURCE_WITHDRAWABLE":
        return "success"
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
            return {
                "justification_and_finalization": "poor_support",
                "sync_aggregate": "minority_participate",
            }.get(handler_name)
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


def input_profile_case_name(dimension_group: tuple[dict[str, Any], ...]) -> str:
    labels = [
        (
            f"{dimension['profile_model']}_{dimension['dimension']}_"
            f"{safe_case_value(dimension['value'])}"
        )
        for dimension in dimension_group
    ]
    return "input_" + "_x_".join(labels)


def input_profile_tag(dimension: dict[str, Any]) -> str:
    return (
        f"input_profile:{dimension['profile_model']}."
        f"{dimension['dimension']}:{dimension['value']}"
    )


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
