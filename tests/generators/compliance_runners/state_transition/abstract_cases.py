from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from itertools import combinations, product
from typing import Any

from tests.generators.compliance_runners.py_to_mzn import Convertor, get_solutions
from tests.generators.compliance_runners.semantic_gen import make_strategy_goal

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
    "sync_aggregate_input": "sync_aggregate_input.py",
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


@dataclass(frozen=True)
class InputProfileCompletionResult:
    """Completion result for joining selected coverage constraints to input models."""

    completed_profiles: dict[str, dict[str, Any]] | None
    status: str
    reason: str | None = None

    @property
    def matched(self) -> bool:
        return self.completed_profiles is not None


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

DEFAULT_PROFILE_PARTITION_DIMENSIONS = (
    "branch_target",
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
        "branch_target",
        "header_relation",
        "proposer_relation",
        "proposer_status",
    ),
    "attester_slashing_input": (
        "branch_target",
        "attester_overlap",
        "attestation_data_relation",
        "attester_status",
    ),
    "attestation_input": (
        "branch_target",
        "slot_relation",
        "target_epoch_relation",
        "committee_index_shape",
        "aggregation_shape",
    ),
    "deposit_input": ("branch_target", "recipient_shape"),
    "bls_to_execution_change_input": (
        "branch_target",
        "credential_shape",
        "withdrawal_pubkey_relation",
    ),
    "voluntary_exit_input": ("branch_target", "exit_epoch_relation"),
    "withdrawal_request_input": ("request_kind", "branch_target"),
    "consolidation_request_input": (
        "branch_target",
        "request_kind",
        "target_lookup_shape",
        "source_activity_shape",
        "target_activity_shape",
        "target_credential_shape",
        "churn_shape",
    ),
    "pending_deposits_input": (
        "branch_target",
        "deposit_kind",
        "finality_shape",
        "churn_shape",
        "bridge_state",
    ),
    "pending_consolidations_input": (
        "branch_target",
        "source_shape",
        "balance_shape",
        "queue_shape",
    ),
    "sync_aggregate_input": ("branch_target",),
    "queue": (
        "pending_partial_withdrawals",
        "pending_consolidations",
        "pending_deposits",
        "pending_request",
    ),
    "epoch_boundary": ("branch_target", "epoch_boundary_shape"),
    "participation": (
        "branch_target",
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
    "sync_aggregate": ("sync_aggregate_input", "operation_input", "participation"),
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
    selection: str = "enumeration",
    per_handler_limit: int | None = None,
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")
    if order < 2:
        raise ValueError(f"Profile interaction order must be at least 2: {order}")
    if selection not in ("enumeration", "prioritized"):
        raise ValueError(f"Unknown profile interaction selection: {selection}")
    if selection == "prioritized" and per_handler_limit is None:
        raise ValueError("Prioritized profile interaction selection requires a per-handler limit")

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
        handler_cases = list(
            make_profile_interaction_cases_for_handler(
                handler_name,
                candidates[handler_name],
            )
        )
        if selection == "prioritized":
            yield from select_prioritized_profile_interaction_cases(
                handler_cases,
                limit=per_handler_limit,
            )
            continue
        yield from handler_cases


def make_profile_interaction_cases_for_handler(
    handler_name: str,
    candidates: dict[tuple[str, ...], dict[tuple[Any, ...], tuple[int, dict[str, Any]]]],
) -> Iterable[AbstractStateTransitionCase]:
    if not candidates:
        return
    max_values = max(len(group_candidates) for group_candidates in candidates.values())
    for value_index in range(max_values):
        for dimension_group in candidates:
            ordered_values = sorted(
                candidates[dimension_group],
                key=profile_interaction_value_sort_key,
            )
            if value_index >= len(ordered_values):
                continue
            values = ordered_values[value_index]
            solution_index, profile = candidates[dimension_group][values]
            yield make_profile_interaction_case(
                handler_name,
                solution_index,
                profile,
                dimension_group,
                values,
            )


def make_profile_interaction_case(
    handler_name: str,
    solution_index: int,
    profile: dict[str, Any],
    dimension_group: tuple[str, ...],
    values: tuple[Any, ...],
) -> AbstractStateTransitionCase:
    profile_with_tags = dict(profile)
    profile_with_tags["profile_interaction"] = {
        "dimensions": list(dimension_group),
        "values": [str(value) for value in values],
    }
    profile_with_tags["coverage_tags"] = [
        f"handler:{handler_name}",
        profile_interaction_tag(dimension_group, values),
    ]
    return make_abstract_case(
        handler_name,
        solution_index,
        profile_with_tags,
        case_name=profile_interaction_case_name(
            dimension_group,
            values,
            solution_index,
        ),
    )


def select_prioritized_profile_interaction_cases(
    candidates: list[AbstractStateTransitionCase],
    *,
    limit: int | None,
) -> Iterable[AbstractStateTransitionCase]:
    if limit is None or limit >= len(candidates):
        yield from candidates
        return

    remaining = list(candidates)
    selected = []
    covered_values: set[tuple[str, str]] = set()
    covered_groups: set[tuple[str, ...]] = set()
    covered_interactions: set[tuple[tuple[str, str], ...]] = set()

    while remaining and len(selected) < limit:
        best_index = max(
            range(len(remaining)),
            key=lambda index: profile_interaction_priority_score(
                remaining[index],
                covered_values=covered_values,
                covered_groups=covered_groups,
                covered_interactions=covered_interactions,
                original_index=index,
            ),
        )
        case = remaining.pop(best_index)
        selected.append(case)
        values = profile_interaction_case_values(case)
        covered_values.update(values)
        covered_groups.add(tuple(dimension for dimension, _ in values))
        covered_interactions.add(values)

    yield from selected


def profile_interaction_priority_score(
    case: AbstractStateTransitionCase,
    *,
    covered_values: set[tuple[str, str]],
    covered_groups: set[tuple[str, ...]],
    covered_interactions: set[tuple[tuple[str, str], ...]],
    original_index: int,
) -> tuple[int, int, int, int]:
    values = profile_interaction_case_values(case)
    group = tuple(dimension for dimension, _ in values)
    new_group = int(group not in covered_groups)
    new_values = sum(1 for value in values if value not in covered_values)
    new_interaction = int(values not in covered_interactions)
    return (new_values, new_group, new_interaction, -original_index)


def profile_interaction_case_values(
    case: AbstractStateTransitionCase,
) -> tuple[tuple[str, str], ...]:
    interaction = case.profile.get("profile_interaction", {})
    dimensions = interaction.get("dimensions", [])
    values = interaction.get("values", [])
    return tuple(
        (dimension, str(value))
        for dimension, value in zip(dimensions, values, strict=True)
    )


def enumerate_input_profile_cases(
    handlers: Iterable[str],
    *,
    order: int = 1,
    selection: str = "enumeration",
    per_handler_limit: int | None = None,
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")
    if order < 1:
        raise ValueError(f"Input profile order must be at least 1: {order}")
    if selection not in ("enumeration", "prioritized"):
        raise ValueError(f"Unknown input profile selection: {selection}")
    if selection == "prioritized" and per_handler_limit is None:
        raise ValueError("Prioritized input profile selection requires a per-handler limit")

    base_profiles = first_materializable_profiles(requested_handlers)
    for handler_name in requested_handlers:
        if handler_name not in base_profiles:
            continue
        solution_index, base_profile = base_profiles[handler_name]
        dimensions = input_profile_dimensions_for_handler(handler_name, base_profile)
        cases = enumerate_input_profile_dimension_groups(
            handler_name,
            solution_index,
            base_profile,
            dimensions,
            order=order,
        )
        if selection == "prioritized":
            candidates = list(cases)
            yield from select_prioritized_input_profile_cases(
                candidates,
                limit=per_handler_limit,
            )
            continue
        yield from cases


def select_prioritized_input_profile_cases(
    candidates: list[AbstractStateTransitionCase],
    *,
    limit: int | None,
) -> Iterable[AbstractStateTransitionCase]:
    if limit is None or limit >= len(candidates):
        yield from candidates
        return

    remaining = list(candidates)
    selected = []
    covered_intents: set[str] = set()
    covered_values: set[tuple[str, str]] = set()
    covered_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()

    while remaining and len(selected) < limit:
        best_index = max(
            range(len(remaining)),
            key=lambda index: input_profile_priority_score(
                remaining[index],
                covered_intents=covered_intents,
                covered_values=covered_values,
                covered_pairs=covered_pairs,
                original_index=index,
            ),
        )
        case = remaining.pop(best_index)
        selected.append(case)
        intent = case.profile.get("guide_intent")
        if intent is not None:
            covered_intents.add(intent)
        values = input_profile_case_values(case)
        covered_values.update(values)
        covered_pairs.update(input_profile_case_value_pairs(values))

    yield from selected


def input_profile_priority_score(
    case: AbstractStateTransitionCase,
    *,
    covered_intents: set[str],
    covered_values: set[tuple[str, str]],
    covered_pairs: set[tuple[tuple[str, str], tuple[str, str]]],
    original_index: int,
) -> tuple[int, int, int, int, int]:
    intent = case.profile.get("guide_intent")
    values = input_profile_case_values(case)
    pairs = input_profile_case_value_pairs(values)
    new_intent = int(intent is not None and intent not in covered_intents)
    new_values = sum(1 for value in values if value not in covered_values)
    new_pairs = sum(1 for pair in pairs if pair not in covered_pairs)
    has_intent = int(intent is not None)
    # Prefer earlier cases for deterministic tie-breaking.
    return (new_intent, new_values, new_pairs, has_intent, -original_index)


def input_profile_case_values(case: AbstractStateTransitionCase) -> tuple[tuple[str, str], ...]:
    constraints = case.profile.get("input_profile_constraints", {})
    values = []
    for profile_model, profile_values in constraints.items():
        for dimension, value in profile_values.items():
            values.append((f"{profile_model}.{dimension}", str(value)))
    return tuple(sorted(values))


def input_profile_case_value_pairs(
    values: tuple[tuple[str, str], ...],
) -> tuple[tuple[tuple[str, str], tuple[str, str]], ...]:
    return tuple(combinations(values, 2))


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
                "include_in_coverage": True,
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
                    "include_in_coverage": True,
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
    return complete_input_profiles_result(
        handler_name,
        base_profile,
        dimension_group,
    ).completed_profiles


def complete_input_profiles_result(
    handler_name: str,
    base_profile: dict[str, Any],
    dimension_group: tuple[dict[str, Any], ...],
) -> InputProfileCompletionResult:
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
                    return InputProfileCompletionResult(
                        completed_profiles=None,
                        status="uncompletable",
                        reason=f"{profile_model} constraints have no matching row",
                    )
                completed_profiles[profile_model] = dict(completed_profile)
                continue
            if not validator_state_constraints_match_base(base_profile, model_constraints):
                return InputProfileCompletionResult(
                    completed_profiles=None,
                    status="uncompletable",
                    reason="validator_state constraints do not match base profile",
                )
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
            return InputProfileCompletionResult(
                completed_profiles=None,
                status="uncompletable",
                reason=f"{profile_model} constraints have no matching row",
            )
        completed_profiles[profile_model] = dict(completed_profile)

    return InputProfileCompletionResult(
        completed_profiles=completed_profiles,
        status="matched",
    )


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
    completion = complete_input_profiles_result(handler_name, base_profile, dimension_group)
    input_profiles = completion.completed_profiles
    if input_profiles is None:
        raise ValueError(f"Cannot complete input profile case: {dimension_group}")
    profile["input_profile_completion"] = {
        "status": completion.status,
        "reason": completion.reason,
    }
    if "validator_state" in input_profiles:
        profile.update(input_profiles["validator_state"])
    profile["input_profiles"] = input_profiles
    profile["input_profile_constraints"] = input_profile_constraints(dimension_group)
    profile["input_profile_coverage_constraints"] = input_profile_coverage_constraints(
        dimension_group
    )
    strategy_goal = input_profile_strategy_goal(
        handler_name,
        profile["input_profile_coverage_constraints"],
    )
    profile["strategy_goal_id"] = strategy_goal.goal_id
    profile["strategy_goal_kind"] = strategy_goal.kind
    profile["strategy_goal_labels"] = list(strategy_goal.labels)
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


def input_profile_strategy_goal(
    handler_name: str,
    constraints: dict[str, dict[str, Any]],
):
    return make_strategy_goal(
        handler=handler_name,
        kind="input_profile",
        labels=tuple(
            f"{profile_model}.{dimension}:{value}"
            for profile_model, profile_constraints in constraints.items()
            for dimension, value in profile_constraints.items()
        ),
        symbolic=True,
        completable=True,
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


def input_profile_coverage_constraints(
    dimension_group: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    return input_profile_constraints(
        tuple(
            dimension
            for dimension in dimension_group
            if dimension.get("include_in_coverage", True)
        )
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
        return deposit_input_intent(dimension, value)
    if profile_model == "bls_to_execution_change_input":
        return bls_to_execution_change_input_intent(dimension, value)
    if profile_model == "voluntary_exit_input":
        return voluntary_exit_input_intent(dimension, value)
    if profile_model == "withdrawal_request_input":
        return withdrawal_request_input_intent(dimension, value)
    if profile_model == "consolidation_request_input":
        return consolidation_request_input_intent(dimension, value)
    if profile_model == "pending_deposits_input":
        return pending_deposits_input_intent(dimension, value)
    if profile_model == "pending_consolidations_input":
        return pending_consolidations_input_intent(dimension, value)
    if profile_model == "sync_aggregate_input":
        return sync_aggregate_input_intent(dimension, value)
    if profile_model == "queue":
        return queue_input_intent(handler_name, dimension, value)
    if profile_model == "epoch_boundary":
        return epoch_boundary_input_intent(handler_name, dimension, value)
    if profile_model == "participation":
        return participation_input_intent(handler_name, dimension, value)
    if profile_model == "validator_state":
        return validator_state_input_intent(handler_name, dimension, value)
    raise ValueError(f"Unknown input profile model: {profile_model}")


def operation_input_intent(handler_name: str, dimension: str, value: str) -> str | None:
    if dimension == "signature_shape" and value == "SIGNATURE_INVALID":
        return {
            "sync_aggregate": "bad_signature",
        }.get(handler_name)
    if dimension == "lookup_shape" and value == "LOOKUP_MISSING":
        return {
            "withdrawal_request": "pubkey_missing",
        }.get(handler_name)
    if dimension == "source_address_shape" and value == "SOURCE_ADDRESS_INVALID":
        return {
            "withdrawal_request": "bad_source_address",
        }.get(handler_name)
    if (
        dimension == "source_target_relation"
        and value == "SOURCE_TARGET_SAME"
    ):
        return None
    return None


def proposer_slashing_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "PROPOSER_SLASHING_SUCCESS": "success",
            "PROPOSER_SLASHING_SAME_HEADER": "same_header",
            "PROPOSER_SLASHING_PROPOSER_MISMATCH": "proposer_mismatch",
            "PROPOSER_SLASHING_ALREADY_SLASHED": "already_slashed",
            "PROPOSER_SLASHING_BAD_SIGNATURE": "bad_signature",
        }.get(value)
    return None


def attester_slashing_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "ATTESTER_SLASHING_SUCCESS": "success",
            "ATTESTER_SLASHING_NOT_SLASHABLE_DATA": "not_slashable_data",
            "ATTESTER_SLASHING_NO_OVERLAP": "no_overlap",
            "ATTESTER_SLASHING_ALREADY_SLASHED": "already_slashed",
            "ATTESTER_SLASHING_BAD_SIGNATURE": "bad_signature",
        }.get(value)
    return None


def attestation_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "ATTESTATION_SUCCESS": "success",
            "ATTESTATION_PREVIOUS_EPOCH_SUCCESS": "previous_epoch_success",
            "ATTESTATION_FUTURE_SLOT": "future_slot",
            "ATTESTATION_WRONG_TARGET_EPOCH": "wrong_target_epoch",
            "ATTESTATION_BAD_COMMITTEE_INDEX": "bad_committee_index",
            "ATTESTATION_EMPTY_AGGREGATION": "empty_aggregation",
            "ATTESTATION_BAD_SIGNATURE": "bad_signature",
        }.get(value)
    return None


def deposit_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "DEPOSIT_NEW_VALIDATOR": "new_validator",
            "DEPOSIT_TOP_UP_EXISTING_VALIDATOR": "top_up_existing_validator",
            "DEPOSIT_INVALID_PROOF": "invalid_proof",
            "DEPOSIT_INVALID_SIGNATURE": "invalid_signature",
        }.get(value)
    return None


def bls_to_execution_change_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "BLS_TO_EXECUTION_SUCCESS": "success",
            "BLS_TO_EXECUTION_OUT_OF_RANGE": "out_of_range",
            "BLS_TO_EXECUTION_NOT_BLS_CREDENTIALS": "not_bls_credentials",
            "BLS_TO_EXECUTION_PUBKEY_MISMATCH": "pubkey_mismatch",
            "BLS_TO_EXECUTION_BAD_SIGNATURE": "bad_signature",
        }.get(value)
    return None


def voluntary_exit_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "VOLUNTARY_EXIT_SUCCESS": "success",
            "VOLUNTARY_EXIT_INACTIVE": "inactive",
            "VOLUNTARY_EXIT_ALREADY_EXITED": "already_exited",
            "VOLUNTARY_EXIT_FUTURE_EPOCH": "future_epoch",
            "VOLUNTARY_EXIT_NOT_ACTIVE_LONG_ENOUGH": "not_active_long_enough",
            "VOLUNTARY_EXIT_PENDING_WITHDRAWAL": "pending_withdrawal",
        }.get(value)
    return None


def withdrawal_request_input_intent(dimension: str, value: str) -> str | None:
    if dimension != "branch_target":
        return None
    return {
        "WITHDRAWAL_QUEUE_FULL_PARTIAL": "queue_full",
        "WITHDRAWAL_PUBKEY_MISSING": "pubkey_missing",
        "WITHDRAWAL_BAD_SOURCE_ADDRESS": "bad_source_address",
        "WITHDRAWAL_SOURCE_INACTIVE": "source_inactive",
        "WITHDRAWAL_SOURCE_EXITING": "source_exiting",
        "WITHDRAWAL_NOT_ACTIVE_LONG_ENOUGH": "not_active_long_enough",
        "WITHDRAWAL_FULL_EXIT_PENDING_WITHDRAWAL": "full_exit_with_pending_withdrawal",
        "WITHDRAWAL_FULL_EXIT_SUCCESS": "success_full_exit",
        "WITHDRAWAL_PARTIAL_CONDITIONS_NOT_MET": "partial_conditions_not_met",
        "WITHDRAWAL_PARTIAL_SUCCESS": "success_partial_withdrawal",
    }.get(value)


def consolidation_request_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "CONSOLIDATION_SWITCH_SUCCESS": "switch_to_compounding_success",
            "CONSOLIDATION_SWITCH_PUBKEY_MISSING": "switch_pubkey_missing",
            "CONSOLIDATION_SWITCH_BAD_SOURCE_ADDRESS": "switch_bad_source_address",
            "CONSOLIDATION_SWITCH_SOURCE_INACTIVE": "switch_source_inactive",
            "CONSOLIDATION_SWITCH_SOURCE_EXITING": "switch_source_exiting",
            "CONSOLIDATION_SOURCE_EQUALS_TARGET": "source_equals_target",
            "CONSOLIDATION_QUEUE_FULL": "queue_full",
            "CONSOLIDATION_CHURN_TOO_LOW": "churn_too_low",
            "CONSOLIDATION_SOURCE_MISSING": "source_missing",
            "CONSOLIDATION_TARGET_MISSING": "target_missing",
            "CONSOLIDATION_BAD_SOURCE_ADDRESS": "bad_source_address",
            "CONSOLIDATION_TARGET_NOT_COMPOUNDING": "target_not_compounding",
            "CONSOLIDATION_SOURCE_INACTIVE": "source_inactive",
            "CONSOLIDATION_TARGET_INACTIVE": "target_inactive",
            "CONSOLIDATION_SOURCE_EXITING": "source_exiting",
            "CONSOLIDATION_TARGET_EXITING": "target_exiting",
            "CONSOLIDATION_SOURCE_NOT_ACTIVE_LONG_ENOUGH": "source_not_active_long_enough",
            "CONSOLIDATION_SOURCE_PENDING_WITHDRAWAL": "source_pending_withdrawal",
            "CONSOLIDATION_SUCCESS": "success",
        }.get(value)
    return None


def pending_deposits_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "PENDING_DEPOSITS_SUCCESS_TOP_UP": "success_top_up",
            "PENDING_DEPOSITS_NOT_FINALIZED": "not_finalized",
            "PENDING_DEPOSITS_CHURN_LIMIT_REACHED": "churn_limit_reached",
            "PENDING_DEPOSITS_EXITED_VALIDATOR_POSTPONED": "exited_validator_postponed",
            "PENDING_DEPOSITS_WITHDRAWABLE_VALIDATOR": "withdrawable_validator",
            "PENDING_DEPOSITS_ETH1_BRIDGE_BLOCKS_REQUEST": "eth1_bridge_blocks_request",
            "PENDING_DEPOSITS_MAX_PER_EPOCH_REACHED": "max_per_epoch_reached",
            "PENDING_DEPOSITS_NEW_VALIDATOR": "new_validator",
        }.get(value)
    return None


def pending_consolidations_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "PENDING_CONSOLIDATIONS_EMPTY_QUEUE": "empty_queue",
            "PENDING_CONSOLIDATIONS_SUCCESS": "success",
            "PENDING_CONSOLIDATIONS_NOT_WITHDRAWABLE": "not_withdrawable",
            "PENDING_CONSOLIDATIONS_SLASHED_SOURCE_SKIPPED": "slashed_source_skipped",
            "PENDING_CONSOLIDATIONS_BALANCE_LESS_THAN_EFFECTIVE_BALANCE": (
                "source_balance_less_than_effective_balance"
            ),
            "PENDING_CONSOLIDATIONS_BALANCE_GREATER_THAN_EFFECTIVE_BALANCE": (
                "source_balance_greater_than_effective_balance"
            ),
            "PENDING_CONSOLIDATIONS_BLOCKED_AFTER_PROCESSED": "blocked_after_processed",
        }.get(value)
    return None


def sync_aggregate_input_intent(dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "SYNC_AGGREGATE_ALL_PARTICIPATE": "all_participate",
            "SYNC_AGGREGATE_MAJORITY_PARTICIPATE": "majority_participate",
            "SYNC_AGGREGATE_MINORITY_PARTICIPATE": "minority_participate",
            "SYNC_AGGREGATE_NONE_PARTICIPATE": "none_participate",
            "SYNC_AGGREGATE_BAD_SIGNATURE": "bad_signature",
        }.get(value)
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
        return None
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
        }.get(handler_name)
    if dimension == "pending_request" and value == "REQUEST_CONSOLIDATION":
        return {"pending_consolidations": "success"}.get(handler_name)
    return None


def epoch_boundary_input_intent(handler_name: str, dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "EPOCH_GENESIS_SKIP": {
                "justification_and_finalization": "genesis_skip",
                "inactivity_updates": "genesis_skip",
                "rewards_and_penalties": "genesis_skip",
            },
            "ETH1_DATA_PERIOD_BOUNDARY": {"eth1_data_reset": "period_boundary"},
            "ETH1_DATA_NON_BOUNDARY": {"eth1_data_reset": "non_boundary"},
            "HISTORICAL_SUMMARIES_PERIOD_BOUNDARY": {
                "historical_summaries_update": "period_boundary"
            },
            "HISTORICAL_SUMMARIES_NON_BOUNDARY": {
                "historical_summaries_update": "non_boundary"
            },
            "SYNC_COMMITTEE_PERIOD_BOUNDARY": {"sync_committee_updates": "period_boundary"},
            "SYNC_COMMITTEE_GENESIS_PERIOD_BOUNDARY": {
                "sync_committee_updates": "genesis_period_boundary"
            },
            "SYNC_COMMITTEE_NON_BOUNDARY": {"sync_committee_updates": "non_boundary"},
            "SLASHINGS_RESET_NONZERO": {"slashings_reset": "reset_nonzero"},
            "SLASHINGS_RESET_ALREADY_ZERO": {"slashings_reset": "already_zero"},
            "RANDAO_RESET_TO_CURRENT_MIX": {"randao_mixes_reset": "reset_to_current_mix"},
            "RANDAO_ALREADY_CURRENT_MIX": {"randao_mixes_reset": "already_current_mix"},
        }.get(value, {}).get(handler_name)
    return None


def participation_input_intent(handler_name: str, dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "PARTICIPATION_FINALITY_GENESIS_SKIP": {"justification_and_finalization": "genesis_skip"},
            "PARTICIPATION_FINALITY_CURRENT_JUSTIFIED": {
                "justification_and_finalization": "current_justified"
            },
            "PARTICIPATION_FINALITY_PREVIOUS_JUSTIFIED": {
                "justification_and_finalization": "previous_justified"
            },
            "PARTICIPATION_FINALITY_POOR_SUPPORT": {"justification_and_finalization": "poor_support"},
            "PARTICIPATION_FINALITY_FINALIZE_CURRENT": {"justification_and_finalization": "finalize_current"},
            "PARTICIPATION_FINALITY_FINALIZE_234": {"justification_and_finalization": "finalize_234"},
            "PARTICIPATION_FINALITY_FINALIZE_23": {"justification_and_finalization": "finalize_23"},
            "PARTICIPATION_FINALITY_FINALIZE_123": {"justification_and_finalization": "finalize_123"},
            "PARTICIPATION_INACTIVITY_GENESIS_SKIP": {"inactivity_updates": "genesis_skip"},
            "PARTICIPATION_INACTIVITY_PARTICIPATING_RECOVERY": {
                "inactivity_updates": "participating_recovery"
            },
            "PARTICIPATION_INACTIVITY_NON_PARTICIPATING_NO_LEAK": {
                "inactivity_updates": "non_participating_no_leak"
            },
            "PARTICIPATION_INACTIVITY_NON_PARTICIPATING_LEAK": {
                "inactivity_updates": "non_participating_leak"
            },
            "PARTICIPATION_REWARDS_GENESIS_SKIP": {"rewards_and_penalties": "genesis_skip"},
            "PARTICIPATION_REWARDS_FULL_PARTICIPATION": {
                "rewards_and_penalties": "full_participation_reward"
            },
            "PARTICIPATION_REWARDS_EMPTY_PARTICIPATION": {
                "rewards_and_penalties": "empty_participation_penalty"
            },
            "PARTICIPATION_REWARDS_INACTIVITY_LEAK_PENALTY": {
                "rewards_and_penalties": "inactivity_leak_penalty"
            },
            "PARTICIPATION_REWARDS_INACTIVITY_LEAK_FULL_PARTICIPATION": {
                "rewards_and_penalties": "inactivity_leak_full_participation"
            },
            "PARTICIPATION_FLAGS_ALL_ZERO": {"participation_flag_updates": "all_zero"},
            "PARTICIPATION_FLAGS_CURRENT_FILLED": {
                "participation_flag_updates": "current_filled"
            },
            "PARTICIPATION_FLAGS_PREVIOUS_FILLED": {
                "participation_flag_updates": "previous_filled"
            },
        }.get(value, {}).get(handler_name)
    return None


def validator_state_input_intent(handler_name: str, dimension: str, value: str) -> str | None:
    if dimension == "branch_target":
        return {
            "DEPOSIT_REQUEST_START_INDEX_UNSET": {
                "deposit_request": "start_index_unset",
            },
            "DEPOSIT_REQUEST_START_INDEX_SET": {
                "deposit_request": "start_index_set",
            },
            "REGISTRY_NO_CHANGE": {"registry_updates": "no_change"},
            "REGISTRY_ACTIVATION_QUEUE": {"registry_updates": "activation_queue"},
            "REGISTRY_EJECTION": {"registry_updates": "ejection"},
            "REGISTRY_ACTIVATION": {"registry_updates": "activation"},
            "SLASHINGS_NO_SLASHED_VALIDATORS": {"slashings": "no_slashed_validators"},
            "SLASHINGS_PENALTY_APPLIED": {"slashings": "penalty_applied"},
            "SLASHINGS_WRONG_WITHDRAWABLE_EPOCH": {"slashings": "wrong_withdrawable_epoch"},
            "SLASHINGS_ZERO_SLASHING_BALANCE": {"slashings": "zero_slashing_balance"},
            "EFFECTIVE_BALANCE_NO_CHANGE_AT_THRESHOLD": {
                "effective_balance_updates": "no_change_at_threshold"
            },
            "EFFECTIVE_BALANCE_STEP_DOWN": {"effective_balance_updates": "step_down"},
            "EFFECTIVE_BALANCE_STEP_UP": {"effective_balance_updates": "step_up"},
            "EFFECTIVE_BALANCE_CAP_AT_MAX": {"effective_balance_updates": "cap_at_max"},
        }.get(value, {}).get(handler_name)
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
