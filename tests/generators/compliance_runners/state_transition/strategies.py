from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from tests.generators.compliance_runners.gen_base import (
    AspectAssignment,
    AspectDimension,
    enumerate_strategy,
    n_wise_strategy,
    StrategyCase,
)

from .abstract_cases import (
    complete_input_profiles,
    first_materializable_profiles,
    input_intent_for_dimension,
    input_profile_dimensions_for_handler,
)


@dataclass(frozen=True)
class InputProfileStrategyContext:
    """State-transition data needed to interpret an input-profile strategy."""

    handler_name: str
    solution_index: int
    base_profile: dict[str, Any]
    dimensions: tuple[AspectDimension, ...]


def load_input_profile_strategy_context(handler_name: str) -> InputProfileStrategyContext | None:
    base_profiles = first_materializable_profiles((handler_name,))
    if handler_name not in base_profiles:
        return None

    solution_index, base_profile = base_profiles[handler_name]
    raw_dimensions = input_profile_dimensions_for_handler(handler_name, base_profile)
    grouped_dimensions: OrderedDict[tuple[str, str], list[Any]] = OrderedDict()

    for dimension in raw_dimensions:
        key = (dimension["profile_model"], dimension["dimension"])
        grouped_dimensions.setdefault(key, [])
        if dimension["value"] not in grouped_dimensions[key]:
            grouped_dimensions[key].append(dimension["value"])

    return InputProfileStrategyContext(
        handler_name=handler_name,
        solution_index=solution_index,
        base_profile=base_profile,
        dimensions=tuple(
            AspectDimension(
                model=profile_model,
                name=dimension,
                values=tuple(values),
            )
            for (profile_model, dimension), values in grouped_dimensions.items()
        ),
    )


def input_profile_n_wise_program(
    context: InputProfileStrategyContext,
    *,
    order: int,
    include_lower_orders: bool = False,
):
    """Build a semantic strategy for handler input-profile n-wise coverage."""
    return n_wise_strategy(
        context.dimensions,
        order,
        coverage_kind=f"input_profile:{context.handler_name}",
        include_lower_orders=include_lower_orders,
    )


def input_profile_assignment_to_dimension(
    handler_name: str,
    assignment: AspectAssignment,
) -> dict[str, Any]:
    profile_model = assignment.dimension.model
    if profile_model is None:
        raise ValueError(f"Input-profile assignment has no model: {assignment}")
    return {
        "profile_model": profile_model,
        "dimension": assignment.dimension.name,
        "value": assignment.value,
        "intent": input_intent_for_dimension(
            handler_name,
            profile_model,
            assignment.dimension.name,
            assignment.value,
        ),
    }


def input_profile_assignments_to_dimensions(
    handler_name: str,
    assignments: tuple[AspectAssignment, ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        input_profile_assignment_to_dimension(handler_name, assignment)
        for assignment in assignments
    )


def input_profile_case_is_completable(
    context: InputProfileStrategyContext,
    case: StrategyCase[tuple[AspectAssignment, ...]],
) -> bool:
    dimension_group = input_profile_assignments_to_dimensions(
        context.handler_name,
        case.value,
    )
    return complete_input_profiles(
        context.handler_name,
        context.base_profile,
        dimension_group,
    ) is not None


def enumerate_input_profile_strategy_cases(
    handler_name: str,
    *,
    order: int,
    include_lower_orders: bool = False,
    limit: int | None = None,
) -> Iterable[StrategyCase[tuple[AspectAssignment, ...]]]:
    """Dry-run input-profile strategy cases without materializing vectors."""
    context = load_input_profile_strategy_context(handler_name)
    if context is None:
        return

    program = input_profile_n_wise_program(
        context,
        order=order,
        include_lower_orders=include_lower_orders,
    )
    yield from enumerate_strategy(
        program,
        accepts=lambda case: input_profile_case_is_completable(context, case),
        limit=limit,
    )
