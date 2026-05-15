from __future__ import annotations

import argparse
import sys
from itertools import combinations, product
from typing import Any

from ruamel.yaml import YAML

from .abstract_cases import (
    complete_profile_model_constraints,
    DEFAULT_PROFILE_PARTITION_DIMENSIONS,
    group_profile_constraints,
    HANDLER_INPUT_PROFILE_MODELS,
    HANDLER_NAMES,
    INPUT_PROFILE_DIMENSIONS,
    profile_constraints_compatible,
    profiles_for_compatibility,
    solve_profile_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print raw n-wise input profile combinations for a state-transition handler"
    )
    parser.add_argument("handler", choices=HANDLER_NAMES)
    parser.add_argument("order", type=int, help="Combination order, e.g. 1, 2, or 3.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of combinations to print.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw combinations without MiniZinc solution compatibility filtering.",
    )
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Include a deterministic completed profile row for each handler profile model.",
    )
    args = parser.parse_args()

    if args.order < 1:
        raise ValueError(f"Combination order must be at least 1: {args.order}")

    dimensions = collect_handler_profile_dimensions(args.handler)
    combinations_iter = enumerate_nwise_combinations(
        args.handler,
        dimensions,
        args.order,
        compatible=not args.raw,
        complete=args.complete,
    )
    if args.limit is not None:
        combinations_iter = take(combinations_iter, args.limit)

    yaml = YAML()
    yaml.dump(
        {
            "handler": args.handler,
            "order": args.order,
            "compatible": not args.raw,
            "dimensions": [
                {
                    "profile": dimension.profile,
                    "dimension": dimension.name,
                    "values": list(dimension.values),
                }
                for dimension in dimensions
            ],
            "combinations": list(combinations_iter),
        },
        sys.stdout,
    )


class ProfileDimension:
    def __init__(self, profile: str, name: str, values: tuple[Any, ...]) -> None:
        self.profile = profile
        self.name = name
        self.values = values

    @property
    def key(self) -> str:
        return f"{self.profile}.{self.name}"


def collect_handler_profile_dimensions(handler: str) -> list[ProfileDimension]:
    dimensions = []
    for profile_model in HANDLER_INPUT_PROFILE_MODELS[handler]:
        if profile_model == "validator_state":
            profile_values = profiles_for_compatibility("validator_state")
            dimensions.extend(
                collect_profile_dimensions(
                    profile_model,
                    DEFAULT_PROFILE_PARTITION_DIMENSIONS,
                    profile_values,
                )
            )
        else:
            profile_values = tuple(solve_profile_model(profile_model))
            dimensions.extend(
                collect_profile_dimensions(
                    profile_model,
                    INPUT_PROFILE_DIMENSIONS[profile_model],
                    profile_values,
                )
            )
    return dimensions


def collect_profile_dimensions(
    profile_model: str,
    dimension_names: tuple[str, ...],
    profile_values: tuple[dict[str, Any], ...],
) -> list[ProfileDimension]:
    dimensions = []
    for dimension_name in dimension_names:
        values = sorted(
            {profile[dimension_name] for profile in profile_values},
            key=value_sort_key,
        )
        dimensions.append(ProfileDimension(profile_model, dimension_name, tuple(values)))
    return dimensions


def enumerate_nwise_combinations(
    handler: str,
    dimensions: list[ProfileDimension],
    order: int,
    *,
    compatible: bool,
    complete: bool,
):
    for dimension_group in combinations(dimensions, order):
        value_lists = [dimension.values for dimension in dimension_group]
        for values in product(*value_lists):
            constraints = tuple(
                (dimension.profile, dimension.name, value)
                for dimension, value in zip(dimension_group, values, strict=True)
            )
            if compatible and not profile_constraints_compatible(constraints):
                continue
            combination = {
                dimension.key: value
                for dimension, value in zip(dimension_group, values, strict=True)
            }
            if complete:
                yield {
                    "constraints": combination,
                    "completed_profiles": complete_combination(handler, constraints),
                }
            else:
                yield combination


def complete_combination(
    handler: str,
    constraints: tuple[tuple[str, str, Any], ...],
) -> dict[str, dict[str, Any]]:
    grouped_constraints = group_profile_constraints(constraints)
    completed_profiles = {}
    for profile_model in HANDLER_INPUT_PROFILE_MODELS[handler]:
        completed_profile = complete_profile_model_constraints(
            profile_model,
            grouped_constraints.get(profile_model, []),
        )
        if completed_profile is None:
            continue
        completed_profiles[profile_model] = dict(completed_profile)
    return completed_profiles


def take(items, limit: int):
    for index, item in enumerate(items):
        if index >= limit:
            return
        yield item


def value_sort_key(value: Any) -> tuple[int, str]:
    if isinstance(value, bool):
        return (0 if value else 1, str(value))
    return (0, str(value))


if __name__ == "__main__":
    main()
