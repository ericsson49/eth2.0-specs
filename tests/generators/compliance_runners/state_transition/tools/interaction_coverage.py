from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

DEFAULT_INTERACTION_DIMENSIONS = ("stage", "runner", "handler", "intent", "outcome")
DEFAULT_MAX_ORDER = 2


def interaction_settings(ontology: dict[str, Any]) -> tuple[tuple[str, ...], int]:
    observed = ontology.get("interaction_coverage", {}).get("observed", {})
    dimensions = tuple(
        observed.get("dimensions", {}).get("common", DEFAULT_INTERACTION_DIMENSIONS)
    )
    max_order = int(observed.get("max_order", DEFAULT_MAX_ORDER))
    return dimensions, max_order


def add_stage_dimension(
    cases: list[dict[str, Any]],
    stages: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    return [
        {
            **case,
            "stage": stage_for_handler(str(case["handler"]), stages),
        }
        for case in cases
    ]


def stage_for_handler(handler: str, stages: dict[str, tuple[str, ...]]) -> str:
    for stage_name, handlers in stages.items():
        if handler in handlers:
            return stage_name
    return "ungrouped"


def collect_observed_interactions(
    cases: list[dict[str, Any]],
    dimensions: tuple[str, ...],
    *,
    max_order: int,
) -> dict[tuple[str, ...], Counter[tuple[str, ...]]]:
    interactions = {}
    upper_order = min(max_order, len(dimensions))
    for order in range(2, upper_order + 1):
        for dimension_group in combinations(dimensions, order):
            counter: Counter[tuple[str, ...]] = Counter()
            for case in cases:
                values = tuple(dimension_value(case, dimension) for dimension in dimension_group)
                if all(value is not None for value in values):
                    counter[values] += 1
            interactions[dimension_group] = counter
    return interactions


def dimension_value(case: dict[str, Any], dimension: str) -> str | None:
    if dimension == "intent":
        value = case.get("guide_intent")
    else:
        value = case.get(dimension)
    if value is None:
        profile = case.get("profile", {})
        if isinstance(profile, dict):
            value = profile.get(dimension)
    if value is None:
        return None
    return str(value)


def format_interaction_report(
    cases: list[dict[str, Any]],
    dimensions: tuple[str, ...],
    *,
    max_order: int,
) -> str:
    lines = [
        "Observed Interaction Coverage",
        "=============================",
        "",
        f"dimensions: {', '.join(dimensions)}",
        f"max order: {max_order}",
        "",
    ]
    interactions = collect_observed_interactions(cases, dimensions, max_order=max_order)
    if not interactions:
        lines.append("No interaction dimensions configured.")
        return "\n".join(lines)

    for dimension_group, counter in interactions.items():
        label = " x ".join(dimension_group)
        lines.append(label)
        lines.append("-" * len(label))
        lines.append(f"observed combinations: {len(counter)}")
        for values, count in sorted(counter.items()):
            lines.append(f"  {' x '.join(values)}: {count}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_interaction_summary(
    cases: list[dict[str, Any]],
    dimensions: tuple[str, ...],
    *,
    max_order: int,
) -> list[str]:
    lines = ["Observed Interactions", "---------------------"]
    interactions = collect_observed_interactions(cases, dimensions, max_order=max_order)
    if not interactions:
        lines.append("not configured")
        return lines

    lines.append(f"dimensions: {', '.join(dimensions)}")
    lines.append(f"max order: {max_order}")
    for dimension_group, counter in interactions.items():
        label = " x ".join(dimension_group)
        lines.append(f"{label}: {len(counter)} observed combinations")
    return lines
