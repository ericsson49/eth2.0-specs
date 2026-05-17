from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .generate_vectors import normalize_handlers
from .suite_config import (
    read_yaml,
    resolve_campaign_config_path,
    resolve_suite_config_path,
    suite_name_from_config,
)


@dataclass(frozen=True)
class InputProfileStrategyFormula:
    """Authored input-profile strategy formula."""

    handlers: tuple[str, ...]
    order: int
    include_lower_orders: bool


@dataclass(frozen=True)
class NamedInputProfileStrategyFormula:
    """Named input-profile formula from a suite, campaign entry, or scratch file."""

    name: str
    formula: InputProfileStrategyFormula


def input_profile_formula_from_generation_config(
    generation_config: dict[str, Any],
) -> InputProfileStrategyFormula | None:
    if generation_config.get("mode") != "input_profile":
        return None
    order = int(generation_config.get("input_profile_order", 1))
    return InputProfileStrategyFormula(
        handlers=tuple(
            normalize_handlers(
                generation_config.get("handlers"),
                stages=generation_config.get("stages"),
            )
        ),
        order=order,
        include_lower_orders=bool(generation_config.get("include_lower_orders", order > 1)),
    )


def input_profile_formula_from_data(data: dict[str, Any]) -> InputProfileStrategyFormula:
    strategy = data.get("strategy") or data.get("kind") or data.get("mode")
    if strategy != "input_profile":
        raise ValueError(f"Unsupported strategy formula: {strategy}")
    order = int(data.get("order", data.get("input_profile_order", 1)))
    return InputProfileStrategyFormula(
        handlers=tuple(
            normalize_handlers(
                data.get("handlers"),
                stages=data.get("stages"),
            )
        ),
        order=order,
        include_lower_orders=bool(data.get("include_lower_orders", order > 1)),
    )


def load_input_profile_formula(path: Path) -> InputProfileStrategyFormula:
    data = read_yaml(path)
    if "generation" in data:
        formula = input_profile_formula_from_generation_config(data["generation"])
        if formula is None:
            raise ValueError(f"Suite generation config is not input_profile: {path}")
        return formula
    return input_profile_formula_from_data(data)


def load_input_profile_formula_from_suite(suite: str) -> InputProfileStrategyFormula:
    suite_config = read_yaml(resolve_suite_config_path(suite))
    formula = input_profile_formula_from_generation_config(suite_config["generation"])
    if formula is None:
        raise ValueError(f"Suite generation config is not input_profile: {suite}")
    return formula


def load_named_input_profile_formula_from_suite(suite: str) -> NamedInputProfileStrategyFormula:
    suite_config_path = resolve_suite_config_path(suite)
    suite_config = read_yaml(suite_config_path)
    formula = input_profile_formula_from_generation_config(suite_config["generation"])
    if formula is None:
        raise ValueError(f"Suite generation config is not input_profile: {suite}")
    return NamedInputProfileStrategyFormula(
        name=suite_name_from_config(suite_config, suite_config_path),
        formula=formula,
    )


def load_named_input_profile_formulas_from_campaign(
    campaign: str,
) -> tuple[NamedInputProfileStrategyFormula, ...]:
    campaign_config = read_yaml(resolve_campaign_config_path(campaign))
    formulas = []
    for suite_entry in campaign_config["suites"]:
        suite_name = suite_entry if isinstance(suite_entry, str) else suite_entry["suite"]
        suite_config_path = resolve_suite_config_path(suite_name)
        suite_config = read_yaml(suite_config_path)
        formula = input_profile_formula_from_generation_config(suite_config["generation"])
        if formula is None:
            continue
        formulas.append(
            NamedInputProfileStrategyFormula(
                name=suite_name_from_config(suite_config, suite_config_path),
                formula=formula,
            )
        )
    return tuple(formulas)


def input_profile_formulas_from_generation_configs(
    generation_configs: Iterable[dict[str, Any]],
) -> Iterable[InputProfileStrategyFormula]:
    for generation_config in generation_configs:
        formula = input_profile_formula_from_generation_config(generation_config)
        if formula is not None:
            yield formula
