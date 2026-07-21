from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from tests.generators.compliance_runners.semantic_gen import enumerate_strategy

from ..abstract_cases import HANDLER_NAMES
from ..goal_ledger import ExpectedGoal, write_goal_ledger
from ..strategies import (
    enumerate_input_profile_strategy_cases,
    enumerate_input_profile_strategy_goals,
    input_profile_n_wise_program,
    load_input_profile_strategy_context,
)
from ..strategy_formula import (
    InputProfileStrategyFormula,
    load_input_profile_formula,
    load_named_input_profile_formula_from_suite,
    load_named_input_profile_formulas_from_campaign,
    NamedInputProfileStrategyFormula,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview semantic state-transition generation strategies."
    )
    parser.add_argument(
        "handlers",
        nargs="*",
        default=HANDLER_NAMES,
        help=(
            "Handlers to preview when no formula or suite is supplied. "
            "Defaults to all known handlers."
        ),
    )
    formula_source = parser.add_mutually_exclusive_group()
    formula_source.add_argument(
        "--formula",
        type=Path,
        help=(
            "YAML formula to preview. Useful for experiments before committing "
            "the formula to a suite config."
        ),
    )
    formula_source.add_argument(
        "--suite",
        help="Suite config name or path to preview.",
    )
    formula_source.add_argument(
        "--campaign",
        help="Campaign config name or path to preview.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="n-wise input-profile interaction order.",
    )
    parser.add_argument(
        "--include-lower-orders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include 1-wise .. n-wise combinations instead of exactly n-wise. "
            "Defaults to true to match input_profile generation."
        ),
    )
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        help="Print up to N completed semantic cases for each handler.",
    )
    parser.add_argument(
        "--goals-output",
        type=Path,
        help="Write symbolic and completable strategy goals to a JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formulas = resolve_formulas(args)

    if args.goals_output is not None:
        write_expected_goals(
            args.goals_output,
            formulas=formulas,
        )

    print("| formula | handler | order | dimensions | symbolic | completable |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for named_formula in formulas:
        for handler_name in named_formula.formula.handlers:
            print_handler_summary(
                named_formula.name,
                handler_name,
                order=named_formula.formula.order,
                include_lower_orders=named_formula.formula.include_lower_orders,
                show=args.show,
            )


def resolve_formulas(args: argparse.Namespace) -> tuple[NamedInputProfileStrategyFormula, ...]:
    if args.formula is not None:
        return (
            NamedInputProfileStrategyFormula(
                name=args.formula.stem,
                formula=load_input_profile_formula(args.formula),
            ),
        )
    if args.suite is not None:
        return (load_named_input_profile_formula_from_suite(args.suite),)
    if args.campaign is not None:
        return load_named_input_profile_formulas_from_campaign(args.campaign)

    unknown_handlers = set(args.handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")
    return (
        NamedInputProfileStrategyFormula(
            name="ad_hoc",
            formula=InputProfileStrategyFormula(
                handlers=tuple(args.handlers),
                order=args.order,
                include_lower_orders=args.include_lower_orders,
            ),
        ),
    )


def print_handler_summary(
    formula_name: str,
    handler_name: str,
    *,
    order: int,
    include_lower_orders: bool,
    show: int,
) -> None:
    context = load_input_profile_strategy_context(handler_name)
    if context is None:
        print(f"| {formula_name} | {handler_name} | {order} | 0 | 0 | 0 |")
        return

    program = input_profile_n_wise_program(
        context,
        order=order,
        include_lower_orders=include_lower_orders,
    )
    symbolic_count = sum(1 for _case in enumerate_strategy(program))
    completable_cases = tuple(
        enumerate_input_profile_strategy_cases(
            handler_name,
            order=order,
            include_lower_orders=include_lower_orders,
        )
    )
    print(
        f"| {formula_name} | {handler_name} | {order} | {len(context.dimensions)} | "
        f"{symbolic_count} | {len(completable_cases)} |"
    )
    for case in completable_cases[:show]:
        print(f"  - {format_case_labels(assignment.label() for assignment in case.value)}")


def format_case_labels(labels: Iterable[str]) -> str:
    return ", ".join(labels)


def write_expected_goals(
    output_path: Path,
    *,
    formulas: tuple[NamedInputProfileStrategyFormula, ...],
) -> None:
    goals_by_id = {}
    for named_formula in formulas:
        formula = named_formula.formula
        for handler_name in formula.handlers:
            for goal in enumerate_input_profile_strategy_goals(
                handler_name,
                order=formula.order,
                include_lower_orders=formula.include_lower_orders,
            ):
                goals_by_id[goal.goal_id] = ExpectedGoal(
                    goal_id=goal.goal_id,
                    handler=goal.handler,
                    kind=goal.kind,
                    labels=goal.labels,
                    symbolic=goal.symbolic,
                    completable=goal.completable,
                )
    write_goal_ledger(
        output_path,
        list(goals_by_id.values()),
        metadata={
            "formulas": [
                {
                    "name": named_formula.name,
                    "handlers": list(named_formula.formula.handlers),
                    "order": named_formula.formula.order,
                    "include_lower_orders": named_formula.formula.include_lower_orders,
                }
                for named_formula in formulas
            ],
        },
    )


if __name__ == "__main__":
    main()
