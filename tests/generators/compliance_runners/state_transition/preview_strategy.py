from __future__ import annotations

import argparse
from collections.abc import Iterable

from tests.generators.compliance_runners.gen_base import enumerate_strategy

from .abstract_cases import HANDLER_NAMES
from .strategies import (
    enumerate_input_profile_strategy_cases,
    input_profile_n_wise_program,
    load_input_profile_strategy_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview semantic state-transition generation strategies."
    )
    parser.add_argument(
        "handlers",
        nargs="*",
        default=HANDLER_NAMES,
        help="Handlers to preview. Defaults to all known handlers.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="n-wise input-profile interaction order.",
    )
    parser.add_argument(
        "--include-lower-orders",
        action="store_true",
        help="Include 1-wise .. n-wise combinations instead of exactly n-wise.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        help="Print up to N completed semantic cases for each handler.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown_handlers = set(args.handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")

    print("| handler | dimensions | symbolic | completable |")
    print("| --- | ---: | ---: | ---: |")
    for handler_name in args.handlers:
        print_handler_summary(
            handler_name,
            order=args.order,
            include_lower_orders=args.include_lower_orders,
            show=args.show,
        )


def print_handler_summary(
    handler_name: str,
    *,
    order: int,
    include_lower_orders: bool,
    show: int,
) -> None:
    context = load_input_profile_strategy_context(handler_name)
    if context is None:
        print(f"| {handler_name} | 0 | 0 | 0 |")
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
        f"| {handler_name} | {len(context.dimensions)} | "
        f"{symbolic_count} | {len(completable_cases)} |"
    )
    for case in completable_cases[:show]:
        print(f"  - {format_case_labels(assignment.label() for assignment in case.value)}")


def format_case_labels(labels: Iterable[str]) -> str:
    return ", ".join(labels)


if __name__ == "__main__":
    main()
