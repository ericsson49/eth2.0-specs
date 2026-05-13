from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from eth_consensus_specs.test.context import spec_state_test, with_electra_and_later
from eth_consensus_specs.test.helpers.constants import ELECTRA, MINIMAL
from tests.generators.compliance_runners.gen_base.output import dump_test_case_result
from tests.generators.compliance_runners.gen_base.pytest_support import configure_generator_context
from tests.infra.dumper import Dumper

from .abstract_cases import (
    enumerate_guided_operation_cases,
    enumerate_materializable_operation_cases,
    select_abstract_cases,
)
from .materializers import materialize_case, MATERIALIZED_HANDLER_NAMES, UnsupportedProfileError

configure_generator_context()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate state-transition compliance vectors from MiniZinc abstract cases"
    )
    parser.add_argument("--output", type=Path, default=Path("comptests"))
    parser.add_argument("--fork", default=ELECTRA)
    parser.add_argument("--preset", default=MINIMAL)
    parser.add_argument("--per-handler-limit", type=int, default=5)
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Only write cases where the operation changes post-state.",
    )
    parser.add_argument(
        "--unchanged-only",
        action="store_true",
        help="Only write valid cases where the operation leaves post-state unchanged.",
    )
    parser.add_argument(
        "--invalid-only",
        action="store_true",
        help="Only write invalid operation cases that are expected to raise an assertion.",
    )
    parser.add_argument(
        "--guided",
        action="store_true",
        help="Generate handler-specific guard-intent cases for coverage evaluation.",
    )
    parser.add_argument(
        "--handler",
        action="append",
        help=(
            "Handler to generate. Defaults to withdrawal_request. Can be repeated. "
            "Use 'all' for every materialized state-transition handler."
        ),
    )
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    generate_vectors(
        output_dir=args.output,
        fork_name=args.fork,
        preset_name=args.preset,
        handlers=normalize_handlers(args.handler),
        per_handler_limit=args.per_handler_limit,
        changed_only=args.changed_only,
        unchanged_only=args.unchanged_only,
        invalid_only=args.invalid_only,
        guided=args.guided,
        keep_existing=args.keep_existing,
    )


def normalize_handlers(handlers: list[str] | None) -> list[str]:
    requested_handlers = handlers or ["withdrawal_request"]
    if "all" in requested_handlers:
        requested_handlers = list(MATERIALIZED_HANDLER_NAMES) + [
            handler for handler in requested_handlers if handler != "all"
        ]
    return list(dict.fromkeys(requested_handlers))


def generate_vectors(
    *,
    output_dir: Path,
    fork_name: str,
    preset_name: str,
    handlers: list[str],
    per_handler_limit: int,
    changed_only: bool,
    unchanged_only: bool,
    invalid_only: bool,
    guided: bool,
    keep_existing: bool,
) -> None:
    selected_filters = sum([changed_only, unchanged_only, invalid_only])
    if selected_filters > 1:
        raise ValueError("--changed-only, --unchanged-only, and --invalid-only are exclusive")

    dumper = Dumper()
    if guided:
        abstract_cases = enumerate_guided_operation_cases(handlers=handlers)
    elif changed_only or invalid_only:
        abstract_cases = enumerate_materializable_operation_cases(handlers=handlers)
    else:
        abstract_cases = select_abstract_cases(
            per_handler_limit=per_handler_limit,
            handlers=handlers,
        )
    written_counts = {handler: 0 for handler in handlers}

    for abstract_case in abstract_cases:
        if written_counts[abstract_case.handler_name] >= per_handler_limit:
            continue
        result = materialize_with_base_state(
            abstract_case,
            fork_name=fork_name,
            preset_name=preset_name,
            invalid_operation=invalid_only,
        )
        if result is None:
            continue
        if changed_only and not is_changed_post_state(result):
            continue
        if unchanged_only and not is_unchanged_post_state(result):
            continue
        if invalid_only and is_operation_valid(result):
            continue
        result.test_case.set_output_dir(str(output_dir))
        if result.test_case.dir.exists() and not keep_existing:
            shutil.rmtree(result.test_case.dir)
        dump_test_case_result(result, dumper)
        written_counts[abstract_case.handler_name] += 1
        if all(count >= per_handler_limit for count in written_counts.values()):
            return


def materialize_with_base_state(
    abstract_case,
    *,
    fork_name: str,
    preset_name: str,
    invalid_operation: bool,
):
    @with_electra_and_later
    @spec_state_test
    def get_result(spec, state):
        try:
            yield (
                materialize_case(
                    spec,
                    state,
                    abstract_case,
                    fork_name=fork_name,
                    preset_name=preset_name,
                    invalid_operation=invalid_operation,
                ),
            )
        except UnsupportedProfileError:
            return

    results = list(get_result(phase=fork_name, preset=preset_name, bls_active=False))
    if not results:
        return None
    ((result,),) = results
    return result


def is_changed_post_state(result) -> bool:
    parts = {name: data for name, kind, data in result.case_parts if kind == "ssz"}
    if "post" not in parts:
        return False
    return parts["pre"] != parts["post"]


def is_unchanged_post_state(result) -> bool:
    parts = {name: data for name, kind, data in result.case_parts if kind == "ssz"}
    if "post" not in parts:
        return False
    return parts["pre"] == parts["post"]


def is_operation_valid(result) -> bool:
    return bool(result.meta["operation_valid"])


if __name__ == "__main__":
    main()
