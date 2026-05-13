from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from eth_consensus_specs.test.context import spec_state_test, with_electra_and_later
from eth_consensus_specs.test.helpers.constants import ELECTRA, MINIMAL
from tests.generators.compliance_runners.gen_base.output import dump_test_case_result
from tests.generators.compliance_runners.gen_base.pytest_support import configure_generator_context
from tests.infra.dumper import Dumper

from .abstract_cases import enumerate_matching_abstract_cases, select_abstract_cases
from .materializers import materialize_case, UnsupportedProfileError

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
        "--handler",
        action="append",
        help="Handler to generate. Defaults to withdrawal_request. Can be repeated.",
    )
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    generate_vectors(
        output_dir=args.output,
        fork_name=args.fork,
        preset_name=args.preset,
        handlers=args.handler or ["withdrawal_request"],
        per_handler_limit=args.per_handler_limit,
        changed_only=args.changed_only,
        keep_existing=args.keep_existing,
    )


def generate_vectors(
    *,
    output_dir: Path,
    fork_name: str,
    preset_name: str,
    handlers: list[str],
    per_handler_limit: int,
    changed_only: bool,
    keep_existing: bool,
) -> None:
    dumper = Dumper()
    if changed_only:
        abstract_cases = enumerate_matching_abstract_cases(handlers=handlers)
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
        )
        if result is None:
            continue
        if changed_only and not has_changed_post_state(result):
            continue
        result.test_case.set_output_dir(str(output_dir))
        if result.test_case.dir.exists() and not keep_existing:
            shutil.rmtree(result.test_case.dir)
        dump_test_case_result(result, dumper)
        written_counts[abstract_case.handler_name] += 1
        if all(count >= per_handler_limit for count in written_counts.values()):
            return


def materialize_with_base_state(abstract_case, *, fork_name: str, preset_name: str):
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
                ),
            )
        except UnsupportedProfileError:
            return

    results = list(get_result(phase=fork_name, preset=preset_name, bls_active=False))
    if not results:
        return None
    ((result,),) = results
    return result


def has_changed_post_state(result) -> bool:
    parts = {name: data for name, kind, data in result.case_parts if kind == "ssz"}
    return parts["pre"] != parts["post"]


if __name__ == "__main__":
    main()
