from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from eth_consensus_specs.test.context import spec_state_test, with_electra_and_later
from eth_consensus_specs.test.helpers.constants import ELECTRA, MINIMAL
from tests.generators.compliance_runners.gen_base.output import dump_test_case_result
from tests.generators.compliance_runners.gen_base.pytest_support import configure_generator_context
from tests.infra.dumper import Dumper

from .abstract_cases import (
    enumerate_guided_operation_cases,
    enumerate_input_profile_cases,
    enumerate_materializable_operation_cases,
    enumerate_profile_interaction_cases,
    enumerate_profile_partition_cases,
    HANDLER_NAMES,
    select_abstract_cases,
)
from .materializers import materialize_case, MATERIALIZED_HANDLER_NAMES, UnsupportedProfileError
from .ontology import stage_handlers

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
        "--mode",
        choices=(
            "simple",
            "handler_touch",
            "profile_partition",
            "profile_interaction",
            "input_profile",
            "guided",
        ),
        help=(
            "Generation strategy. Defaults to 'guided' when --guided is set, "
            "otherwise 'simple'."
        ),
    )
    parser.add_argument(
        "--profile-dimension",
        action="append",
        help=(
            "Profile dimension to cover in profile_partition mode. Can be repeated. "
            "Defaults to the built-in validator state dimensions."
        ),
    )
    parser.add_argument(
        "--profile-interaction-order",
        type=int,
        default=2,
        help="Profile interaction order in profile_interaction mode. Defaults to pairwise.",
    )
    parser.add_argument(
        "--profile-interaction-selection",
        choices=("enumeration", "prioritized"),
        default="enumeration",
        help=(
            "Profile interaction candidate selection in profile_interaction mode. "
            "Defaults to deterministic enumeration."
        ),
    )
    parser.add_argument(
        "--input-profile-order",
        type=int,
        default=1,
        help="Input profile interaction order in input_profile mode. Defaults to one-wise.",
    )
    parser.add_argument(
        "--input-profile-selection",
        choices=("enumeration", "prioritized"),
        default="enumeration",
        help=(
            "Input profile candidate selection in input_profile mode. "
            "Defaults to deterministic enumeration."
        ),
    )
    parser.add_argument(
        "--handler",
        action="append",
        help=(
            "Handler to generate. Defaults to withdrawal_request. Can be repeated. "
            "Use 'all' for every materialized state-transition handler."
        ),
    )
    parser.add_argument(
        "--stage",
        action="append",
        help="Named handler stage from the ontology, e.g. validator_lifecycle. Can be repeated.",
    )
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    generate_vectors(
        output_dir=args.output,
        fork_name=args.fork,
        preset_name=args.preset,
        handlers=normalize_handlers(args.handler, stages=args.stage),
        per_handler_limit=args.per_handler_limit,
        changed_only=args.changed_only,
        unchanged_only=args.unchanged_only,
        invalid_only=args.invalid_only,
        guided=args.guided,
        mode=args.mode,
        profile_dimensions=args.profile_dimension,
        profile_interaction_order=args.profile_interaction_order,
        profile_interaction_selection=args.profile_interaction_selection,
        input_profile_order=args.input_profile_order,
        input_profile_selection=args.input_profile_selection,
        keep_existing=args.keep_existing,
    )


def normalize_handlers(
    handlers: list[str] | None,
    *,
    stages: list[str] | None = None,
) -> list[str]:
    requested_handlers = list(handlers or [])
    stage_map = stage_handlers()
    for stage in stages or []:
        if stage not in stage_map:
            raise ValueError(f"Unknown stage: {stage}")
        requested_handlers.extend(stage_map[stage])
    if not requested_handlers:
        requested_handlers = ["withdrawal_request"]
    if "all" in requested_handlers:
        requested_handlers = list(MATERIALIZED_HANDLER_NAMES) + [
            handler for handler in requested_handlers if handler != "all"
        ]
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")
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
    mode: str | None = None,
    profile_dimensions: list[str] | None = None,
    profile_interaction_order: int = 2,
    profile_interaction_selection: str = "enumeration",
    input_profile_order: int = 1,
    input_profile_selection: str = "enumeration",
    keep_existing: bool,
    distribution: dict[str, dict[str, int]] | None = None,
) -> None:
    selected_filters = sum([changed_only, unchanged_only, invalid_only])
    if selected_filters > 1:
        raise ValueError("--changed-only, --unchanged-only, and --invalid-only are exclusive")

    dumper = Dumper()
    distribution_tracker = DistributionTracker.from_config(distribution)
    generation_mode = normalize_generation_mode(mode, guided)
    if generation_mode == "guided":
        abstract_cases = enumerate_guided_operation_cases(handlers=handlers)
    elif generation_mode == "handler_touch":
        abstract_cases = enumerate_materializable_operation_cases(handlers=handlers)
    elif generation_mode == "profile_partition":
        abstract_cases = enumerate_profile_partition_cases(
            handlers=handlers,
            dimensions=profile_dimensions,
        )
    elif generation_mode == "profile_interaction":
        abstract_cases = enumerate_profile_interaction_cases(
            handlers=handlers,
            dimensions=profile_dimensions,
            order=profile_interaction_order,
            selection=profile_interaction_selection,
            per_handler_limit=per_handler_limit,
        )
    elif generation_mode == "input_profile":
        abstract_cases = enumerate_input_profile_cases(
            handlers=handlers,
            order=input_profile_order,
            selection=input_profile_selection,
            per_handler_limit=per_handler_limit,
        )
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
        if not distribution_tracker.accepts(result):
            continue
        result.test_case.set_output_dir(str(output_dir))
        if result.test_case.dir.exists() and not keep_existing:
            shutil.rmtree(result.test_case.dir)
        dump_test_case_result(result, dumper)
        distribution_tracker.record(result)
        written_counts[abstract_case.handler_name] += 1
        if distribution_tracker.satisfied:
            return
        if all(count >= per_handler_limit for count in written_counts.values()):
            return


def normalize_generation_mode(mode: str | None, guided: bool) -> str:
    if mode is not None:
        if mode not in (
            "simple",
            "handler_touch",
            "profile_partition",
            "profile_interaction",
            "input_profile",
            "guided",
        ):
            raise ValueError(f"Unsupported generation mode: {mode}")
        return mode
    if guided:
        return "guided"
    return "simple"


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


def classify_result_outcome(result) -> str:
    if not is_operation_valid(result):
        return "assertion_failure"
    if is_changed_post_state(result):
        return "changed"
    return "no_change"


@dataclass
class DistributionTracker:
    quotas: dict[str, dict[str, int]] = field(default_factory=dict)
    counts: dict[str, Counter] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, dict[str, int]] | None) -> DistributionTracker:
        if not config:
            return cls()
        quotas = {
            dimension: {str(name): int(limit) for name, limit in values.items()}
            for dimension, values in config.items()
        }
        validate_distribution_quotas(quotas)
        return cls(
            quotas=quotas,
            counts={dimension: Counter() for dimension in quotas},
        )

    @property
    def enabled(self) -> bool:
        return bool(self.quotas)

    @property
    def satisfied(self) -> bool:
        if not self.enabled:
            return False
        return all(
            self.counts[dimension][name] >= limit
            for dimension, dimension_quotas in self.quotas.items()
            for name, limit in dimension_quotas.items()
        )

    def accepts(self, result) -> bool:
        if not self.enabled:
            return True
        labels = distribution_labels(result)
        for dimension, dimension_quotas in self.quotas.items():
            label = labels[dimension]
            if label not in dimension_quotas:
                return False
            if self.counts[dimension][label] >= dimension_quotas[label]:
                return False
        return True

    def record(self, result) -> None:
        if not self.enabled:
            return
        labels = distribution_labels(result)
        for dimension in self.quotas:
            self.counts[dimension][labels[dimension]] += 1


def validate_distribution_quotas(quotas: dict[str, dict[str, int]]) -> None:
    supported_dimensions = {"outcomes", "runners", "handlers"}
    unknown_dimensions = set(quotas) - supported_dimensions
    if unknown_dimensions:
        raise ValueError(f"Unsupported distribution dimensions: {sorted(unknown_dimensions)}")

    for dimension, dimension_quotas in quotas.items():
        if not dimension_quotas:
            raise ValueError(f"Distribution dimension {dimension!r} must not be empty")
        negative = {
            name: limit
            for name, limit in dimension_quotas.items()
            if limit < 0
        }
        if negative:
            raise ValueError(f"Distribution quotas must be non-negative: {negative}")


def distribution_labels(result) -> dict[str, str]:
    return {
        "outcomes": classify_result_outcome(result),
        "runners": result.test_case.runner_name,
        "handlers": result.test_case.handler_name,
    }


if __name__ == "__main__":
    main()
