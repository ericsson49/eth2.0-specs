from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .check_reproducible import check_suite_reproducible
from .generate_vectors import generate_vectors, normalize_handlers
from .suite_config import (
    default_suite_coverage_dir,
    default_suite_output_dir,
    read_yaml,
    resolve_suite_config_path,
)
from .summarize_suite import summarize_suite

RUNNER_TEST = Path("tests/generators/compliance_runners/state_transition/runner/test_run.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reproducible state-transition test-suite profile"
    )
    parser.add_argument(
        "--suite",
        default="electra_operations_guided",
        help="Suite config name or path. Defaults to electra_operations_guided.",
    )
    parser.add_argument(
        "--generate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate vectors from the suite profile.",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate generated vectors with the local runner.",
    )
    parser.add_argument(
        "--coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure coverage for generated vectors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override generation output directory.",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        help="Override coverage output directory.",
    )
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print a suite health summary after generation, validation, or coverage.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional file to write the suite health summary to.",
    )
    parser.add_argument(
        "--check-reproducible",
        action="store_true",
        help="Generate the suite twice in temporary directories and compare outputs.",
    )
    parser.add_argument(
        "--keep-reproducibility-temp",
        action="store_true",
        help="Keep reproducibility comparison directories for inspection.",
    )
    args = parser.parse_args()

    suite_config_path = resolve_suite_config_path(args.suite)
    suite_config = read_yaml(suite_config_path)
    generation_config = suite_config["generation"]
    output_dir = args.output or default_suite_output_dir(suite_config, suite_config_path)

    if args.generate:
        generate_from_config(generation_config, output_dir)

    if args.validate and not args.coverage:
        validate_suite(output_dir)

    if args.coverage:
        coverage_config = suite_config.get("coverage", {})
        coverage_output = args.coverage_output or default_suite_coverage_dir(
            suite_config,
            suite_config_path,
        )
        measure_from_config(coverage_config, test_dir=output_dir, output_dir=coverage_output)
    else:
        coverage_config = suite_config.get("coverage", {})
        coverage_output = args.coverage_output or default_suite_coverage_dir(
            suite_config,
            suite_config_path,
        )

    if args.summary:
        ontology_path = coverage_config.get("ontology")
        summary = summarize_suite(
            test_dir=output_dir,
            ontology_path=Path(ontology_path) if ontology_path else None,
            coverage_dir=coverage_output if args.coverage or coverage_output.exists() else None,
            distribution=generation_config.get("distribution"),
            profile_dimensions=generation_config.get("profile_dimensions"),
            profile_interaction_order=generation_config.get("profile_interaction_order"),
        )
        print(summary)
        if args.summary_output is not None:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(summary)

    if args.check_reproducible:
        result = check_suite_reproducible(
            str(suite_config_path),
            keep_temp=args.keep_reproducibility_temp,
        )
        print(result.format())
        if not result.reproducible:
            raise SystemExit(1)


def generate_from_config(generation_config: dict, output_dir: Path) -> None:
    keep_existing = generation_config.get("keep_existing", False)
    if output_dir.exists() and not keep_existing:
        shutil.rmtree(output_dir)

    generate_vectors(
        output_dir=output_dir,
        fork_name=generation_config["fork"],
        preset_name=generation_config["preset"],
        handlers=normalize_handlers(
            generation_config.get("handlers"),
            stages=generation_config.get("stages"),
        ),
        per_handler_limit=generation_config["per_handler_limit"],
        changed_only=generation_config.get("changed_only", False),
        unchanged_only=generation_config.get("unchanged_only", False),
        invalid_only=generation_config.get("invalid_only", False),
        guided=generation_config.get("guided", False),
        mode=generation_config.get("mode"),
        profile_dimensions=generation_config.get("profile_dimensions"),
        profile_interaction_order=generation_config.get("profile_interaction_order", 2),
        profile_interaction_selection=generation_config.get(
            "profile_interaction_selection",
            "enumeration",
        ),
        input_profile_order=generation_config.get("input_profile_order", 1),
        input_profile_selection=generation_config.get(
            "input_profile_selection",
            "enumeration",
        ),
        keep_existing=keep_existing,
        distribution=generation_config.get("distribution"),
    )


def validate_suite(output_dir: Path) -> None:
    validate_suites([output_dir])


def validate_suites(output_dirs: list[Path]) -> None:
    pytest_args = [
        str(RUNNER_TEST),
        "-q",
    ]
    for output_dir in output_dirs:
        pytest_args.extend(["--test-dir", str(output_dir)])
    exit_code = pytest.main(
        pytest_args
    )
    if exit_code != 0:
        raise SystemExit(int(exit_code))


def measure_from_config(
    coverage_config: dict,
    *,
    test_dir: Path | list[Path],
    output_dir: Path,
) -> None:
    test_dirs = test_dir if isinstance(test_dir, list) else [test_dir]
    args = [
        sys.executable,
        "-m",
        "tests.generators.compliance_runners.state_transition.measure_coverage",
        "--output",
        str(output_dir),
    ]
    for path in test_dirs:
        args.extend(["--test-dir", str(path)])
    for source_file in coverage_config.get("source_files", []):
        args.extend(["--source-file", str(source_file)])
    if coverage_config.get("target_config"):
        args.extend(["--target-config", str(coverage_config["target_config"])])
    if coverage_config.get("ontology"):
        args.extend(["--ontology", str(coverage_config["ontology"])])
    if not coverage_config.get("html", True):
        args.append("--no-html")
    if not coverage_config.get("annotate", True):
        args.append("--no-annotate")
    if not coverage_config.get("json", True):
        args.append("--no-json")

    exit_code = subprocess.run(args, check=False).returncode
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
