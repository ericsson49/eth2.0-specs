from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from .generate_vectors import generate_vectors
from .summarize_suite import summarize_suite

DEFAULT_SUITE_CONFIG_DIR = Path(
    "tests/generators/compliance_runners/state_transition/suite_configs"
)
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
    args = parser.parse_args()

    suite_config_path = resolve_suite_config_path(args.suite)
    suite_config = read_yaml(suite_config_path)
    generation_config = suite_config["generation"]
    output_dir = args.output or Path(generation_config["output"])

    if args.generate:
        generate_from_config(generation_config, output_dir)

    if args.validate and not args.coverage:
        validate_suite(output_dir)

    if args.coverage:
        coverage_config = suite_config.get("coverage", {})
        coverage_output = args.coverage_output or Path(coverage_config["output"])
        measure_from_config(coverage_config, test_dir=output_dir, output_dir=coverage_output)
    else:
        coverage_config = suite_config.get("coverage", {})
        coverage_output = args.coverage_output or Path(coverage_config.get("output", ""))

    if args.summary:
        ontology_path = coverage_config.get("ontology")
        summary = summarize_suite(
            test_dir=output_dir,
            ontology_path=Path(ontology_path) if ontology_path else None,
            coverage_dir=coverage_output if args.coverage or coverage_output.exists() else None,
        )
        print(summary)
        if args.summary_output is not None:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(summary)


def resolve_suite_config_path(suite: str) -> Path:
    suite_path = Path(suite)
    if suite_path.exists():
        return suite_path
    candidate = DEFAULT_SUITE_CONFIG_DIR / f"{suite}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unknown suite config: {suite}")


def read_yaml(path: Path):
    yaml = YAML(typ="safe")
    return yaml.load(path.read_text())


def generate_from_config(generation_config: dict, output_dir: Path) -> None:
    keep_existing = generation_config.get("keep_existing", False)
    if output_dir.exists() and not keep_existing:
        shutil.rmtree(output_dir)

    generate_vectors(
        output_dir=output_dir,
        fork_name=generation_config["fork"],
        preset_name=generation_config["preset"],
        handlers=generation_config["handlers"],
        per_handler_limit=generation_config["per_handler_limit"],
        changed_only=generation_config.get("changed_only", False),
        unchanged_only=generation_config.get("unchanged_only", False),
        invalid_only=generation_config.get("invalid_only", False),
        guided=generation_config.get("guided", False),
        keep_existing=keep_existing,
    )


def validate_suite(output_dir: Path) -> None:
    exit_code = pytest.main(
        [
            str(RUNNER_TEST),
            "--test-dir",
            str(output_dir),
            "-q",
        ]
    )
    if exit_code != 0:
        raise SystemExit(int(exit_code))


def measure_from_config(coverage_config: dict, *, test_dir: Path, output_dir: Path) -> None:
    args = [
        sys.executable,
        "-m",
        "tests.generators.compliance_runners.state_transition.measure_coverage",
        "--test-dir",
        str(test_dir),
        "--output",
        str(output_dir),
    ]
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
