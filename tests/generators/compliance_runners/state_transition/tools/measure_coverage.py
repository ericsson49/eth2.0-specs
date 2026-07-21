from __future__ import annotations

import argparse
import ast
import json as jsonlib
from pathlib import Path

import pytest
from coverage import Coverage
from ruamel.yaml import YAML

from .interaction_coverage import (
    add_stage_dimension,
    format_interaction_report,
    interaction_settings,
)
from ..ontology import (
    intent_outcomes_by_runner,
    load_test_ontology,
    stage_handlers,
    target_functions_by_runner,
)

RUNNER_TEST = Path("tests/generators/compliance_runners/state_transition/runner/test_run.py")
PYSPEC_ROOT = Path("tests/core/pyspec/eth_consensus_specs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run generated state-transition vectors and write coverage reports"
    )
    parser.add_argument(
        "--test-dir",
        action="append",
        type=Path,
        required=True,
        help=(
            "Directory containing generated state-transition compliance tests. "
            "Can be repeated for campaign-level coverage."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("coverage_reports/state_transition"),
        help="Directory for coverage data and reports.",
    )
    parser.add_argument(
        "--source-file",
        action="append",
        type=Path,
        help=(
            "Source file to include in focused reports. Defaults to pyspec files "
            "inferred from generated manifests. Can be repeated."
        ),
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        help=(
            "YAML mapping of runner -> handler -> target function names. "
            "Defaults to the state-transition test ontology. Deprecated in favor "
            "of --ontology."
        ),
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        help="YAML ontology declaring target functions, guide intents, and expected outcomes.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Do not write an HTML coverage report.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Do not write coverage.py annotated source files.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write a JSON coverage report.",
    )
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    exit_code = measure_coverage(
        test_dirs=args.test_dir,
        output_dir=args.output,
        source_files=args.source_file,
        target_config=args.target_config,
        ontology_path=args.ontology,
        html=not args.no_html,
        annotate=not args.no_annotate,
        json=not args.no_json,
        start=args.start,
        limit=args.limit,
    )
    raise SystemExit(exit_code)


def measure_coverage(
    *,
    test_dirs: list[Path],
    output_dir: Path,
    source_files: list[Path] | None,
    target_config: Path | None,
    ontology_path: Path | None,
    html: bool,
    annotate: bool,
    json: bool,
    start: int | None,
    limit: int | None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_files = resolve_report_files(test_dirs, source_files)
    ontology = load_test_ontology(ontology_path)
    target_functions = resolve_target_functions(test_dirs, target_config, ontology)
    intent_outcomes = infer_suite_intent_outcomes(test_dirs, ontology)
    stages = stage_handlers(ontology)
    interaction_dimensions, interaction_max_order = interaction_settings(ontology)

    cov = Coverage(
        branch=True,
        data_file=str(output_dir / ".coverage"),
    )
    cov.erase()
    cov.start()
    pytest_args = [
        str(RUNNER_TEST),
        "-q",
    ]
    for test_dir in test_dirs:
        pytest_args.extend(["--test-dir", str(test_dir)])
    if start is not None:
        pytest_args.extend(["--start", str(start)])
    if limit is not None:
        pytest_args.extend(["--limit", str(limit)])
    pytest_exit_code = pytest.main(pytest_args)
    cov.stop()
    cov.save()

    text_report_path = output_dir / "coverage.txt"
    with text_report_path.open("w") as report:
        cov.report(morfs=report_files, file=report, show_missing=False)
    cov.report(morfs=report_files, show_missing=False)

    missing_report_path = output_dir / "coverage_missing.txt"
    with missing_report_path.open("w") as report:
        cov.report(morfs=report_files, file=report, show_missing=True)

    json_report_path = output_dir / "coverage.json"
    internal_json_path = json_report_path if json else output_dir / ".coverage-summary.json"
    cov.json_report(
        morfs=report_files,
        outfile=str(internal_json_path),
        show_contexts=False,
    )
    summary_path = output_dir / "coverage_summary.txt"
    write_function_summary(internal_json_path, summary_path)
    target_summary_path = output_dir / "target_coverage.txt"
    write_target_summary(internal_json_path, target_summary_path, target_functions)
    semantic_summary_path = output_dir / "semantic_coverage.txt"
    write_semantic_summary(test_dirs, semantic_summary_path, intent_outcomes)
    interaction_summary_path = output_dir / "interaction_coverage.txt"
    write_interaction_summary(
        test_dirs,
        interaction_summary_path,
        stages,
        dimensions=interaction_dimensions,
        max_order=interaction_max_order,
    )
    if json:
        pass
    else:
        internal_json_path.unlink()
    if html:
        cov.html_report(
            morfs=report_files,
            directory=str(output_dir / "html"),
        )
    if annotate:
        cov.annotate(
            morfs=report_files,
            directory=str(output_dir / "annotated"),
        )

    print(f"Coverage data: {output_dir / '.coverage'}")
    print(f"Text report:   {text_report_path}")
    print(f"Full missing:  {missing_report_path}")
    print(f"Summary:       {summary_path}")
    print(f"Target report: {target_summary_path}")
    print(f"Semantic:      {semantic_summary_path}")
    print(f"Interactions:  {interaction_summary_path}")
    if json:
        print(f"JSON report:   {json_report_path}")
    if html:
        print(f"HTML report:   {output_dir / 'html' / 'index.html'}")
    if annotate:
        print(f"Annotations:   {output_dir / 'annotated'}")

    return int(pytest_exit_code)


def write_function_summary(json_report_path: Path, output_path: Path) -> None:
    report = jsonlib.loads(json_report_path.read_text())
    lines = []
    for filename, file_report in sorted(report["files"].items()):
        path = Path(filename)
        function_ranges = collect_function_ranges(path)
        executed_lines = set(file_report["executed_lines"])
        missing_lines = set(file_report["missing_lines"])
        statement_lines = executed_lines | missing_lines
        executed_branches = [tuple(branch) for branch in file_report.get("executed_branches", [])]
        missing_branches = [tuple(branch) for branch in file_report.get("missing_branches", [])]

        lines.append(f"{filename}")
        lines.append("-" * len(filename))
        lines.append(format_file_summary(file_report["summary"]))
        lines.append("")
        touched = 0
        omitted = 0
        for function_name, start, body_start, end in function_ranges:
            function_statements = {
                line for line in statement_lines if body_start <= line <= end
            }
            if not function_statements:
                continue
            executed = {line for line in executed_lines if body_start <= line <= end}
            missing = sorted(line for line in missing_lines if body_start <= line <= end)
            function_missing_branches = [
                branch for branch in missing_branches if body_start <= branch[0] <= end
            ]
            function_executed_branches = [
                branch for branch in executed_branches if body_start <= branch[0] <= end
            ]
            if not executed and not function_executed_branches:
                omitted += 1
                continue
            touched += 1
            lines.extend(
                format_function_summary(
                    function_name=function_name,
                    start=start,
                    end=end,
                    statement_count=len(function_statements),
                    executed_count=len(executed),
                    missing=missing,
                    executed_branches=function_executed_branches,
                    missing_branches=function_missing_branches,
                )
            )
        if touched == 0:
            lines.append("No touched functions in focused report.")
        if omitted:
            lines.append(f"Untouched functions omitted: {omitted}")
        lines.append("")

    output_path.write_text("\n".join(lines))


def write_target_summary(
    json_report_path: Path,
    output_path: Path,
    target_functions: dict[str, tuple[str, ...]],
) -> None:
    report = jsonlib.loads(json_report_path.read_text())
    lines = ["Target Coverage", "===============", ""]
    if not target_functions:
        lines.append("No target functions configured for this suite.")
        output_path.write_text("\n".join(lines))
        return

    totals = CoverageTotals()
    for filename, file_report in sorted(report["files"].items()):
        path = Path(filename)
        function_ranges = {
            function_name: (start, body_start, end)
            for function_name, start, body_start, end in collect_function_ranges(path)
        }
        executed_lines = set(file_report["executed_lines"])
        missing_lines = set(file_report["missing_lines"])
        statement_lines = executed_lines | missing_lines
        executed_branches = [tuple(branch) for branch in file_report.get("executed_branches", [])]
        missing_branches = [tuple(branch) for branch in file_report.get("missing_branches", [])]

        file_targets = [
            target_name for target_name in target_functions if target_name in function_ranges
        ]
        if not file_targets:
            continue

        lines.append(filename)
        lines.append("-" * len(filename))
        for target_name in file_targets:
            start, body_start, end = function_ranges[target_name]
            target = summarize_function_range(
                statement_lines=statement_lines,
                executed_lines=executed_lines,
                missing_lines=missing_lines,
                executed_branches=executed_branches,
                missing_branches=missing_branches,
                body_start=body_start,
                end=end,
            )
            totals.add(target)
            lines.extend(
                format_target_summary(
                    label=target_functions[target_name][0],
                    function_name=target_name,
                    start=start,
                    end=end,
                    target=target,
                )
            )
        lines.append("")

    lines.append("Total")
    lines.append("-----")
    lines.extend(format_totals(totals))
    output_path.write_text("\n".join(lines))


def write_semantic_summary(
    test_dirs: list[Path],
    output_path: Path,
    intent_outcomes: dict[str, dict[str, dict[str, str]]],
) -> None:
    manifests = load_case_metadata(test_dirs)
    lines = ["Semantic Coverage", "=================", ""]
    totals = SemanticTotals()
    if not intent_outcomes:
        lines.append("No semantic intents configured for this suite.")
        output_path.write_text("\n".join(lines))
        return

    for runner, handlers in sorted(intent_outcomes.items()):
        for handler, intents in sorted(handlers.items()):
            cases = [
                case
                for case in manifests
                if case["runner"] == runner and case["handler"] == handler
            ]
            if not cases:
                continue
            lines.append(f"{runner}/{handler}")
            lines.append("-" * len(f"{runner}/{handler}"))
            for intent_name, expected_outcome in sorted(intents.items()):
                matching_cases = [
                    case for case in cases if case["guide_intent"] == intent_name
                ]
                actual_outcomes = sorted({case["outcome"] for case in matching_cases})
                covered = bool(matching_cases)
                outcome_ok = covered and actual_outcomes == [expected_outcome]
                totals.add(covered=covered, outcome_ok=outcome_ok)
                status = "ok" if outcome_ok else "missing" if not covered else "mismatch"
                lines.append(
                    f"{intent_name}: {status} "
                    f"(expected {expected_outcome}, actual {actual_outcomes or ['none']})"
                )
            lines.append("")

    lines.append("Total")
    lines.append("-----")
    lines.append(f"intents:  {totals.covered_intents}/{totals.total_intents}")
    lines.append(f"outcomes: {totals.valid_outcomes}/{totals.total_intents}")
    output_path.write_text("\n".join(lines))


def write_interaction_summary(
    test_dirs: list[Path],
    output_path: Path,
    stages: dict[str, tuple[str, ...]],
    *,
    dimensions: tuple[str, ...],
    max_order: int,
) -> None:
    cases = add_stage_dimension(load_case_metadata(test_dirs), stages)
    output_path.write_text(
        format_interaction_report(
            cases,
            dimensions,
            max_order=max_order,
        )
    )


class SemanticTotals:
    def __init__(self) -> None:
        self.total_intents = 0
        self.covered_intents = 0
        self.valid_outcomes = 0

    def add(self, *, covered: bool, outcome_ok: bool) -> None:
        self.total_intents += 1
        if covered:
            self.covered_intents += 1
        if outcome_ok:
            self.valid_outcomes += 1


def load_case_metadata(test_dirs: Path | list[Path]) -> list[dict[str, object]]:
    yaml = YAML(typ="safe")
    cases = []
    if isinstance(test_dirs, Path):
        test_dirs = [test_dirs]
    elif isinstance(test_dirs, str):
        test_dirs = [Path(test_dirs)]

    for test_dir in test_dirs:
        cases.extend(load_case_metadata_from_dir(test_dir, yaml))
    return cases


def load_case_metadata_from_dir(test_dir: Path, yaml: YAML) -> list[dict[str, object]]:
    cases = []
    for manifest_path in test_dir.rglob("manifest.yaml"):
        manifest = yaml.load(manifest_path.read_text())
        meta = yaml.load((manifest_path.parent / "meta.yaml").read_text())
        cases.append(
            {
                "runner": manifest["runner"],
                "handler": manifest["handler"],
                "guide_intent": meta.get("profile", {}).get("guide_intent"),
                "profile": meta.get("profile", {}),
                "strategy_goal_id": meta.get("strategy_goal_id"),
                "strategy_goal_kind": meta.get("strategy_goal_kind"),
                "strategy_goal_labels": meta.get("strategy_goal_labels", []),
                "outcome": classify_case_outcome(meta),
            }
        )
    return cases


def classify_case_outcome(meta: dict[str, object]) -> str:
    if not meta.get("operation_valid", True):
        return "assertion_failure"
    if meta.get("post_state_changed"):
        return "changed"
    return "no_change"


class CoverageTotals:
    def __init__(self) -> None:
        self.statements = 0
        self.executed = 0
        self.branches = 0
        self.covered_branches = 0

    def add(self, target: dict[str, object]) -> None:
        self.statements += int(target["statement_count"])
        self.executed += int(target["executed_count"])
        self.branches += int(target["branch_count"])
        self.covered_branches += int(target["covered_branch_count"])


def summarize_function_range(
    *,
    statement_lines: set[int],
    executed_lines: set[int],
    missing_lines: set[int],
    executed_branches: list[tuple[int, int]],
    missing_branches: list[tuple[int, int]],
    body_start: int,
    end: int,
) -> dict[str, object]:
    function_statements = {line for line in statement_lines if body_start <= line <= end}
    executed = {line for line in executed_lines if body_start <= line <= end}
    missing = sorted(line for line in missing_lines if body_start <= line <= end)
    function_missing_branches = [
        branch for branch in missing_branches if body_start <= branch[0] <= end
    ]
    function_executed_branches = [
        branch for branch in executed_branches if body_start <= branch[0] <= end
    ]
    return {
        "statement_count": len(function_statements),
        "executed_count": len(executed),
        "missing": missing,
        "branch_count": len(function_executed_branches) + len(function_missing_branches),
        "covered_branch_count": len(function_executed_branches),
        "missing_branches": function_missing_branches,
    }


def format_target_summary(
    *,
    label: str,
    function_name: str,
    start: int,
    end: int,
    target: dict[str, object],
) -> list[str]:
    statement_count = int(target["statement_count"])
    executed_count = int(target["executed_count"])
    branch_count = int(target["branch_count"])
    covered_branch_count = int(target["covered_branch_count"])
    statement_percent = percent(executed_count, statement_count)
    branch_percent = percent(covered_branch_count, branch_count)
    lines = [
        f"{label}: {function_name} ({start}-{end})",
        f"  statements: {executed_count}/{statement_count} ({statement_percent:.1f}%)",
        f"  branches:   {covered_branch_count}/{branch_count} ({branch_percent:.1f}%)",
    ]
    missing = target["missing"]
    if missing:
        lines.append(f"  missing lines: {format_ranges(missing)}")
    missing_branches = target["missing_branches"]
    if missing_branches:
        lines.append(f"  missing branches: {format_branches(missing_branches)}")
    return lines


def format_totals(totals: CoverageTotals) -> list[str]:
    return [
        f"statements: {totals.executed}/{totals.statements} "
        f"({percent(totals.executed, totals.statements):.1f}%)",
        f"branches:   {totals.covered_branches}/{totals.branches} "
        f"({percent(totals.covered_branches, totals.branches):.1f}%)",
    ]


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 100.0
    return numerator / denominator * 100


def collect_function_ranges(path: Path) -> list[tuple[str, int, int, int]]:
    tree = ast.parse(path.read_text())
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body_start = node.body[0].lineno if node.body else node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            ranges.append((node.name, node.lineno, body_start, end))
    return sorted(ranges, key=lambda item: item[1])


def format_file_summary(summary: dict) -> str:
    return (
        f"statements: {summary['covered_lines']}/{summary['num_statements']} "
        f"({summary['percent_covered_display']}%), "
        f"branches: {summary['covered_branches']}/{summary['num_branches']}, "
        f"partial branches: {summary['num_partial_branches']}"
    )


def format_function_summary(
    *,
    function_name: str,
    start: int,
    end: int,
    statement_count: int,
    executed_count: int,
    missing: list[int],
    executed_branches: list[tuple[int, int]],
    missing_branches: list[tuple[int, int]],
) -> list[str]:
    percent = 100.0 if statement_count == 0 else executed_count / statement_count * 100
    lines = [
        f"{function_name} ({start}-{end})",
        f"  statements: {executed_count}/{statement_count} ({percent:.1f}%)",
    ]
    if missing:
        lines.append(f"  missing lines: {format_ranges(missing)}")
    if executed_branches or missing_branches:
        lines.append(
            f"  branches: {len(executed_branches)} covered, {len(missing_branches)} missing"
        )
    if missing_branches:
        lines.append(f"  missing branches: {format_branches(missing_branches)}")
    return lines


def format_ranges(numbers: list[int]) -> str:
    ranges = []
    start = numbers[0]
    previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(format_range(start, previous))
        start = number
        previous = number
    ranges.append(format_range(start, previous))
    return ", ".join(ranges)


def format_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"


def format_branches(branches: list[tuple[int, int]]) -> str:
    return ", ".join(f"{source}->{format_branch_target(target)}" for source, target in branches)


def format_branch_target(target: int) -> str:
    if target < 0:
        return "exit"
    return str(target)


def resolve_report_files(test_dirs: list[Path], source_files: list[Path] | None) -> list[str]:
    if source_files:
        return [str(path) for path in source_files]

    inferred_files = sorted(infer_pyspec_files(test_dirs))
    if inferred_files:
        return [str(path) for path in inferred_files]
    return [str(PYSPEC_ROOT)]


def resolve_target_functions(
    test_dirs: list[Path],
    target_config: Path | None,
    ontology: dict,
) -> dict[str, tuple[str, ...]]:
    target_config_data = load_target_config(target_config, ontology)
    suite_targets = infer_suite_targets(test_dirs, target_config_data)
    target_functions = {}
    for runner, handlers in suite_targets.items():
        for handler, functions in handlers.items():
            for function_name in functions:
                target_functions[function_name] = (f"{runner}/{handler}",)
    return target_functions


def load_target_config(target_config: Path | None, ontology: dict) -> dict:
    if target_config is None:
        return target_functions_by_runner(ontology)
    yaml = YAML(typ="safe")
    return yaml.load(target_config.read_text())


def infer_suite_targets(test_dirs: list[Path], target_config: dict) -> dict[str, dict[str, tuple[str, ...]]]:
    yaml = YAML(typ="safe")
    targets = {}
    for test_dir in test_dirs:
        for manifest_path in test_dir.rglob("manifest.yaml"):
            manifest = yaml.load(manifest_path.read_text())
            runner = manifest["runner"]
            handler = manifest["handler"]
            functions = target_config.get(runner, {}).get(handler)
            if not functions:
                continue
            if isinstance(functions, dict):
                functions = functions["functions"]
            targets.setdefault(runner, {})[handler] = tuple(functions)
    return targets


def infer_suite_intent_outcomes(
    test_dirs: list[Path],
    ontology: dict,
) -> dict[str, dict[str, dict[str, str]]]:
    configured_outcomes = intent_outcomes_by_runner(ontology)
    yaml = YAML(typ="safe")
    outcomes = {}
    for test_dir in test_dirs:
        for manifest_path in test_dir.rglob("manifest.yaml"):
            manifest = yaml.load(manifest_path.read_text())
            runner = manifest["runner"]
            handler = manifest["handler"]
            handler_outcomes = configured_outcomes.get(runner, {}).get(handler)
            if not handler_outcomes:
                continue
            outcomes.setdefault(runner, {})[handler] = handler_outcomes
    return outcomes


def infer_pyspec_files(test_dirs: list[Path]) -> set[Path]:
    yaml = YAML(typ="safe")
    files = set()
    for test_dir in test_dirs:
        for manifest_path in test_dir.rglob("manifest.yaml"):
            manifest = yaml.load(manifest_path.read_text())
            fork = manifest["fork"]
            preset = manifest["preset"]
            source_file = PYSPEC_ROOT / fork / f"{preset}.py"
            if source_file.exists():
                files.add(source_file)
    return files


if __name__ == "__main__":
    main()
