from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .measure_coverage import load_case_metadata
from .ontology import (
    intent_outcomes_by_runner,
    load_test_ontology,
    stage_handlers,
    target_functions_by_runner,
)
from .suite_config import read_yaml, resolve_suite_config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a generated state-transition compliance suite"
    )
    parser.add_argument(
        "--test-dir",
        action="append",
        type=Path,
        required=True,
        help=(
            "Directory containing generated state-transition compliance tests. "
            "Can be repeated for campaign-level summaries."
        ),
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        help="YAML ontology declaring target functions, guide intents, and expected outcomes.",
    )
    parser.add_argument(
        "--suite",
        help="Optional suite config name or path. Used for distribution quota reporting.",
    )
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        help="Directory containing coverage reports written by measure_coverage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to write the summary to. The summary is always printed.",
    )
    args = parser.parse_args()

    suite_config = read_yaml(resolve_suite_config_path(args.suite)) if args.suite else None
    coverage_ontology = suite_config.get("coverage", {}).get("ontology") if suite_config else None
    ontology_path = args.ontology or (Path(coverage_ontology) if coverage_ontology else None)
    summary = summarize_suite(
        test_dir=args.test_dir,
        ontology_path=ontology_path,
        coverage_dir=args.coverage_dir,
        distribution=suite_config.get("generation", {}).get("distribution")
        if suite_config
        else None,
    )
    print(summary)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary)


def summarize_suite(
    *,
    test_dir: Path | list[Path],
    ontology_path: Path | None = None,
    coverage_dir: Path | None = None,
    distribution: dict[str, dict[str, int]] | None = None,
    title: str = "State Transition Suite Summary",
) -> str:
    ontology = load_test_ontology(ontology_path)
    cases = load_rich_case_metadata(test_dir)
    stages = stage_handlers(ontology)
    intent_outcomes = intent_outcomes_by_runner(ontology)
    target_functions = target_functions_by_runner(ontology)

    lines = [title, "=" * len(title), ""]
    lines.extend(format_suite_shape(cases))
    lines.append("")
    lines.extend(format_stage_summary(cases, stages, intent_outcomes, coverage_dir))
    lines.append("")
    lines.extend(format_outcome_counts(cases))
    lines.append("")
    lines.extend(format_distribution(cases, distribution))
    lines.append("")
    lines.extend(format_ontology_fit(cases, intent_outcomes, target_functions))
    lines.append("")
    lines.extend(format_semantic_outcomes(cases, intent_outcomes))
    lines.append("")
    lines.extend(format_target_coverage(coverage_dir))
    return "\n".join(lines)


def load_rich_case_metadata(test_dir: Path) -> list[dict[str, Any]]:
    return load_case_metadata(test_dir)


def format_suite_shape(cases: list[dict[str, Any]]) -> list[str]:
    lines = ["Suite Shape", "-----------"]
    if not cases:
        lines.append("No generated cases found.")
        return lines

    lines.append(f"total cases: {len(cases)}")
    for runner, runner_cases in sorted(group_by(cases, "runner").items()):
        lines.append(f"{runner}: {len(runner_cases)} cases")
        for handler, handler_cases in sorted(group_by(runner_cases, "handler").items()):
            intents = sorted(
                intent for intent in {case["guide_intent"] for case in handler_cases} if intent
            )
            intent_summary = f" [{', '.join(intents)}]" if intents else ""
            lines.append(f"  {handler}: {len(handler_cases)}{intent_summary}")
    return lines


def format_stage_summary(
    cases: list[dict[str, Any]],
    stages: dict[str, tuple[str, ...]],
    intent_outcomes: dict[str, dict[str, dict[str, str]]],
    coverage_dir: Path | None,
) -> list[str]:
    lines = ["Stage Summary", "-------------"]
    if not stages:
        lines.append("not configured")
        return lines

    target_totals = parse_stage_target_totals(coverage_dir, stages)
    for stage_name, handlers in stages.items():
        stage_cases = [
            case
            for case in cases
            if case["handler"] in handlers
        ]
        outcome_counts = Counter(case["outcome"] for case in stage_cases)
        intent_covered, intent_total, outcome_valid = stage_semantic_totals(
            stage_cases,
            handlers,
            intent_outcomes,
        )
        outcome_summary = format_counter(outcome_counts) if outcome_counts else "none"
        lines.append(f"{stage_name}: {len(stage_cases)} cases ({outcome_summary})")
        lines.append(f"  intents: {intent_covered}/{intent_total}")
        lines.append(f"  outcomes: {outcome_valid}/{intent_total}")
        if stage_name in target_totals:
            statement_done, statement_total, branch_done, branch_total = target_totals[stage_name]
            lines.append(
                f"  targets: statements {statement_done}/{statement_total} "
                f"({percent_string(statement_done, statement_total)}), "
                f"branches {branch_done}/{branch_total} "
                f"({percent_string(branch_done, branch_total)})"
            )
    ungrouped_cases = [
        case
        for case in cases
        if not any(case["handler"] in handlers for handlers in stages.values())
    ]
    if ungrouped_cases:
        lines.append(f"ungrouped: {len(ungrouped_cases)} cases")
    return lines


def stage_semantic_totals(
    cases: list[dict[str, Any]],
    handlers: tuple[str, ...],
    intent_outcomes: dict[str, dict[str, dict[str, str]]],
) -> tuple[int, int, int]:
    covered = 0
    total = 0
    valid = 0
    for runner, runner_handlers in intent_outcomes.items():
        for handler in handlers:
            if handler not in runner_handlers:
                continue
            for intent_name, expected_outcome in runner_handlers[handler].items():
                matching_cases = [
                    case
                    for case in cases
                    if case["runner"] == runner
                    and case["handler"] == handler
                    and case["guide_intent"] == intent_name
                ]
                actual_outcomes = sorted({case["outcome"] for case in matching_cases})
                total += 1
                if matching_cases:
                    covered += 1
                if matching_cases and actual_outcomes == [expected_outcome]:
                    valid += 1
    return covered, total, valid


def parse_stage_target_totals(
    coverage_dir: Path | None,
    stages: dict[str, tuple[str, ...]],
) -> dict[str, tuple[int, int, int, int]]:
    if coverage_dir is None:
        return {}
    target_report = coverage_dir / "target_coverage.txt"
    if not target_report.exists():
        return {}

    totals = {stage_name: [0, 0, 0, 0] for stage_name in stages}
    pattern = re.compile(
        r"^(operations|epoch_processing)/([^:]+): .*\n"
        r"  statements: (\d+)/(\d+) \([^)]+\)\n"
        r"  branches:\s+(\d+)/(\d+) \([^)]+\)",
        re.MULTILINE,
    )
    for _, handler, statement_done, statement_total, branch_done, branch_total in pattern.findall(
        target_report.read_text()
    ):
        for stage_name, handlers in stages.items():
            if handler not in handlers:
                continue
            totals[stage_name][0] += int(statement_done)
            totals[stage_name][1] += int(statement_total)
            totals[stage_name][2] += int(branch_done)
            totals[stage_name][3] += int(branch_total)
    return {
        stage_name: tuple(stage_totals)
        for stage_name, stage_totals in totals.items()
        if stage_totals[1] or stage_totals[3]
    }


def format_outcome_counts(cases: list[dict[str, Any]]) -> list[str]:
    lines = ["Outcome Counts", "--------------"]
    if not cases:
        lines.append("No generated cases found.")
        return lines

    for runner, runner_cases in sorted(group_by(cases, "runner").items()):
        lines.append(runner)
        for handler, handler_cases in sorted(group_by(runner_cases, "handler").items()):
            counts = Counter(case["outcome"] for case in handler_cases)
            lines.append(f"  {handler}: {format_counter(counts)}")
    return lines


def format_distribution(
    cases: list[dict[str, Any]],
    distribution: dict[str, dict[str, int]] | None,
) -> list[str]:
    lines = ["Distribution Quotas", "-------------------"]
    if not distribution:
        lines.append("not configured")
        return lines

    labels = {
        "outcomes": Counter(case["outcome"] for case in cases),
        "runners": Counter(case["runner"] for case in cases),
        "handlers": Counter(case["handler"] for case in cases),
    }
    for dimension, quotas in sorted(distribution.items()):
        lines.append(dimension)
        for name, requested in sorted(quotas.items()):
            actual = labels.get(dimension, Counter())[name]
            unmet = max(0, int(requested) - actual)
            status = "ok" if unmet == 0 else f"unmet {unmet}"
            lines.append(f"  {name}: {actual}/{requested} ({status})")
    return lines


def format_ontology_fit(
    cases: list[dict[str, Any]],
    intent_outcomes: dict[str, dict[str, dict[str, str]]],
    target_functions: dict[str, dict[str, tuple[str, ...]]],
) -> list[str]:
    lines = ["Ontology Fit", "------------"]
    generated_handlers = {
        (case["runner"], case["handler"])
        for case in cases
    }
    ontology_handlers = {
        (runner, handler)
        for runner, handlers in intent_outcomes.items()
        for handler in handlers
    }
    target_handlers = {
        (runner, handler)
        for runner, handlers in target_functions.items()
        for handler in handlers
    }

    missing_intent_handlers = sorted(generated_handlers - ontology_handlers)
    missing_target_handlers = sorted(generated_handlers - target_handlers)
    unused_intent_handlers = sorted(ontology_handlers - generated_handlers)

    lines.append(f"generated handlers: {len(generated_handlers)}")
    lines.append(f"ontology intent handlers: {len(ontology_handlers)}")
    lines.append(f"ontology target handlers: {len(target_handlers)}")
    lines.append(format_handler_list("generated without intent ontology", missing_intent_handlers))
    lines.append(format_handler_list("generated without target ontology", missing_target_handlers))
    lines.append(format_handler_list("ontology intents not generated", unused_intent_handlers))

    missing_intents = collect_missing_intents(cases, intent_outcomes)
    if missing_intents:
        lines.append("missing intents:")
        for runner, handler, intent_name in missing_intents:
            lines.append(f"  {runner}/{handler}: {intent_name}")
    else:
        lines.append("missing intents: none")
    return lines


def collect_missing_intents(
    cases: list[dict[str, Any]],
    intent_outcomes: dict[str, dict[str, dict[str, str]]],
) -> list[tuple[str, str, str]]:
    generated_intents = {
        (case["runner"], case["handler"], case["guide_intent"])
        for case in cases
        if case["guide_intent"] is not None
    }
    missing = []
    generated_handlers = {(case["runner"], case["handler"]) for case in cases}
    for runner, handlers in intent_outcomes.items():
        for handler, intents in handlers.items():
            if (runner, handler) not in generated_handlers:
                continue
            for intent_name in intents:
                if (runner, handler, intent_name) not in generated_intents:
                    missing.append((runner, handler, intent_name))
    return sorted(missing)


def format_semantic_outcomes(
    cases: list[dict[str, Any]],
    intent_outcomes: dict[str, dict[str, dict[str, str]]],
) -> list[str]:
    lines = ["Semantic Outcomes", "-----------------"]
    covered = 0
    valid = 0
    total = 0
    mismatches = []

    for runner, handlers in sorted(intent_outcomes.items()):
        for handler, intents in sorted(handlers.items()):
            handler_cases = [
                case for case in cases if case["runner"] == runner and case["handler"] == handler
            ]
            if not handler_cases:
                continue
            for intent_name, expected_outcome in sorted(intents.items()):
                matching_cases = [
                    case for case in handler_cases if case["guide_intent"] == intent_name
                ]
                actual_outcomes = sorted({case["outcome"] for case in matching_cases})
                outcome_ok = bool(matching_cases) and actual_outcomes == [expected_outcome]
                total += 1
                if matching_cases:
                    covered += 1
                if outcome_ok:
                    valid += 1
                else:
                    mismatches.append(
                        f"{runner}/{handler}/{intent_name}: expected {expected_outcome}, "
                        f"actual {actual_outcomes or ['none']}"
                    )

    lines.append(f"intents: {covered}/{total}")
    lines.append(f"outcomes: {valid}/{total}")
    if mismatches:
        lines.append("mismatches:")
        lines.extend(f"  {mismatch}" for mismatch in mismatches)
    else:
        lines.append("mismatches: none")
    return lines


def format_target_coverage(coverage_dir: Path | None) -> list[str]:
    lines = ["Target Coverage", "---------------"]
    if coverage_dir is None:
        lines.append("not available; pass --coverage-dir to include target totals")
        return lines

    target_report = coverage_dir / "target_coverage.txt"
    if not target_report.exists():
        lines.append(f"not available; missing {target_report}")
        return lines

    totals = parse_target_totals(target_report.read_text())
    if totals is None:
        lines.append(f"not available; could not parse {target_report}")
        return lines

    statement_done, statement_total, statement_percent, branch_done, branch_total, branch_percent = totals
    lines.append(f"statements: {statement_done}/{statement_total} ({statement_percent})")
    lines.append(f"branches:   {branch_done}/{branch_total} ({branch_percent})")
    return lines


def parse_target_totals(text: str) -> tuple[str, str, str, str, str, str] | None:
    statement_matches = re.findall(r"statements: (\d+)/(\d+) \(([^)]+)\)", text)
    branch_matches = re.findall(r"branches:\s+(\d+)/(\d+) \(([^)]+)\)", text)
    if not statement_matches or not branch_matches:
        return None
    return (*statement_matches[-1], *branch_matches[-1])


def format_counter(counter: Counter) -> str:
    return ", ".join(f"{name}: {counter[name]}" for name in sorted(counter))


def percent_string(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "100.0%"
    return f"{numerator / denominator * 100:.1f}%"


def group_by(cases: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for case in cases:
        grouped[case[key]].append(case)
    return grouped


def format_handler_list(label: str, handlers: list[tuple[str, str]]) -> str:
    if not handlers:
        return f"{label}: none"
    return f"{label}: {', '.join(f'{runner}/{handler}' for runner, handler in handlers)}"


if __name__ == "__main__":
    main()
