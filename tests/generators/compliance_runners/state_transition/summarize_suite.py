from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .measure_coverage import load_case_metadata
from .ontology import intent_outcomes_by_runner, load_test_ontology, target_functions_by_runner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a generated state-transition compliance suite"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing generated state-transition compliance tests.",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        help="YAML ontology declaring target functions, guide intents, and expected outcomes.",
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

    summary = summarize_suite(
        test_dir=args.test_dir,
        ontology_path=args.ontology,
        coverage_dir=args.coverage_dir,
    )
    print(summary)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary)


def summarize_suite(
    *,
    test_dir: Path,
    ontology_path: Path | None = None,
    coverage_dir: Path | None = None,
) -> str:
    ontology = load_test_ontology(ontology_path)
    cases = load_rich_case_metadata(test_dir)
    intent_outcomes = intent_outcomes_by_runner(ontology)
    target_functions = target_functions_by_runner(ontology)

    lines = ["State Transition Suite Summary", "==============================", ""]
    lines.extend(format_suite_shape(cases))
    lines.append("")
    lines.extend(format_outcome_counts(cases))
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


def format_outcome_counts(cases: list[dict[str, Any]]) -> list[str]:
    lines = ["Outcome Counts", "--------------"]
    if not cases:
        lines.append("No generated cases found.")
        return lines

    for runner, runner_cases in sorted(group_by(cases, "runner").items()):
        lines.append(runner)
        for handler, handler_cases in sorted(group_by(runner_cases, "handler").items()):
            counts = Counter(case["outcome"] for case in handler_cases)
            summary = ", ".join(f"{name}: {counts[name]}" for name in sorted(counts))
            lines.append(f"  {handler}: {summary}")
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
