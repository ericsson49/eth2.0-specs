from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compare_strategy_funnel import ExpectedGoal, load_realized_goals, RealizedGoal
from .goal_ledger import load_expected_goals, write_goal_ledger
from .strategies import enumerate_input_profile_strategy_goals
from .strategy_formula import input_profile_formulas_from_generation_configs
from .suite_config import (
    read_yaml,
    resolve_campaign_config_path,
    resolve_suite_config_path,
)


@dataclass(frozen=True)
class GoalFunnel:
    symbolic: int
    completable: int
    materialized: int

    @property
    def missing(self) -> int:
        return self.completable - self.materialized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a lean state-transition generation and coverage report."
    )
    parser.add_argument(
        "--test-dir",
        action="append",
        type=Path,
        required=True,
        help="Generated test directory. Can be repeated.",
    )
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        help="Coverage directory containing target_coverage.txt.",
    )
    parser.add_argument(
        "--suite",
        help="Suite config name or path to derive expected goals.",
    )
    parser.add_argument(
        "--campaign",
        help="Campaign config name or path to derive expected goals.",
    )
    parser.add_argument(
        "--expected-goals",
        type=Path,
        help="JSON goal ledger emitted by preview_strategy --goals-output.",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=0,
        help="Show up to N missing completable goals per handler.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to write the report. The report is always printed.",
    )
    args = parser.parse_args()

    expected_goals = resolve_expected_goals(
        suite=args.suite,
        campaign=args.campaign,
        expected_goals_path=args.expected_goals,
    )
    report = format_lean_report(
        test_dirs=args.test_dir,
        coverage_dir=args.coverage_dir,
        expected_goals=expected_goals,
        show_missing=args.show_missing,
    )
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)


def resolve_expected_goals(
    *,
    suite: str | None,
    campaign: str | None,
    expected_goals_path: Path | None,
) -> list[ExpectedGoal]:
    selected_sources = sum(source is not None for source in (suite, campaign, expected_goals_path))
    if selected_sources > 1:
        raise ValueError("--suite, --campaign, and --expected-goals are mutually exclusive")
    if expected_goals_path is not None:
        return load_expected_goals(expected_goals_path)
    if suite is not None:
        suite_config = read_yaml(resolve_suite_config_path(suite))
        return expected_goals_from_generation_configs([suite_config["generation"]])
    if campaign is not None:
        campaign_config = read_yaml(resolve_campaign_config_path(campaign))
        generation_configs = []
        for suite_entry in campaign_config["suites"]:
            suite_name = suite_entry if isinstance(suite_entry, str) else suite_entry["suite"]
            suite_config = read_yaml(resolve_suite_config_path(suite_name))
            generation_configs.append(suite_config["generation"])
        return expected_goals_from_generation_configs(generation_configs)
    return []


def load_or_create_expected_goals(
    *,
    generation_configs: Iterable[dict[str, Any]],
    ledger_path: Path,
    refresh: bool,
) -> list[ExpectedGoal]:
    if ledger_path.exists() and not refresh:
        return load_expected_goals(ledger_path)
    expected_goals = expected_goals_from_generation_configs(generation_configs)
    write_goal_ledger(ledger_path, expected_goals)
    return expected_goals


def expected_goals_from_generation_configs(
    generation_configs: Iterable[dict[str, Any]],
) -> list[ExpectedGoal]:
    goals_by_id: dict[str, ExpectedGoal] = {}
    for formula in input_profile_formulas_from_generation_configs(generation_configs):
        for handler in formula.handlers:
            for goal in enumerate_input_profile_strategy_goals(
                handler,
                order=formula.order,
                include_lower_orders=formula.include_lower_orders,
            ):
                existing = goals_by_id.get(goal.goal_id)
                expected_goal = ExpectedGoal(
                    goal_id=goal.goal_id,
                    handler=goal.handler,
                    kind=goal.kind,
                    labels=goal.labels,
                    symbolic=goal.symbolic,
                    completable=goal.completable,
                )
                if existing is None:
                    goals_by_id[goal.goal_id] = expected_goal
                    continue
                goals_by_id[goal.goal_id] = ExpectedGoal(
                    goal_id=existing.goal_id,
                    handler=existing.handler,
                    kind=existing.kind,
                    labels=existing.labels,
                    symbolic=existing.symbolic or expected_goal.symbolic,
                    completable=existing.completable or expected_goal.completable,
                )
    return list(goals_by_id.values())


def format_lean_report(
    *,
    test_dirs: list[Path],
    coverage_dir: Path | None,
    expected_goals: list[ExpectedGoal],
    show_missing: int = 0,
    title: str = "State Transition Lean Report",
) -> str:
    realized_goals = load_realized_goals(test_dirs)
    lines = [title, "=" * len(title), ""]
    lines.extend(format_code_coverage(coverage_dir))
    lines.append("")
    lines.extend(format_goal_funnel(expected_goals, realized_goals))
    if show_missing:
        lines.append("")
        lines.extend(format_missing_goals(expected_goals, realized_goals, limit=show_missing))
    lines.append("")
    lines.extend(format_realized_goal_kinds(realized_goals))
    return "\n".join(lines)


def format_code_coverage(coverage_dir: Path | None) -> list[str]:
    lines = ["Code Coverage", "-------------"]
    if coverage_dir is None:
        lines.append("not available")
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
    lines.append(f"target statements: {statement_done}/{statement_total} ({statement_percent})")
    lines.append(f"target branches:   {branch_done}/{branch_total} ({branch_percent})")
    return lines


def parse_target_totals(text: str) -> tuple[str, str, str, str, str, str] | None:
    statement_matches = re.findall(r"statements: (\d+)/(\d+) \(([^)]+)\)", text)
    branch_matches = re.findall(r"branches:\s+(\d+)/(\d+) \(([^)]+)\)", text)
    if not statement_matches or not branch_matches:
        return None
    return (*statement_matches[-1], *branch_matches[-1])


def format_goal_funnel(
    expected_goals: list[ExpectedGoal],
    realized_goals: list[RealizedGoal],
) -> list[str]:
    lines = ["Goal Funnel", "-----------"]
    if not expected_goals:
        lines.append("not configured")
        return lines

    funnel = compute_goal_funnel(expected_goals, realized_goals)
    lines.append(f"symbolic goals:     {funnel.symbolic}")
    lines.append(f"completable goals:  {funnel.completable}")
    lines.append(f"materialized goals: {funnel.materialized}")
    lines.append(f"missing goals:      {funnel.missing}")
    lines.append("")
    lines.append("| handler | symbolic | completable | materialized | missing |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for handler, handler_funnel in sorted(
        compute_goal_funnel_by_handler(expected_goals, realized_goals).items()
    ):
        lines.append(
            f"| {handler} | {handler_funnel.symbolic} | "
            f"{handler_funnel.completable} | {handler_funnel.materialized} | "
            f"{handler_funnel.missing} |"
        )
    return lines


def compute_goal_funnel(
    expected_goals: list[ExpectedGoal],
    realized_goals: list[RealizedGoal],
) -> GoalFunnel:
    realized_ids = {goal.goal_id for goal in realized_goals}
    completable_goals = [goal for goal in expected_goals if goal.completable]
    return GoalFunnel(
        symbolic=sum(1 for goal in expected_goals if goal.symbolic),
        completable=len(completable_goals),
        materialized=sum(1 for goal in completable_goals if goal.goal_id in realized_ids),
    )


def compute_goal_funnel_by_handler(
    expected_goals: list[ExpectedGoal],
    realized_goals: list[RealizedGoal],
) -> dict[str, GoalFunnel]:
    realized_ids_by_handler: dict[str, set[str]] = defaultdict(set)
    for goal in realized_goals:
        realized_ids_by_handler[goal.handler].add(goal.goal_id)

    expected_by_handler: dict[str, list[ExpectedGoal]] = defaultdict(list)
    for goal in expected_goals:
        expected_by_handler[goal.handler].append(goal)

    return {
        handler: compute_goal_funnel(
            handler_goals,
            [
                RealizedGoal(goal_id=goal_id, handler=handler, case_path=Path())
                for goal_id in realized_ids_by_handler.get(handler, set())
            ],
        )
        for handler, handler_goals in expected_by_handler.items()
    }


def format_missing_goals(
    expected_goals: list[ExpectedGoal],
    realized_goals: list[RealizedGoal],
    *,
    limit: int,
) -> list[str]:
    lines = ["Missing Completable Goals", "-------------------------"]
    realized_ids_by_handler: dict[str, set[str]] = defaultdict(set)
    for goal in realized_goals:
        realized_ids_by_handler[goal.handler].add(goal.goal_id)

    missing_by_handler: dict[str, list[ExpectedGoal]] = defaultdict(list)
    for goal in expected_goals:
        if not goal.completable:
            continue
        if goal.goal_id in realized_ids_by_handler.get(goal.handler, set()):
            continue
        missing_by_handler[goal.handler].append(goal)

    if not missing_by_handler:
        lines.append("none")
        return lines

    for handler, goals in sorted(missing_by_handler.items()):
        lines.append(handler)
        for goal in goals[:limit]:
            lines.append(f"  {goal.goal_id}: {', '.join(goal.labels)}")
    return lines


def format_realized_goal_kinds(realized_goals: list[RealizedGoal]) -> list[str]:
    lines = ["Materialized Goal Kinds", "-----------------------"]
    if not realized_goals:
        lines.append("not configured")
        return lines
    handler_counts = Counter(goal.handler for goal in realized_goals)
    kind_counts = Counter(goal.kind or "unknown" for goal in realized_goals)
    labels = Counter(label for goal in realized_goals for label in goal.labels)
    lines.append(f"goal-backed cases: {len(realized_goals)}")
    lines.append(f"kinds: {format_counter(kind_counts)}")
    if labels:
        lines.append("top labels:")
        for label, count in sorted(labels.items(), key=lambda item: (-item[1], item[0]))[:10]:
            lines.append(f"  {label}: {count}")
    lines.append("handlers:")
    for handler, count in sorted(handler_counts.items()):
        lines.append(f"  {handler}: {count}")
    return lines


def format_counter(counter: Counter) -> str:
    return ", ".join(f"{name}: {counter[name]}" for name in sorted(counter))


if __name__ == "__main__":
    main()
