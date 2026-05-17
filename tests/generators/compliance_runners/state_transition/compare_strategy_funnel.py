from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml import YAML


@dataclass(frozen=True)
class ExpectedGoal:
    goal_id: str
    handler: str
    kind: str
    labels: tuple[str, ...]
    symbolic: bool
    completable: bool


@dataclass(frozen=True)
class RealizedGoal:
    goal_id: str
    handler: str
    case_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dry-run strategy goals against generated vectors."
    )
    parser.add_argument(
        "--expected-goals",
        type=Path,
        required=True,
        help="JSON file emitted by preview_strategy --goals-output.",
    )
    parser.add_argument(
        "--test-dir",
        action="append",
        type=Path,
        required=True,
        help="Generated test directory. Can be repeated.",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=0,
        help="Show up to N completable-but-unmaterialized goals per handler.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_goals = load_expected_goals(args.expected_goals)
    realized_goals = load_realized_goals(args.test_dir)
    print_funnel_report(
        expected_goals,
        realized_goals,
        show_missing=args.show_missing,
    )


def load_expected_goals(path: Path) -> list[ExpectedGoal]:
    data = json.loads(path.read_text())
    return [
        ExpectedGoal(
            goal_id=goal["goal_id"],
            handler=goal["handler"],
            kind=goal["kind"],
            labels=tuple(goal["labels"]),
            symbolic=bool(goal["symbolic"]),
            completable=bool(goal["completable"]),
        )
        for goal in data["goals"]
    ]


def load_realized_goals(test_dirs: Iterable[Path]) -> list[RealizedGoal]:
    yaml = YAML(typ="safe")
    realized = []
    for test_dir in test_dirs:
        for manifest_path in test_dir.rglob("manifest.yaml"):
            meta_path = manifest_path.parent / "meta.yaml"
            if not meta_path.exists():
                continue
            manifest = yaml.load(manifest_path.read_text())
            meta = yaml.load(meta_path.read_text())
            goal_id = meta.get("strategy_goal_id") or meta.get("profile", {}).get(
                "strategy_goal_id"
            )
            if goal_id is None:
                continue
            realized.append(
                RealizedGoal(
                    goal_id=goal_id,
                    handler=manifest["handler"],
                    case_path=manifest_path.parent,
                )
            )
    return realized


def print_funnel_report(
    expected_goals: list[ExpectedGoal],
    realized_goals: list[RealizedGoal],
    *,
    show_missing: int,
) -> None:
    expected_by_handler = group_expected_goals(expected_goals)
    realized_ids_by_handler = group_realized_goal_ids(realized_goals)

    print("| handler | symbolic | completable | materialized | missing |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for handler in sorted(expected_by_handler):
        handler_goals = expected_by_handler[handler]
        symbolic = sum(1 for goal in handler_goals if goal.symbolic)
        completable_goals = [goal for goal in handler_goals if goal.completable]
        materialized_ids = realized_ids_by_handler.get(handler, set())
        materialized = sum(1 for goal in completable_goals if goal.goal_id in materialized_ids)
        missing = len(completable_goals) - materialized
        print(
            f"| {handler} | {symbolic} | {len(completable_goals)} | "
            f"{materialized} | {missing} |"
        )
        if show_missing:
            print_missing_examples(
                completable_goals,
                materialized_ids,
                limit=show_missing,
            )


def group_expected_goals(goals: Iterable[ExpectedGoal]) -> dict[str, list[ExpectedGoal]]:
    grouped: dict[str, list[ExpectedGoal]] = defaultdict(list)
    for goal in goals:
        grouped[goal.handler].append(goal)
    return dict(grouped)


def group_realized_goal_ids(goals: Iterable[RealizedGoal]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for goal in goals:
        grouped[goal.handler].add(goal.goal_id)
    return dict(grouped)


def print_missing_examples(
    completable_goals: list[ExpectedGoal],
    materialized_ids: set[str],
    *,
    limit: int,
) -> None:
    shown = 0
    for goal in completable_goals:
        if goal.goal_id in materialized_ids:
            continue
        print(f"  - {goal.goal_id}: {format_labels(goal.labels)}")
        shown += 1
        if shown >= limit:
            return


def format_labels(labels: tuple[str, ...]) -> str:
    return ", ".join(labels)


if __name__ == "__main__":
    main()
