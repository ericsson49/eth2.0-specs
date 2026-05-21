from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GOAL_LEDGER_FILENAME = "strategy_goals.json"


@dataclass(frozen=True)
class ExpectedGoal:
    goal_id: str
    handler: str
    kind: str
    labels: tuple[str, ...]
    symbolic: bool
    completable: bool


def goal_ledger_path(output_dir: Path) -> Path:
    return output_dir / GOAL_LEDGER_FILENAME


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


def write_goal_ledger(
    path: Path,
    expected_goals: Sequence[ExpectedGoal],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "strategy": "input_profile",
        "goals": [expected_goal_to_json_data(goal) for goal in expected_goals],
    }
    if metadata is not None:
        data.update(metadata)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def expected_goal_to_json_data(goal: ExpectedGoal) -> dict[str, Any]:
    return {
        "goal_id": goal.goal_id,
        "handler": goal.handler,
        "kind": goal.kind,
        "labels": list(goal.labels),
        "symbolic": goal.symbolic,
        "completable": goal.completable,
    }
