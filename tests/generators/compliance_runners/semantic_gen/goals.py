from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyGoal:
    """A stable semantic goal that can be tracked across generator stages."""

    goal_id: str
    handler: str
    kind: str
    labels: tuple[str, ...]
    symbolic: bool = True
    completable: bool = False
    metadata: dict[str, Any] | None = None

    def to_json_data(self) -> dict[str, Any]:
        data = asdict(self)
        data["labels"] = list(self.labels)
        return data


def make_goal_id(
    *,
    handler: str,
    kind: str,
    labels: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "handler": handler,
            "kind": kind,
            "labels": list(labels),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_strategy_goal(
    *,
    handler: str,
    kind: str,
    labels: tuple[str, ...],
    symbolic: bool = True,
    completable: bool = False,
    metadata: dict[str, Any] | None = None,
) -> StrategyGoal:
    return StrategyGoal(
        goal_id=make_goal_id(handler=handler, kind=kind, labels=labels),
        handler=handler,
        kind=kind,
        labels=labels,
        symbolic=symbolic,
        completable=completable,
        metadata=metadata,
    )
