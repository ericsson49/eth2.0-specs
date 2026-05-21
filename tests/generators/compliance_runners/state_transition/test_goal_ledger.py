from __future__ import annotations

from . import lean_report
from .goal_ledger import ExpectedGoal, load_expected_goals, write_goal_ledger


def test_goal_ledger_round_trip(tmp_path):
    path = tmp_path / "strategy_goals.json"
    expected = [
        ExpectedGoal(
            goal_id="goal-1",
            handler="withdrawal_request",
            kind="input_profile",
            labels=("request_kind:full_exit",),
            symbolic=True,
            completable=True,
        )
    ]

    write_goal_ledger(path, expected, metadata={"formulas": [{"name": "scratch"}]})

    assert load_expected_goals(path) == expected


def test_existing_goal_ledger_is_reused_without_recomputation(tmp_path, monkeypatch):
    path = tmp_path / "strategy_goals.json"
    expected = [
        ExpectedGoal(
            goal_id="goal-1",
            handler="withdrawal_request",
            kind="input_profile",
            labels=("request_kind:full_exit",),
            symbolic=True,
            completable=True,
        )
    ]
    write_goal_ledger(path, expected)

    def fail_recompute(_generation_configs):
        raise AssertionError("expected cached ledger to be reused")

    monkeypatch.setattr(lean_report, "expected_goals_from_generation_configs", fail_recompute)

    assert (
        lean_report.load_or_create_expected_goals(
            generation_configs=[{"mode": "input_profile"}],
            ledger_path=path,
            refresh=False,
        )
        == expected
    )
