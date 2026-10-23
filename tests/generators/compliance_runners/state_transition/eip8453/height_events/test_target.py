import pytest

from eth_consensus_specs.eip8453 import mainnet, minimal as spec
from tests.generators.compliance_runners.state_transition.evaluation.coverage_dsl import (
    Context,
    score,
)

from .observation import observe_attributes
from .target import TARGET


def context(target=(True, True, False), progress=None, *, delta=0, pending=()):
    state = spec.BeaconState(
        slot=spec.SLOTS_PER_EPOCH * 2,
        target_slot=spec.SLOTS_PER_EPOCH * 2 - spec.TIMEOUT_DELAY_SLOTS - delta,
        validators=spec.Validators(
            data=[
                spec.Validator(activation_epoch=0, exit_epoch=100, slashed=False) for _ in range(3)
            ]
        ),
        inactivity_scores=spec.InactivityScores(data=[0] * 3),
        target_participation=spec.HeightParticipation(data=[spec.Boolean(x) for x in target]),
        progress_participation=spec.HeightParticipation(
            data=[spec.Boolean(x) for x in (target if progress is None else progress)]
        ),
        pending_finality=spec.PendingFinalities(
            data=[
                spec.PendingFinality(
                    checkpoint=spec.HeightCheckpoint(height=i + 1),
                    slot=i + 1,
                    participation=spec.HeightParticipation(data=[spec.Boolean(x) for x in bits]),
                )
                for i, bits in enumerate(pending)
            ]
        ),
        finalized_checkpoint=spec.HeightCheckpoint(height=0),
        finalized_slot=0,
        target_checkpoint=spec.HeightCheckpoint(height=10),
        latest_block_header=spec.BeaconBlockHeader(slot=spec.SLOTS_PER_EPOCH * 2),
    )
    return Context(spec, state, None, None, {})


def record(ctx):
    return TARGET.record(TARGET.observation(ctx))


@pytest.mark.parametrize(("votes", "boundary"), [(1, "LT"), (2, "EQ"), (3, "GT")])
def test_quorum_boundary(votes, boundary):
    ctx = context([i < votes for i in range(3)])
    result = record(ctx)
    assert result["target_boundary"] == boundary
    assert result["advance_reason"] == ("NONE" if votes == 1 else "JUSTIFICATION")


@pytest.mark.parametrize(
    ("delta", "bucket"), [(-2, "LT_FAR"), (-1, "LT_1"), (0, "EQ"), (1, "GT_1"), (2, "GT_FAR")]
)
def test_timeout_boundary_and_justification_precedence(delta, bucket):
    ctx = context([False] * 3, [True] * 3, delta=delta)
    assert record(ctx)["timeout"] == bucket
    assert record(ctx)["advance_reason"] == ("TIMEOUT" if delta >= 0 else "NONE")
    assert record(context(delta=delta))["advance_reason"] == "JUSTIFICATION"


def test_zero_weight_is_not_quorum_even_at_equality():
    ctx = context([True] * 3)
    ctx.pre.inactivity_scores = spec.InactivityScores(data=[spec.EJECTION_INACTIVITY_SCORE] * 3)
    result = record(ctx)
    assert result["target_boundary"] == "EQ"
    assert result["target_positive"] is False
    assert result["advance_reason"] == "NONE"


def test_weights_filter_inactive_and_slashed_and_clamp_scores():
    ctx = context([True] * 3)
    ctx.pre.validators[0].slashed = True
    ctx.pre.validators[1].exit_epoch = 2
    ctx.pre.inactivity_scores[2] = spec.EJECTION_INACTIVITY_SCORE + 1
    attrs = observe_attributes(ctx)
    assert attrs["total_weight"] == spec.EJECTION_INACTIVITY_SCORE**2
    assert attrs["target_weight"] == attrs["progress_weight"] == 0


def test_last_qualifying_entry_prunes_earlier_nonqualifying_entries():
    ctx = context(pending=[[True] * 3, [False] * 3, [True] * 3, [False] * 3])
    attrs = observe_attributes(ctx)
    assert (attrs["pending_count"], attrs["qualifying_count"], attrs["retained_count"]) == (4, 2, 1)
    result = record(ctx)
    assert result["evict_oldest"] is False
    spec.process_height_events(ctx.pre)
    assert ctx.pre.finalized_checkpoint.height == 3
    assert [p.checkpoint.height for p in ctx.pre.pending_finality] == [4, 10]
    assert ctx.pre.target_checkpoint.height == 11
    assert not any(ctx.pre.target_participation)
    assert not any(ctx.pre.progress_participation)


def test_full_window_eviction_after_pruning():
    ctx = context(pending=[[False] * 3] * spec.PENDING_FINALITY_WINDOW)
    assert record(ctx)["evict_oldest"] is True
    spec.process_height_events(ctx.pre)
    assert [p.checkpoint.height for p in ctx.pre.pending_finality] == [2, 3, 4, 10]


def test_timeout_does_not_append_pending_finality():
    ctx = context([False] * 3, [True] * 3)
    assert record(ctx)["advance_reason"] == "TIMEOUT"
    spec.process_height_events(ctx.pre)
    assert len(ctx.pre.pending_finality) == 0
    assert ctx.pre.target_checkpoint.height == 11


def test_constants_bind_to_selected_preset():

    for selected in (spec, mainnet):
        assert TARGET.for_spec(selected).bound_constants == {
            "window": int(selected.PENDING_FINALITY_WINDOW)
        }


def test_profiles_enumerate_only_coherent_obligations():
    target = TARGET.for_spec(spec)
    for name in ("smoke", "normal", "standard", "max"):
        assignments = target.profiles[name].run()
        assert assignments
        if name != "max":
            continue
        for assignment in assignments:
            row = dict(assignment)
            assert row["evict_oldest"] == (
                row["advance_reason"] == "JUSTIFICATION" and row["window_full_after_pruning"]
            )
            if row["pending"] == "ZERO":
                assert row["qualifying"] == row["retained"] == "ZERO"
            if row["window_full_after_pruning"]:
                assert row["qualifying"] == "ZERO"


def test_observed_corpus_scores_and_reports_missing_coverage():

    target = TARGET.for_spec(spec)
    report = score(target, [record(context())], target.profiles["smoke"])
    assert 0 < report.covered < report.total
    assert report.uncovered
    assert not report.unexpected
