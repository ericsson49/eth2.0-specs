"""Independent concrete corpora cover each target and exercise the executable spec."""

from itertools import product

import pytest

from eth_consensus_specs.eip8453 import mainnet, minimal
from tests.generators.compliance_runners.state_transition.evaluation.coverage_dsl import (
    Context,
    score,
)

from .inactivity_updates.observation import observe_attributes as observe_inactivity
from .inactivity_updates.target import TARGET as INACTIVITY
from .participation_flag_updates.observation import observe_attributes as observe_flags
from .participation_flag_updates.target import TARGET as FLAGS


def context(
    spec, *, active=True, slashed=False, exit_set=False, flags=0, progress=False, inactivity=0
):
    epoch = 3
    validator = spec.Validator(
        activation_epoch=0 if active or exit_set else epoch + 1,
        exit_epoch=(epoch + 1 if active else epoch) if exit_set else spec.FAR_FUTURE_EPOCH,
        withdrawable_epoch=spec.FAR_FUTURE_EPOCH,
        slashed=slashed,
        effective_balance=spec.MIN_ACTIVATION_BALANCE,
    )
    state = spec.BeaconState(
        slot=epoch * spec.SLOTS_PER_EPOCH,
        validators=spec.Validators(data=[validator]),
        balances=spec.Balances(data=[spec.MIN_ACTIVATION_BALANCE]),
        inactivity_scores=spec.InactivityScores(data=[inactivity]),
        round_participation=spec.RoundParticipation(data=[spec.ParticipationFlags(flags)]),
        progress_participation=spec.HeightParticipation(data=[spec.Boolean(progress)]),
        target_participation=spec.HeightParticipation(data=[spec.Boolean()]),
    )
    return Context(spec, state, None, None, {})


def assert_complete(target, records):
    for name in ("smoke", "normal", "standard", "max"):
        report = score(target, records, target.profiles[name])
        assert report.total > 0
        assert report.covered == report.total
        assert report.uncovered == report.unexpected == []


@pytest.mark.parametrize("spec", [minimal, mainnet], ids=["minimal", "mainnet"])
def test_inactivity_corpus_covers_profiles_and_matches_execution(spec):
    target = INACTIVITY.for_spec(spec)
    recovery = int(spec.config.INACTIVITY_SCORE_RECOVERY_RATE)
    bias = int(spec.config.INACTIVITY_SCORE_BIAS)
    threshold = int(spec.EJECTION_INACTIVITY_SCORE)
    # Explicit specification boundaries, independently of the target callback.
    scores = [
        0,
        recovery - 2,
        recovery - 1,
        recovery,
        recovery + 1,
        recovery + 2,
        threshold - bias - 2,
        threshold - bias - 1,
        threshold - bias,
        threshold - bias + 1,
        threshold - bias + 2,
    ]
    records = []
    for active, slashed, flag, exit_set in product((False, True), repeat=4):
        for old_score in scores:
            ctx = context(
                spec,
                active=active,
                slashed=slashed,
                exit_set=exit_set,
                flags=(1 << int(spec.PROGRESS_FLAG_INDEX)) if flag else 0,
                inactivity=old_score,
            )
            record = target.record(target.observation(ctx))
            records.append(record)
            expected_score = old_score
            if active:
                expected_score = (
                    max(0, old_score - recovery) if flag and not slashed else old_score + bias
                )
            request_exit = active and (slashed or not flag) and expected_score >= threshold
            assert record["exit_requested"] == request_exit
            assert record["new_exit"] == (request_exit and not exit_set)
            post = ctx.pre.copy()
            spec.process_inactivity_updates(post)
            assert int(post.inactivity_scores[0]) == expected_score
            assert post.balances == ctx.pre.balances
            if request_exit and not exit_set:
                assert post.validators[0].exit_epoch != spec.FAR_FUTURE_EPOCH
            else:
                assert post.validators[0].exit_epoch == ctx.pre.validators[0].exit_epoch
    assert_complete(target, records)


@pytest.mark.parametrize("spec", [minimal, mainnet], ids=["minimal", "mainnet"])
def test_flag_corpus_covers_profiles_and_matches_execution(spec):
    target = FLAGS.for_spec(spec)
    progress_mask = 1 << int(spec.PROGRESS_FLAG_INDEX)
    finality_mask = 1 << int(spec.FINALITY_FLAG_INDEX)
    records = []
    for active, slashed, old_progress, old_finality, carry in product((False, True), repeat=5):
        old_flags = (progress_mask if old_progress else 0) | (finality_mask if old_finality else 0)
        ctx = context(spec, active=active, slashed=slashed, flags=old_flags, progress=carry)
        records.append(target.record(target.observation(ctx)))
        post = ctx.pre.copy()
        spec.process_participation_flag_updates(post)
        assert int(post.round_participation[0]) == (progress_mask if carry else 0)
        assert post.progress_participation == ctx.pre.progress_participation
        assert post.target_participation == ctx.pre.target_participation
        assert post.validators == ctx.pre.validators
    assert_complete(target, records)


@pytest.mark.parametrize("observe", [observe_inactivity, observe_flags])
def test_explicit_multi_validator_focus_and_invalid_indices(observe):
    ctx = context(minimal)
    ctx.pre.validators.append(ctx.pre.validators[0].copy())
    ctx.pre.inactivity_scores.append(123)
    ctx.pre.round_participation.append(minimal.ParticipationFlags(3))
    ctx.pre.progress_participation.append(minimal.Boolean(1))
    with pytest.raises(ValueError, match="validator_index"):
        observe(ctx)
    for index in (-1, 2, True, "1"):
        with pytest.raises(ValueError, match="validator_index"):
            observe(Context(minimal, ctx.pre, None, None, {"validator_index": index}))
    focused = Context(minimal, ctx.pre, None, None, {"validator_index": 1})
    attrs = observe(focused)
    if observe is observe_inactivity:
        assert attrs["score"] == 123
        assert attrs["progress_flag"] is True
    else:
        assert attrs["old_progress"] is True
        assert attrs["old_other_flags"] is True
        assert attrs["height_progress"] is True


def test_inactivity_reads_round_flags_not_height_participation():
    ctx = context(minimal, flags=0, progress=True)
    assert observe_inactivity(ctx)["progress_flag"] is False
    ctx = context(minimal, flags=1 << int(minimal.PROGRESS_FLAG_INDEX), progress=False)
    assert observe_inactivity(ctx)["progress_flag"] is True
