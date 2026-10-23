"""Exercise every feasible witness and the serialized state-vector interface."""

from types import SimpleNamespace

import pytest

from eth_consensus_specs.eip8453 import mainnet, minimal
from tests.generators.compliance_runners.state_transition.evaluation.coverage_dsl import Context

from .materializer import HeightEventsMaterializer
from .target import TARGET


@pytest.mark.parametrize("spec", [minimal, mainnet], ids=["minimal", "mainnet"])
def test_all_feasible_witnesses_match_coverage_and_post_state(spec):
    target = TARGET.for_spec(spec)
    materializer = HeightEventsMaterializer(spec)
    for assignment in target.profiles["max"].run():
        solution = SimpleNamespace(**dict(assignment))
        pre = materializer.build_pre_state(solution)
        observed = target.record(target.observation(Context(spec, pre, None, None, {})))
        assert observed == vars(solution)
        assert all(
            not t or p
            for t, p in zip(pre.target_participation, pre.progress_participation, strict=True)
        )
        assert len(pre.pending_finality) <= spec.PENDING_FINALITY_WINDOW
        heights = [int(p.checkpoint.height) for p in pre.pending_finality]
        assert heights == sorted(set(heights))
        assert all(
            pre.finalized_checkpoint.height < h < pre.target_checkpoint.height for h in heights
        )

        # With these equal-weight fixtures, two votes qualify when weight is
        # nonzero. Derive the expected state directly, independently of the handler.
        qualifying = [
            p
            for p in pre.pending_finality
            if sum(bool(bit) for bit in p.participation) >= 2 and pre.inactivity_scores[0] == 0
        ]
        finalized = qualifying[-1] if qualifying else None
        checkpoint = finalized.checkpoint if finalized else pre.finalized_checkpoint
        retained = [p for p in pre.pending_finality if p.checkpoint.height > checkpoint.height]
        post = pre.copy()
        spec.process_height_events(post)
        assert post.finalized_checkpoint == checkpoint
        assert post.finalized_slot == (finalized.slot if finalized else pre.finalized_slot)
        if solution.advance_reason == "JUSTIFICATION":
            assert post.justified_checkpoint == pre.target_checkpoint
            if len(retained) == spec.PENDING_FINALITY_WINDOW:
                retained = retained[1:]
            retained.append(
                spec.PendingFinality(
                    checkpoint=pre.target_checkpoint,
                    slot=pre.target_slot,
                    participation=spec.HeightParticipation(data=[spec.Boolean()] * 3),
                )
            )
        else:
            assert post.justified_checkpoint == pre.justified_checkpoint
        assert list(post.pending_finality) == retained
        if solution.advance_reason == "NONE":
            assert post.target_checkpoint == pre.target_checkpoint
            assert post.target_slot == pre.target_slot
            assert post.target_participation == pre.target_participation
            assert post.progress_participation == pre.progress_participation
        else:
            assert post.target_checkpoint.height == pre.target_checkpoint.height + 1
            assert post.target_checkpoint.root == spec.Root()
            assert post.target_slot == pre.latest_block_header.slot
            assert not any(post.target_participation)
            assert not any(post.progress_participation)


@pytest.mark.parametrize("spec", [minimal, mainnet], ids=["minimal", "mainnet"])
def test_smoke_vectors_round_trip_and_claim_all_dimensions(spec):
    target = TARGET.for_spec(spec)
    materializer = HeightEventsMaterializer(spec)
    assert materializer.fork_name == "eip8453"
    for assignment in target.profiles["smoke"].generation_obligations():
        solution = SimpleNamespace(**dict(assignment))
        meta, parts = materializer.materialize_solution(solution)
        assert [(name, kind) for name, kind, _ in parts] == [("pre", "ssz"), ("post", "ssz")]
        pre, post = [spec.BeaconState.decode_bytes(data) for _, _, data in parts]
        assert meta["claimed"] == target.record(
            target.observation(Context(spec, pre, None, post, {}))
        )
        spec.process_height_events(pre)
        assert pre == post
        assert materializer.materialize_solution(solution) == (meta, parts)


def test_infeasible_witness_is_rejected():
    assignment = dict(next(iter(TARGET.for_spec(minimal).profiles["max"].run())))
    assignment.update(target_positive=True, progress_positive=False)
    with pytest.raises(ValueError, match="infeasible"):
        HeightEventsMaterializer(minimal).materialize_solution(SimpleNamespace(**assignment))
