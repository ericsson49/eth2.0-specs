"""Discover and replay serialized round sub-transitions independently."""

import pytest
import snappy

from eth_consensus_specs.test.helpers.specs import spec_targets
from eth_consensus_specs.utils import bls

from . import state_transition
from .discovery import gather_tests


@pytest.mark.parametrize("preset", ["minimal", "mainnet"])
@pytest.mark.parametrize("handler", ["inactivity_updates", "rewards", "participation_flag_updates"])
def test_round_subtransition_replay(tmp_path, monkeypatch, preset, handler):
    spec = spec_targets[preset]["eip8453"]
    flags = (1 << int(spec.PROGRESS_FLAG_INDEX)) | (1 << int(spec.FINALITY_FLAG_INDEX))
    pre = spec.BeaconState(
        slot=spec.SLOTS_PER_ROUND - 1,
        validators=spec.Validators(
            data=[
                spec.Validator(
                    activation_epoch=0,
                    exit_epoch=spec.FAR_FUTURE_EPOCH,
                    effective_balance=spec.MIN_ACTIVATION_BALANCE,
                )
            ]
        ),
        balances=spec.Balances(data=[spec.MIN_ACTIVATION_BALANCE]),
        inactivity_scores=spec.InactivityScores(data=[10]),
        round_participation=spec.RoundParticipation(data=[spec.ParticipationFlags(flags)]),
        progress_participation=spec.HeightParticipation(data=[spec.Boolean()]),
        target_participation=spec.HeightParticipation(data=[spec.Boolean()]),
    )
    post = pre.copy()
    getattr(spec, f"process_{handler}")(post)
    assert post != pre
    assert post.slot == pre.slot

    case_dir = tmp_path / preset / "eip8453" / "round_processing" / handler / "main" / "case_0"
    case_dir.mkdir(parents=True)
    original_bls = bls.bls_active
    (case_dir / "meta.yaml").write_text(f"bls_setting: {int(not original_bls)}\n")
    for name, state in (("pre", pre), ("post", post)):
        (case_dir / f"{name}.ssz_snappy").write_bytes(snappy.compress(state.encode_bytes()))

    def forbidden(*args):
        pytest.fail("replay executed a transition outside the selected handler")

    for name in ("process_slot", "process_slots", "process_round", "process_epoch"):
        monkeypatch.setattr(spec, name, forbidden)
    for other in state_transition.ROUND_PROCESSORS:
        if other != handler:
            monkeypatch.setattr(spec, f"process_{other}", forbidden)

    (case,) = gather_tests(tmp_path, kind="state_transition")
    assert case.runner == "round_processing"
    assert case.kind == "state_transition"
    assert not list(gather_tests(tmp_path, kind="fork_choice"))
    state_transition.run_test(case)
    assert bls.bls_active == original_bls

    # A mismatch anywhere in the state must fail, including unrelated fields.
    post.genesis_time += 1
    post_path = case_dir / "post.ssz_snappy"
    post_path.write_bytes(snappy.compress(post.encode_bytes()))
    with pytest.raises(AssertionError):
        state_transition.run_test(case)
    assert bls.bls_active == original_bls

    # Missing post means an expected abort; normal completion must fail.
    post_path.unlink()
    with pytest.raises(AssertionError):
        state_transition.run_test(case)
    assert bls.bls_active == original_bls

    def abort(state):
        assert state.slot == pre.slot
        raise AssertionError("invalid input")

    monkeypatch.setattr(spec, f"process_{handler}", abort)
    state_transition.run_test(case)
    assert bls.bls_active == original_bls


def test_round_processing_rejects_unknown_handler():
    spec = spec_targets["minimal"]["eip8453"]
    with pytest.raises(ValueError, match="Unsupported round_processing handler"):
        state_transition.run_round_processing_case(spec, None, "registry_updates", None)


@pytest.mark.parametrize("handler", ["inactivity_updates", "rewards", "participation_flag_updates"])
def test_round_processing_rejects_forks_without_rounds(handler):
    spec = spec_targets["minimal"]["gloas"]
    with pytest.raises(ValueError, match="not supported by fork: gloas"):
        state_transition.run_round_processing_case(spec, None, handler, None)
