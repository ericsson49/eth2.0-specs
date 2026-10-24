"""Replay isolation and serialized compliance vector tests."""

from types import SimpleNamespace

import pytest
import snappy

from eth_consensus_specs.test.context import get_copy_of_spec, spec_with_config_overrides
from eth_consensus_specs.test.helpers.genesis import create_genesis_state
from eth_consensus_specs.test.helpers.specs import spec_targets
from eth_consensus_specs.utils import bls
from tests.generators.compliance_runners.state_transition.provider import load_provider
from tests.generators.compliance_runners.state_transition.validation_helpers import bls_enabled

from . import fork_choice, state_transition
from .discovery import gather_tests, VectorCase


@pytest.mark.parametrize("invalid", [False, True])
def test_fork_choice_replay_restores_bls(tmp_path, invalid):
    spec = spec_targets["minimal"]["phase0"]
    state = create_genesis_state(
        spec, [spec.MAX_EFFECTIVE_BALANCE] * 64, spec.MAX_EFFECTIVE_BALANCE
    )
    block = spec.BeaconBlock(state_root=state.hash_tree_root())
    for name, value in (("anchor_state", state), ("anchor_block", block)):
        (tmp_path / f"{name}.ssz_snappy").write_bytes(snappy.compress(value.encode_bytes()))
    (tmp_path / "meta.yaml").write_text(f"bls_setting: {int(not bls.bls_active)}\n")
    (tmp_path / "steps.yaml").write_text("- unknown: true\n" if invalid else "[]\n")
    original = bls.bls_active
    case = VectorCase("minimal", "phase0", "fork_choice", "handler", "main", tmp_path)
    if invalid:
        with pytest.raises(AssertionError):
            fork_choice.run_test(case)
    else:
        fork_choice.run_test(case)
    assert bls.bls_active == original


@pytest.mark.parametrize("provider_name", ["operations", "operations_dispatch"])
@pytest.mark.parametrize("preset", ["minimal", "mainnet"])
@pytest.mark.parametrize("initial_bls", [False, True])
def test_gloas_blocks_generation_and_strict_replay(
    tmp_path, monkeypatch, provider_name, preset, initial_bls
):
    spec, _ = spec_with_config_overrides(
        get_copy_of_spec(spec_targets[preset]["gloas"]), {"GLOAS_FORK_EPOCH": 0}
    )
    monkeypatch.setitem(spec_targets[preset], "gloas", spec)
    monkeypatch.setattr(bls, "bls_active", initial_bls)
    provider = load_provider(provider_name, spec=spec)
    records = [
        record
        for profile in ("normal", "exceptional")
        for record in provider.build_profile(profile, spec=spec)
    ]
    assert {record["accepted"] for record in records} == {False, True}
    materializer = provider.MATERIALIZER(spec, preset_name=preset)
    materializer.test_provider = provider.TARGET.name
    materializer.materialize_reps(tmp_path, [SimpleNamespace(**record) for record in records])
    assert bls.bls_active is initial_bls
    cases = list(gather_tests([tmp_path]))
    assert len(cases) == len(records)
    for case in cases:
        vector = state_transition.get_test_case(spec, case.test_dir, "blocks")
        block = vector["blocks"][0]
        advanced = vector["pre"].copy()
        spec.process_slots(advanced, block.message.slot)
        # Rejected blocks must reach their operation checks with a valid proposer signature.
        with bls_enabled():
            assert spec.verify_block_signature(advanced, block)
        checks = provider.validate_case.for_spec(spec, case.test_dir)
        assert all(check.status == "ok" for check in checks)
        state_transition.run_test(case)
        assert bls.bls_active is initial_bls
