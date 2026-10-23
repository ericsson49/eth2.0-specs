"""Generation, disk discovery, claim validation, and normal compliance replay."""

import pytest
from ruamel.yaml import YAML

from eth_consensus_specs.test.helpers.constants import ALL_PHASES
from eth_consensus_specs.test.helpers.specs import spec_targets
from tests.generators.compliance_runners.runner.discovery import gather_tests
from tests.generators.compliance_runners.runner.state_transition import run_test
from tests.generators.compliance_runners.state_transition.catalog import handlers_for
from tests.generators.compliance_runners.state_transition.provider import (
    load_provider,
    materialize_handler,
    run,
    validate_handler,
)


@pytest.mark.parametrize("preset", ["minimal", "mainnet"])
def test_registered_generation_discovery_validation_and_replay(tmp_path, preset):
    assert run("height_events", tmp_path, "smoke", preset, fork="eip8453") == 0
    cases = list(gather_tests(tmp_path, kind="state_transition"))
    provider = load_provider("eip8453_height_events", spec=spec_targets[preset]["eip8453"])
    expected = provider.build_profile("smoke", spec=spec_targets[preset]["eip8453"])
    assert len(cases) == len(expected) > 0
    assert {(case.fork, case.preset, case.handler) for case in cases} == {
        ("eip8453", preset, "height_events")
    }
    assert validate_handler(tmp_path, "height_events") == 0
    for case in cases:
        run_test(case)

    case_dir = cases[0].test_dir
    post = case_dir / "post.ssz_snappy"
    saved_post = post.read_bytes()
    post.unlink()
    assert validate_handler(tmp_path, "height_events", {case_dir.name}) == 1
    post.write_bytes(saved_post)

    dimensions = case_dir / "dimensions.yaml"
    yaml = YAML()
    metadata = yaml.load(dimensions.read_text())
    metadata["claimed"]["target_positive"] = not metadata["claimed"]["target_positive"]
    with dimensions.open("w") as stream:
        yaml.dump(metadata, stream)
    assert validate_handler(tmp_path, "height_events", {case_dir.name}) == 1


def test_gloas_default_still_generates_and_replays(tmp_path):
    assert run("eth1_data_reset", tmp_path, "smoke") == 0
    cases = list(gather_tests(tmp_path, kind="state_transition"))
    assert cases
    assert all(case.fork == "gloas" for case in cases)
    assert validate_handler(tmp_path, "eth1_data_reset") == 0
    for case in cases:
        run_test(case)


def test_fork_support_is_explicit(tmp_path):
    assert "eip8453" not in ALL_PHASES
    assert set(handlers_for("eip8453")) == {
        "height_events",
        "operations",
        "head_vote",
        "attestation",
        "attester_slashing",
        "legacy_attester_slashing",
        "builder_pending_payments",
        "pending_deposits",
        "rewards",
        "inactivity_updates",
        "participation_flag_updates",
    }
    assert "height_events" not in handlers_for("gloas")
    for fork, handler in (("gloas", "height_events"), ("eip8453", "block_header")):
        with pytest.raises(ValueError, match="unsupported generation handler"):
            run(handler, tmp_path, "smoke", fork=fork)
    with pytest.raises(ValueError, match="does not support fork"):
        load_provider("eip8453_height_events", spec=spec_targets["minimal"]["gloas"])
    with pytest.raises(ValueError, match="does not match preset_name"):
        materialize_handler("height_events", "smoke", tmp_path, spec_targets["mainnet"]["eip8453"])
    assert not list(tmp_path.iterdir())
