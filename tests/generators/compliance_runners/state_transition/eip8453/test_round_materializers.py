"""Single-validator round vectors preserve coverage and fork-specific routing."""

from types import SimpleNamespace

import pytest
from ruamel.yaml import YAML

from eth_consensus_specs.test.helpers.specs import spec_targets
from tests.generators.compliance_runners.runner.discovery import gather_tests
from tests.generators.compliance_runners.runner.state_transition import run_test
from tests.generators.compliance_runners.state_transition.catalog import ALL_HANDLERS
from tests.generators.compliance_runners.state_transition.provider import (
    discover_handlers,
    load_provider,
    providers_for,
    run,
    validate_handler,
)

HANDLERS = ("inactivity_updates", "participation_flag_updates")


@pytest.mark.parametrize("preset", ["minimal", "mainnet"])
@pytest.mark.parametrize("selected_handler", HANDLERS)
def test_every_witness_and_full_post_state(preset, selected_handler):
    spec = spec_targets[preset]["eip8453"]
    provider = load_provider(f"eip8453_{selected_handler}", spec=spec)
    materializer = provider.MATERIALIZER(spec, preset_name=preset)
    for record in provider.build_profile("max", spec=spec):
        solution = SimpleNamespace(**record)
        pre = materializer.build_pre_state(solution)
        assert (int(pre.slot) + 1) % int(spec.SLOTS_PER_ROUND) == 0
        assert len(pre.validators) == 1
        assert provider.recover(pre, None) == record
        expected = pre.copy()
        if selected_handler == "inactivity_updates":
            for i, validator in enumerate(pre.validators):
                epoch = int(pre.slot) // int(spec.SLOTS_PER_EPOCH)
                if not int(validator.activation_epoch) <= epoch < int(validator.exit_epoch):
                    continue
                if not validator.slashed and int(pre.round_participation[i]) & (
                    1 << int(spec.PROGRESS_FLAG_INDEX)
                ):
                    expected.inactivity_scores[i] = max(
                        0,
                        int(pre.inactivity_scores[i])
                        - int(spec.config.INACTIVITY_SCORE_RECOVERY_RATE),
                    )
                else:
                    expected.inactivity_scores[i] += spec.config.INACTIVITY_SCORE_BIAS
                    if expected.inactivity_scores[i] >= spec.EJECTION_INACTIVITY_SCORE:
                        # Churn allocation is owned by the exit helper; verify
                        # the round handler calls it for precisely these inputs.
                        spec.initiate_validator_exit(expected, i)
        else:
            expected.round_participation = spec.RoundParticipation(
                data=[
                    spec.ParticipationFlags((1 << int(spec.PROGRESS_FLAG_INDEX)) if bit else 0)
                    for bit in pre.progress_participation
                ]
            )
        post = pre.copy()
        getattr(spec, f"process_{selected_handler}")(post)
        assert post == expected


@pytest.mark.parametrize("preset", ["minimal", "mainnet"])
def test_all_eip8453_generation_validation_and_replay(tmp_path, preset):
    assert run("all", tmp_path, "smoke", preset, fork="eip8453", seed=124) == 0
    assert set(discover_handlers(tmp_path)) == {*HANDLERS, "height_events"}
    cases = list(gather_tests(tmp_path, kind="state_transition"))
    assert {case.runner for case in cases} == {"operations", "round_processing"}
    for selected_handler in HANDLERS:
        assert validate_handler(tmp_path, selected_handler) == 0
    for case in cases:
        run_test(case)

        if case.runner == "round_processing":
            meta = YAML(typ="safe").load((case.test_dir / "meta.yaml").read_text())
            assert "validator_index" not in meta


@pytest.mark.parametrize("selected_handler", HANDLERS)
def test_mixed_forks_keep_distinct_formats(tmp_path, selected_handler):
    for fork in ("gloas", "eip8453"):
        assert run(selected_handler, tmp_path, "smoke", fork=fork) == 0
    cases = [case for case in gather_tests(tmp_path) if case.handler == selected_handler]
    assert {(case.fork, case.runner) for case in cases} == {
        ("gloas", "epoch_processing"),
        ("eip8453", "round_processing"),
    }
    assert discover_handlers(tmp_path) == [selected_handler]
    assert validate_handler(tmp_path, selected_handler) == 0
    for case in cases:
        run_test(case)

    # A round vector filed as epoch processing must not pass claim validation.
    case = next(case for case in cases if case.fork == "eip8453")
    wrong = (
        tmp_path
        / "minimal"
        / "eip8453"
        / "epoch_processing"
        / selected_handler
        / case.suite
        / case.test_dir.name
    )
    wrong.parent.mkdir(parents=True)
    case.test_dir.rename(wrong)
    yaml = YAML()
    manifest_path = wrong / "manifest.yaml"
    manifest = yaml.load(manifest_path.read_text())
    manifest["runner"] = "epoch_processing"
    with manifest_path.open("w") as stream:
        yaml.dump(manifest, stream)
    with pytest.raises(ValueError, match="does not support"):
        validate_handler(tmp_path, selected_handler)
    misplaced = next(case for case in gather_tests(tmp_path) if case.test_dir == wrong)
    with pytest.raises(ValueError, match="belongs to round_processing"):
        run_test(misplaced)


def test_registration_and_fork_guards():
    assert len(ALL_HANDLERS) == len(set(ALL_HANDLERS))
    for selected_handler in HANDLERS:
        assert all(
            p.runner == "round_processing" for p in providers_for(selected_handler, "eip8453")
        )
        assert all(p.runner == "epoch_processing" for p in providers_for(selected_handler, "gloas"))
        with pytest.raises(ValueError, match="does not support fork"):
            load_provider(f"eip8453_{selected_handler}", spec=spec_targets["minimal"]["gloas"])
