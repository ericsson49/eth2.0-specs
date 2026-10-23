"""Public state-transition generation profile checks."""

from types import SimpleNamespace

import pytest

from tests.generators.compliance_runners.state_transition.provider import load_provider

from . import conftest as generator_options, provider as runner, run as cli
from .catalog import ALL_HANDLERS, FORKS, handlers_for
from .provider import materialize_handler, PROFILES, PROVIDERS


def test_all_is_not_a_generation_profile(tmp_path):
    assert "all" not in PROFILES
    assert "max" in PROFILES
    with pytest.raises(ValueError, match="unknown profile: all"):
        materialize_handler("blocks", "all", tmp_path)


def test_parser_choices_include_handlers_enabled_on_any_fork():
    assert set(ALL_HANDLERS) == {handler for fork in FORKS for handler in handlers_for(fork)}
    assert "blocks" in ALL_HANDLERS
    assert "blocks" in handlers_for("eip8453")
    assert "blocks" not in handlers_for("gloas")
    assert "payload_attestation" not in ALL_HANDLERS


def test_cli_accepts_eip8453_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--fork",
            "eip8453",
            "--handler",
            "blocks",
            "--profile",
            "exceptional",
            "--comptests-output",
            str(tmp_path),
        ],
    )
    calls = []
    monkeypatch.setattr(cli, "run", lambda *args: calls.append(args) or 0)
    assert cli.main() == 0
    assert calls == [("blocks", tmp_path, "exceptional", "minimal", None, "eip8453")]


def test_pytest_parser_accepts_eip8453_blocks():
    parser = pytest.Parser(_ispytest=True)
    generator_options.pytest_addoption(parser)
    args = parser.parse(["--fork", "eip8453", "--handler", "blocks"])
    assert (args.fork, args.handler) == ("eip8453", "blocks")


def test_catalog_providers_expose_all_public_profiles():
    required = {"smoke", "normal", "standard", "max"}
    for provider in PROVIDERS:
        module = load_provider(provider.name)
        assert module.MATERIALIZER.runner_name == provider.runner, provider.name
        assert module.MATERIALIZER.handler_name == provider.handler, provider.name
        assert module.TARGET is load_provider(provider.module).TARGET
        assert callable(module.build_profile), provider.name
        profiles = set(module.TARGET.profiles)
        assert required <= profiles, provider.name
        if "exceptional" not in profiles:
            assert module.build_profile("exceptional", spec=object()) == [], provider.name


def test_shared_withdrawals_materializer_keeps_distinct_targets():
    pending = load_provider("builder_pending_withdrawal_processing")
    aggregate = load_provider("withdrawal_processing")
    assert pending.MATERIALIZER is aggregate.MATERIALIZER
    assert pending.TARGET is not aggregate.TARGET
    assert pending.TARGET.backend == "python"
    assert aggregate.TARGET.backend == "minizinc"


def test_unknown_provider_fails_explicitly():
    with pytest.raises(ValueError, match="unknown provider: missing"):
        load_provider("missing")


@pytest.mark.parametrize("selected_handler", ["blocks", "payload_attestation"])
def test_disabled_generation_handlers_still_validate(monkeypatch, tmp_path, selected_handler):
    registered = runner.providers_for(selected_handler)
    provider = registered[0]
    case = (
        tmp_path
        / "minimal"
        / "gloas"
        / provider.runner
        / selected_handler
        / "pyspec_tests"
        / "case_0000"
    )
    case.mkdir(parents=True)
    (case / "dimensions.yaml").write_text(f"test_provider: {provider.name}\nclaimed: {{}}\n")
    validated = []

    def validate(case_spec, case_dir):
        assert case_spec.fork == "gloas"
        validated.append(case_dir)
        return []

    validator = SimpleNamespace(for_spec=validate)
    monkeypatch.setattr(
        runner, "load_provider", lambda name: SimpleNamespace(validate_case=validator)
    )
    assert runner.validate_cases(tmp_path, selected_handler, validator) == 0
    assert runner.validate_handler(tmp_path, selected_handler) == 0
    assert validated == [case, case]
    assert runner.discover_handlers(tmp_path) == [selected_handler]
    with pytest.raises(ValueError, match="unsupported generation handler"):
        runner.run(selected_handler, tmp_path, "smoke")


def test_materialization_runner_consumes_case_lists(monkeypatch, tmp_path):
    selected_spec = SimpleNamespace(fork="gloas")
    cases = [{"value": 1}, {"value": 2}]

    def build(name, *, spec):
        assert name == "smoke"
        assert spec is selected_spec
        return cases

    class Materializer:
        runner_name = "epoch_processing"
        handler_name = "example"
        fork_name = "gloas"
        preset_name = "minimal"

        def __init__(self, spec, **kwargs):
            assert spec is selected_spec
            assert kwargs == {"fork_name": "gloas", "preset_name": "minimal", "seed": 124}

        def materialize_reps(self, directory, representatives, **kwargs):
            assert directory == tmp_path
            assert [vars(r) for r in representatives] == cases
            assert kwargs == {"case_offset": 7, "clean": False}
            return len(representatives)

    module = SimpleNamespace(
        build_profile=build,
        MATERIALIZER=Materializer,
        validate_case=object(),
    )
    monkeypatch.setattr(runner, "load_provider", lambda *args, **kwargs: module)
    provider = runner.Provider("example", "example", "epoch_processing", "example")
    assert runner._materialize_provider(
        provider,
        "smoke",
        tmp_path,
        7,
        clean=False,
        spec=selected_spec,
        preset_name="minimal",
        seed=124,
    ) == (module.validate_case, 2, "gloas")
