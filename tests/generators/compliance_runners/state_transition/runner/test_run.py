from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from snappy import uncompress

from eth_consensus_specs.test.context import expect_assertion_error
from eth_consensus_specs.test.helpers.specs import spec_targets

StateTransitionTestInfo = namedtuple(
    "StateTransitionTestInfo",
    [
        "preset",
        "fork",
        "runner",
        "handler",
        "suite",
        "test_dir",
    ],
)


def read_yaml(path: Path):
    yaml = YAML(typ="safe")
    return yaml.load(path.read_text())


def read_ssz_snappy(path: Path) -> bytes:
    return uncompress(path.read_bytes())


def decode_file(spec, test_dir: Path, name: str, typ):
    return typ.decode_bytes(read_ssz_snappy(test_dir / f"{name}.ssz_snappy"))


def get_test_case(spec, test_dir: Path, handler: str):
    return {
        "meta": read_yaml(test_dir / "meta.yaml"),
        "pre": decode_file(spec, test_dir, "pre", spec.BeaconState),
        "operation": decode_optional_operation(spec, test_dir, handler),
        "post": decode_optional_post(spec, test_dir),
    }


def decode_optional_operation(spec, test_dir: Path, handler: str):
    if not (test_dir / f"{handler}.ssz_snappy").exists():
        return None
    return decode_operation(spec, test_dir, handler)


def decode_operation(spec, test_dir: Path, handler: str):
    if handler == "deposit_request":
        return decode_file(spec, test_dir, "deposit_request", spec.DepositRequest)
    if handler == "withdrawal_request":
        return decode_file(spec, test_dir, "withdrawal_request", spec.WithdrawalRequest)
    if handler == "consolidation_request":
        return decode_file(spec, test_dir, "consolidation_request", spec.ConsolidationRequest)
    raise ValueError(f"Unsupported operations handler: {handler}")


def decode_optional_post(spec, test_dir: Path):
    post_path = test_dir / "post.ssz_snappy"
    if not post_path.exists():
        return None
    return spec.BeaconState.decode_bytes(read_ssz_snappy(post_path))


def run_test(test_info: StateTransitionTestInfo):
    preset, fork, runner, handler, _, test_dir = test_info
    spec = spec_targets[preset][fork]

    test_case = get_test_case(spec, Path(test_dir), handler)
    state = test_case["pre"]
    expected_post = test_case["post"]

    if runner == "epoch_processing":
        run_epoch_processing_case(spec, state, handler, expected_post)
        return

    if runner != "operations":
        raise ValueError(f"Unsupported state-transition runner: {runner}")

    if handler == "deposit_request":
        run_deposit_request_case(spec, state, test_case["operation"], expected_post)
        return
    if handler == "withdrawal_request":
        run_withdrawal_request_case(spec, state, test_case["operation"], expected_post)
        return
    if handler == "consolidation_request":
        run_consolidation_request_case(spec, state, test_case["operation"], expected_post)
        return

    raise ValueError(f"Unsupported operations handler: {handler}")


def run_epoch_processing_case(spec, state, handler, expected_post):
    if handler == "pending_deposits":
        process_fn = spec.process_pending_deposits
    elif handler == "pending_consolidations":
        process_fn = spec.process_pending_consolidations
    else:
        raise ValueError(f"Unsupported epoch_processing handler: {handler}")
    if expected_post is None:
        expect_assertion_error(lambda: process_fn(state))
        return

    process_fn(state)
    assert state == expected_post


def run_deposit_request_case(spec, state, deposit_request, expected_post):
    if expected_post is None:
        expect_assertion_error(lambda: spec.process_deposit_request(state, deposit_request))
        return

    spec.process_deposit_request(state, deposit_request)
    assert state == expected_post


def run_withdrawal_request_case(spec, state, withdrawal_request, expected_post):
    if expected_post is None:
        expect_assertion_error(lambda: spec.process_withdrawal_request(state, withdrawal_request))
        return

    spec.process_withdrawal_request(state, withdrawal_request)
    assert state == expected_post


def run_consolidation_request_case(spec, state, consolidation_request, expected_post):
    if expected_post is None:
        expect_assertion_error(
            lambda: spec.process_consolidation_request(state, consolidation_request)
        )
        return

    spec.process_consolidation_request(state, consolidation_request)
    assert state == expected_post


def gather_tests(tests_dir) -> Iterable[StateTransitionTestInfo]:
    tests_path = Path(tests_dir)
    for preset in [p.name for p in tests_path.glob("*") if p.name in spec_targets]:
        for fork in [
            f.name for f in (tests_path / preset).glob("*") if f.name in spec_targets[preset]
        ]:
            for test_dir in sorted([td for td in (tests_path / preset / fork).glob("*/*/*/*")]):
                manifest_path = test_dir / "manifest.yaml"
                if not manifest_path.exists():
                    continue
                manifest = read_yaml(manifest_path)
                yield StateTransitionTestInfo(
                    preset,
                    fork,
                    manifest["runner"],
                    manifest["handler"],
                    manifest["suite"],
                    test_dir,
                )


def select_tests(tests, start=None, limit=None):
    if start is not None:
        tests = tests[start:]
    if limit is not None:
        tests = tests[:limit]
    return tests


def _test_id(test_info: StateTransitionTestInfo) -> str:
    return "::".join(
        [
            test_info.preset,
            test_info.fork,
            test_info.runner,
            test_info.handler,
            test_info.suite,
            Path(test_info.test_dir).name,
        ]
    )


def pytest_generate_tests(metafunc):
    if "test_info" not in metafunc.fixturenames:
        return

    tests_dir = metafunc.config.getoption("--test-dir")
    if tests_dir is None:
        raise pytest.UsageError(
            "--test-dir is required when running state-transition compliance tests"
        )

    start = metafunc.config.getoption("--start")
    limit = metafunc.config.getoption("--limit")
    test_infos = select_tests(list(gather_tests(tests_dir)), start=start, limit=limit)
    metafunc.parametrize(
        "test_info",
        test_infos,
        ids=[_test_id(test_info) for test_info in test_infos],
    )


def test_run_state_transition_case(test_info):
    run_test(test_info)
