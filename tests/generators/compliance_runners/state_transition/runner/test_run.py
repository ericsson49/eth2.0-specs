from __future__ import annotations

from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from snappy import uncompress

from eth_consensus_specs.test.context import expect_assertion_error
from eth_consensus_specs.test.helpers.specs import spec_targets
from eth_consensus_specs.utils import bls

OPERATION_INPUTS = {
    "proposer_slashing": ("proposer_slashing", "ProposerSlashing"),
    "attester_slashing": ("attester_slashing", "AttesterSlashing"),
    "attestation": ("attestation", "Attestation"),
    "deposit": ("deposit", "Deposit"),
    "bls_to_execution_change": ("address_change", "SignedBLSToExecutionChange"),
    "deposit_request": ("deposit_request", "DepositRequest"),
    "voluntary_exit": ("voluntary_exit", "SignedVoluntaryExit"),
    "withdrawal_request": ("withdrawal_request", "WithdrawalRequest"),
    "consolidation_request": ("consolidation_request", "ConsolidationRequest"),
}

OPERATION_PROCESSORS = {
    "proposer_slashing": "process_proposer_slashing",
    "attester_slashing": "process_attester_slashing",
    "attestation": "process_attestation",
    "deposit": "process_deposit",
    "bls_to_execution_change": "process_bls_to_execution_change",
    "deposit_request": "process_deposit_request",
    "voluntary_exit": "process_voluntary_exit",
    "withdrawal_request": "process_withdrawal_request",
    "consolidation_request": "process_consolidation_request",
}

EPOCH_PROCESSORS = {
    "justification_and_finalization": "process_justification_and_finalization",
    "registry_updates": "process_registry_updates",
    "slashings": "process_slashings",
    "pending_deposits": "process_pending_deposits",
    "pending_consolidations": "process_pending_consolidations",
    "effective_balance_updates": "process_effective_balance_updates",
}

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
    input_name = OPERATION_INPUTS.get(handler, (handler, None))[0]
    if not (test_dir / f"{input_name}.ssz_snappy").exists():
        return None
    return decode_operation(spec, test_dir, handler)


def decode_operation(spec, test_dir: Path, handler: str):
    if handler in OPERATION_INPUTS:
        input_name, type_name = OPERATION_INPUTS[handler]
        return decode_file(spec, test_dir, input_name, getattr(spec, type_name))
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
    old_bls_active = bls.bls_active
    bls.bls_active = bool(test_case["meta"].get("bls_setting", 0))

    try:
        if runner == "epoch_processing":
            run_epoch_processing_case(spec, state, handler, expected_post)
            return

        if runner != "operations":
            raise ValueError(f"Unsupported state-transition runner: {runner}")

        if handler in OPERATION_PROCESSORS:
            process_fn = getattr(spec, OPERATION_PROCESSORS[handler])
            run_processing_case(
                process_fn,
                state,
                test_case["operation"],
                expected_post,
            )
            return

        raise ValueError(f"Unsupported operations handler: {handler}")
    finally:
        bls.bls_active = old_bls_active


def run_epoch_processing_case(spec, state, handler, expected_post):
    if handler not in EPOCH_PROCESSORS:
        raise ValueError(f"Unsupported epoch_processing handler: {handler}")
    process_fn = getattr(spec, EPOCH_PROCESSORS[handler])
    run_processing_case(process_fn, state, None, expected_post)


def run_processing_case(process_fn, state, operation, expected_post):
    def run_processing():
        if operation is None:
            process_fn(state)
        else:
            process_fn(state, operation)

    if expected_post is None:
        expect_assertion_error(run_processing)
        return

    run_processing()
    assert state == expected_post


def gather_tests(tests_dir) -> Iterable[StateTransitionTestInfo]:
    if isinstance(tests_dir, (list, tuple)):
        for path in tests_dir:
            yield from gather_tests(path)
        return

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
