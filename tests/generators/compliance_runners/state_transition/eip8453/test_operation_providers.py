"""Changed block operations: independent effects, coverage recovery and disk replay."""

from math import isqrt
from types import SimpleNamespace

import pytest

from eth_consensus_specs.eip8453 import mainnet, minimal
from tests.generators.compliance_runners.runner.discovery import gather_tests
from tests.generators.compliance_runners.runner.io import read_yaml
from tests.generators.compliance_runners.runner.state_transition import run_test
from tests.generators.compliance_runners.state_transition.evaluation.cli import run_case
from tests.generators.compliance_runners.state_transition.evaluation.coverage_dsl import score
from tests.generators.compliance_runners.state_transition.provider import (
    load_provider,
    run,
    validate_handler,
)

from .attestation.observation import participants
from .operation_materializer import real_bls, sign

HANDLERS = (
    "head_vote",
    "attestation",
    "attester_slashing",
    "legacy_attester_slashing",
    "operations",
)


def expected_effects(s, selected_handler, pre, op):
    post = pre.copy()
    if selected_handler == "head_vote":
        slot = int(op.data.slot)
        same = slot == 0 or (
            op.data.beacon_block_root == pre.block_roots[slot % s.SLOTS_PER_HISTORICAL_ROOT]
            and op.data.beacon_block_root
            != pre.block_roots[(slot - 1) % s.SLOTS_PER_HISTORICAL_ROOT]
        )
        if same:
            index = (
                int(s.SLOTS_PER_EPOCH)
                if slot // int(s.SLOTS_PER_EPOCH) == int(pre.slot) // int(s.SLOTS_PER_EPOCH)
                else 0
            ) + slot % int(s.SLOTS_PER_EPOCH)
            for i, bit in enumerate(op.aggregation_bits):
                if bit:
                    post.head_vote_participation[index][i] = True
    elif selected_handler == "attestation":
        indices = {
            v for v, bit in zip(participants(s, pre, op), op.aggregation_bits, strict=True) if bit
        }
        total = sum(
            int(v.effective_balance)
            for v in pre.validators
            if v.activation_epoch <= int(pre.slot) // int(s.SLOTS_PER_EPOCH) < v.exit_epoch
        )
        increment = int(s.EFFECTIVE_BALANCE_INCREMENT)
        per_increment = increment * int(s.ROUND_REWARD_FACTOR) // isqrt(max(increment, total))
        numerator = 0
        for i in indices:
            reward = int(pre.validators[i].effective_balance) // increment * per_increment
            for pending in post.pending_finality:
                if pending.checkpoint == op.data.finality and not pending.participation[i]:
                    pending.participation[i] = s.Boolean(1)
                    post.round_participation[i] = s.ParticipationFlags(
                        int(post.round_participation[i]) | (1 << int(s.FINALITY_FLAG_INDEX))
                    )
                    numerator += reward * int(s.FINALITY_WEIGHT)
            if op.data.target == pre.target_checkpoint:
                if not post.progress_participation[i]:
                    post.progress_participation[i] = s.Boolean(1)
                    post.round_participation[i] = s.ParticipationFlags(
                        int(post.round_participation[i]) | (1 << int(s.PROGRESS_FLAG_INDEX))
                    )
                    numerator += reward * int(s.PROGRESS_WEIGHT)
                if not op.data.timeout:
                    post.target_participation[i] = s.Boolean(1)
        divisor = (
            (int(s.WEIGHT_DENOMINATOR) - int(s.PROPOSER_WEIGHT))
            * int(s.WEIGHT_DENOMINATOR)
            // int(s.PROPOSER_WEIGHT)
        )
        post.balances[pre.proposer_lookahead[pre.slot % s.SLOTS_PER_EPOCH]] += s.Gwei(
            numerator // divisor
        )
    elif selected_handler.endswith("attester_slashing"):
        # Validator slashing effects are inherited; independently select the intersection.
        for i in sorted(
            set(op.attestation_1.attesting_indices) & set(op.attestation_2.attesting_indices)
        ):
            v = pre.validators[i]
            if (
                not v.slashed
                and v.activation_epoch
                <= int(pre.slot) // int(s.SLOTS_PER_EPOCH)
                < v.withdrawable_epoch
            ):
                s.slash_validator(post, i)
    else:
        for name, operations in (
            ("attester_slashing", op.attester_slashings),
            ("legacy_attester_slashing", op.legacy_attester_slashings),
            ("attestation", op.attestations),
            ("head_vote", op.head_votes),
        ):
            for item in operations:
                post = expected_effects(s, name, post, item)
    return post


@pytest.mark.parametrize("s", [minimal, mainnet], ids=["minimal", "mainnet"])
@pytest.mark.parametrize("selected_handler", HANDLERS)
def test_all_records(s, selected_handler):
    # Check local processor effects directly; serialized process_operations
    # scenarios are exercised through sanity/blocks in the replay test below.
    provider = load_provider("eip8453_" + selected_handler, spec=s)
    materializer = provider.MATERIALIZER(s)
    records = provider.build_profile("max", spec=s)
    for record in records:
        pre, op = materializer.build_inputs(SimpleNamespace(**record))
        post = pre.copy()
        with real_bls():
            if record["accepted"]:
                getattr(s, "process_" + selected_handler)(post, op)
                assert post == expected_effects(s, selected_handler, pre, op)
            else:
                with pytest.raises(AssertionError):
                    getattr(s, "process_" + selected_handler)(post, op)
                post = None
        assert provider.recover(pre, op, post) == record
    target = provider.TARGET.for_spec(s)
    for name, profile in target.profiles.items():
        # Each generated profile must independently fulfill its own obligations.
        report = score(target, provider.build_profile(name, spec=s), profile)
        assert report.total == report.covered > 0
        assert report.uncovered == report.unexpected == []


@pytest.mark.parametrize("preset", ["minimal", "mainnet"])
@pytest.mark.parametrize("selected_handler", HANDLERS)
def test_generation_validation_replay(tmp_path, preset, selected_handler):
    handler = "blocks" if selected_handler == "operations" else selected_handler
    runner = "sanity" if selected_handler == "operations" else "operations"
    assert run(handler, tmp_path, "smoke", preset, fork="eip8453") == 0
    assert validate_handler(tmp_path, handler) == 0
    cases = list(gather_tests(tmp_path))
    assert cases
    assert {(c.runner, c.handler, c.fork, c.preset) for c in cases} == {
        (runner, handler, "eip8453", preset)
    }
    for case in cases:
        if handler == "blocks":
            assert (case.test_dir / "blocks_0.ssz_snappy").is_file()
            assert not (case.test_dir / "body.ssz_snappy").exists()
            assert read_yaml(case.test_dir / "meta.yaml")["blocks_count"] == 1
            assert (
                read_yaml(case.test_dir / "dimensions.yaml")["test_provider"]
                == "eip8453_operations"
            )
        run_test(case)
    spec = minimal if preset == "minimal" else mainnet
    run_case(cases[0], getattr(spec, "process_" + selected_handler))
    valid = next(c for c in cases if (c.test_dir / "post.ssz_snappy").exists())
    (valid.test_dir / "post.ssz_snappy").unlink()
    assert validate_handler(tmp_path, handler, {valid.test_dir.name}) == 1


def test_block_dispatch_runs_height_events_after_operations(monkeypatch):
    s = minimal
    provider = load_provider("eip8453_operations")
    pre, body = provider.MATERIALIZER(s).build_inputs(SimpleNamespace(case="MIXED"))
    calls = []
    for name in (
        "parent_execution_payload",
        "block_header",
        "withdrawals",
        "execution_payload_bid",
        "randao",
        "sync_aggregate",
    ):
        monkeypatch.setattr(s, "process_" + name, lambda *args, name=name: calls.append(name))
    for name in (
        "operations",
        "legacy_attester_slashing",
        "attestation",
        "head_vote",
        "height_events",
    ):
        fn = getattr(s, "process_" + name)

        def tracked(*args, name=name, fn=fn):
            calls.append(name)
            return fn(*args)

        monkeypatch.setattr(s, "process_" + name, tracked)
    with real_bls():
        s.process_block(pre, s.BeaconBlock(body=body))
    assert calls == [
        "parent_execution_payload",
        "block_header",
        "withdrawals",
        "execution_payload_bid",
        "randao",
        "operations",
        "legacy_attester_slashing",
        "attestation",
        "head_vote",
        "sync_aggregate",
        "height_events",
    ]
    assert pre.finalized_checkpoint.height == 5
    assert pre.justified_checkpoint.height == 10
    assert pre.target_checkpoint.height == 11


@pytest.mark.parametrize("selected_handler", ["attestation", "attester_slashing"])
def test_mixed_fork_operations(tmp_path, selected_handler):
    for fork in ("gloas", "eip8453"):
        assert run(selected_handler, tmp_path, "smoke", fork=fork) == 0
    assert validate_handler(tmp_path, selected_handler) == 0
    cases = list(gather_tests(tmp_path))
    assert {c.fork for c in cases} == {"gloas", "eip8453"}
    for case in cases:
        run_test(case)


def test_two_timeouts_can_still_conflict_with_finality():
    provider = load_provider("eip8453_attester_slashing")
    pre, op = provider.MATERIALIZER(minimal).build_inputs(
        SimpleNamespace(kind="TIMEOUT_CONFLICT", fault="NONE", multiple=False)
    )
    op.attestation_2.data.timeout = True
    op.attestation_2.signature = sign(
        minimal,
        pre,
        op.attestation_2.data,
        minimal.DOMAIN_FINALITY_ATTESTER,
        op.attestation_2.epoch,
        op.attestation_2.attesting_indices,
    )
    post = pre.copy()
    with real_bls():
        minimal.process_attester_slashing(post, op)
    record = provider.recover(pre, op, post)
    assert record["kind"] == "TIMEOUT_CONFLICT"
    assert record["inputs_valid"]
