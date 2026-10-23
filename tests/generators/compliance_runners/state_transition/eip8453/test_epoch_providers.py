"""EIP-8453 changed epoch handlers: materialization, independent effects, replay."""

from types import SimpleNamespace

import pytest

from eth_consensus_specs.eip8453 import mainnet, minimal
from tests.generators.compliance_runners.runner.discovery import gather_tests
from tests.generators.compliance_runners.runner.state_transition import run_test
from tests.generators.compliance_runners.state_transition.evaluation.coverage_dsl import score
from tests.generators.compliance_runners.state_transition.provider import (
    load_provider,
    run,
    validate_handler,
)

HANDLERS = ("builder_pending_payments", "pending_deposits")


def expected_payments(s, pre):
    post = pre.copy()
    n = int(s.SLOTS_PER_EPOCH)
    for i in range(n):
        payment = pre.builder_pending_payments[i]
        votes = sum(bool(bit) for bit in pre.head_vote_participation[i])
        if payment.withdrawal.amount and votes * int(
            s.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR
        ) >= int(s.HEAD_VOTE_COMMITTEE_SIZE) * int(s.BUILDER_PAYMENT_THRESHOLD_NUMERATOR):
            post.builder_pending_withdrawals.append(payment.withdrawal)
    for i in range(n):
        post.builder_pending_payments[i] = pre.builder_pending_payments[n + i]
        post.builder_pending_payments[n + i] = s.BuilderPendingPayment.empty()
        post.head_vote_participation[i] = pre.head_vote_participation[n + i]
        post.head_vote_participation[n + i] = s.HeadVoteCommitteeBits()
    return post


def expected_deposits(s, pre):
    """Existing-validator effects; churn allocation itself is inherited."""
    post = pre.copy()
    budget = int(pre.deposit_balance_to_consume) + int(s.get_activation_churn_limit(pre))
    consumed = 0
    postponed = []
    next_epoch = int(pre.slot) // int(s.SLOTS_PER_EPOCH) + 1
    retained = 0
    stop = len(pre.pending_deposits)
    for i, deposit in enumerate(pre.pending_deposits):
        if int(deposit.slot) > int(pre.finalized_slot) or i >= int(
            s.MAX_PENDING_DEPOSITS_PER_EPOCH
        ):
            stop = i
            break
        index = next(j for j, v in enumerate(pre.validators) if v.pubkey == deposit.pubkey)
        validator = pre.validators[index]
        if int(validator.withdrawable_epoch) < next_epoch:
            post.balances[index] += deposit.amount
        elif int(validator.exit_epoch) != int(s.FAR_FUTURE_EPOCH):
            postponed.append(deposit)
        elif consumed + int(deposit.amount) > budget:
            stop = i
            retained = budget - consumed
            break
        else:
            post.balances[index] += deposit.amount
            consumed += int(deposit.amount)
    post.pending_deposits = s.PendingDeposits(data=list(pre.pending_deposits[stop:]) + postponed)
    post.deposit_balance_to_consume = s.Gwei(retained)
    return post


@pytest.mark.parametrize("spec", [minimal, mainnet], ids=["minimal", "mainnet"])
@pytest.mark.parametrize("selected_handler", HANDLERS)
def test_all_assignments_realized_and_exact_effects(spec, selected_handler):
    provider = load_provider(f"eip8453_{selected_handler}", spec=spec)
    materializer = provider.MATERIALIZER(spec, preset_name=spec.config.PRESET_BASE)
    target = provider.TARGET.for_spec(spec)
    records = []
    for record in provider.build_profile("max", spec=spec):
        pre = materializer.build_pre_state(SimpleNamespace(**record))
        assert (int(pre.slot) + 1) % int(spec.SLOTS_PER_EPOCH) == 0
        assert provider.recover(pre, None) == record
        records.append(record)
        expected = (
            expected_payments(spec, pre)
            if selected_handler == "builder_pending_payments"
            else expected_deposits(spec, pre)
        )
        post = pre.copy()
        getattr(spec, f"process_{selected_handler}")(post)
        assert post == expected
    for name in target.profiles:
        report = score(target, records, target.profiles[name])
        assert report.total == report.covered > 0
        assert report.unexpected == report.uncovered == []


@pytest.mark.parametrize("preset", ["minimal", "mainnet"])
@pytest.mark.parametrize("selected_handler", HANDLERS)
def test_generation_and_replay(tmp_path, preset, selected_handler):
    assert run(selected_handler, tmp_path, "smoke", preset, fork="eip8453") == 0
    cases = list(gather_tests(tmp_path))
    assert cases
    assert {(c.fork, c.runner, c.handler) for c in cases} == {
        ("eip8453", "epoch_processing", selected_handler)
    }
    assert validate_handler(tmp_path, selected_handler) == 0
    for case in cases:
        run_test(case)


def deposit_state(**overrides):
    values = {
        "deposit_vs_finalized_slot": "EQ",
        "finalized_position": "INTERIOR",
        "primary_role": "ACTIVE",
        "amount_vs_churn": "LT",
        "carry_nonzero": True,
        "suffix": "NONE",
    }
    values.update(overrides)
    return (
        load_provider("eip8453_pending_deposits")
        .MATERIALIZER(minimal)
        .build_pre_state(SimpleNamespace(**values))
    )


def test_unfinalized_same_epoch_deposit_stops_before_withdrawn_bypass():
    pre = deposit_state(
        deposit_vs_finalized_slot="GT", primary_role="WITHDRAWN", suffix="FINALIZED"
    )
    assert int(pre.pending_deposits[0].slot) // int(minimal.SLOTS_PER_EPOCH) == int(
        pre.finalized_slot
    ) // int(minimal.SLOTS_PER_EPOCH)
    post = pre.copy()
    minimal.process_pending_deposits(post)
    assert post.pending_deposits == pre.pending_deposits
    assert post.balances == pre.balances
    assert post.deposit_balance_to_consume == 0


def test_deposit_limit_and_empty_queue():
    pre = deposit_state(primary_role="WITHDRAWN")
    deposit = pre.pending_deposits[0]
    limit = int(minimal.MAX_PENDING_DEPOSITS_PER_EPOCH)
    pre.pending_deposits = minimal.PendingDeposits(data=[deposit.copy() for _ in range(limit + 1)])
    expected = expected_deposits(minimal, pre)
    minimal.process_pending_deposits(pre)
    assert pre == expected
    assert len(pre.pending_deposits) == 1
    pre.pending_deposits = minimal.PendingDeposits()
    pre.deposit_balance_to_consume = minimal.Gwei(7)
    expected = expected_deposits(minimal, pre)
    minimal.process_pending_deposits(pre)
    assert pre == expected


def test_changed_epoch_dispatch_excludes_round_work(monkeypatch):
    names = (
        "registry_updates",
        "slashings",
        "pending_deposits",
        "pending_consolidations",
        "builder_pending_payments",
        "effective_balance_updates",
        "slashings_reset",
        "randao_mixes_reset",
        "historical_summaries_update",
        "sync_committee_updates",
        "proposer_lookahead",
        "ptc_window",
    )
    calls = []
    for name in names:
        monkeypatch.setattr(minimal, f"process_{name}", lambda state, name=name: calls.append(name))

    def forbidden(state):
        pytest.fail("epoch processing executed round work")

    for name in ("round", "inactivity_updates", "rewards", "participation_flag_updates"):
        monkeypatch.setattr(minimal, f"process_{name}", forbidden)
    minimal.process_epoch(object())
    assert calls == list(names)


@pytest.mark.parametrize("selected_handler", HANDLERS)
def test_mixed_fork_epoch_providers(tmp_path, selected_handler):
    for fork in ("gloas", "eip8453"):
        assert run(selected_handler, tmp_path, "smoke", fork=fork) == 0
    cases = list(gather_tests(tmp_path))
    assert {(case.fork, case.runner) for case in cases} == {
        ("gloas", "epoch_processing"),
        ("eip8453", "epoch_processing"),
    }
    assert validate_handler(tmp_path, selected_handler) == 0
    for case in cases:
        run_test(case)
