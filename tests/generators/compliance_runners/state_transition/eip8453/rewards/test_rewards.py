"""Reward vectors: independent integer oracle, complete coverage, and replay."""

from math import isqrt
from types import SimpleNamespace

import pytest

from eth_consensus_specs.eip8453 import mainnet, minimal
from eth_consensus_specs.gloas import minimal as gloas
from tests.generators.compliance_runners.runner.discovery import gather_tests
from tests.generators.compliance_runners.runner.state_transition import run_test
from tests.generators.compliance_runners.state_transition.evaluation.coverage_dsl import (
    Context,
    score,
)
from tests.generators.compliance_runners.state_transition.provider import (
    load_provider,
    run,
    validate_handler,
)

from .materializer import RewardsMaterializer
from .observation import observe_attributes
from .target import TARGET


def expected_rewards(spec, state):
    """Independent arithmetic using Python ints/isqrt, no executable-spec helpers."""
    epoch = int(state.slot) // int(spec.SLOTS_PER_EPOCH)
    previous_round = max(0, int(state.slot) // int(spec.SLOTS_PER_ROUND) - 1)
    delay = max(0, previous_round - int(state.finalized_slot) // int(spec.SLOTS_PER_ROUND))
    rewards = [0] * len(state.validators)
    if delay > int(spec.MIN_ROUNDS_TO_INACTIVITY_LEAK):
        return rewards
    active = [
        i
        for i, v in enumerate(state.validators)
        if int(v.activation_epoch) <= epoch < int(v.exit_epoch)
    ]
    increment = int(spec.EFFECTIVE_BALANCE_INCREMENT)
    total = max(increment, sum(int(state.validators[i].effective_balance) for i in active))
    per_increment = increment * int(spec.ROUND_REWARD_FACTOR) // isqrt(total)
    for flag_index, weight in enumerate(spec.PARTICIPATION_FLAG_WEIGHTS_EIP8453):
        participants = [
            i
            for i in active
            if not state.validators[i].slashed
            and int(state.round_participation[i]) & (1 << flag_index)
        ]
        participating = (
            max(increment, sum(int(state.validators[i].effective_balance) for i in participants))
            // increment
        )
        for i in participants:
            base = (int(state.validators[i].effective_balance) // increment) * per_increment
            rewards[i] += (
                base
                * int(weight)
                * participating
                // ((total // increment) * int(spec.WEIGHT_DENOMINATOR))
            )
    return rewards


def assert_execution(spec, pre):
    expected = pre.copy()
    rewards = expected_rewards(spec, pre)
    for i, reward in enumerate(rewards):
        expected.balances[i] += reward
    actual = pre.copy()
    spec.process_rewards(actual)
    assert actual == expected  # Includes flags, scores, validators, and checkpoint fields.
    return rewards


@pytest.mark.parametrize("spec", [minimal, mainnet], ids=["minimal", "mainnet"])
def test_all_witnesses_match_observer_and_exact_rewards(spec):
    provider = load_provider("eip8453_rewards", spec=spec)
    target = TARGET.for_spec(spec)
    materializer = RewardsMaterializer(spec, preset_name=spec.config.PRESET_BASE)
    records = []
    paid = unpaid = 0
    for assignment in provider.build_profile("max", spec=spec):
        pre = materializer.build_pre_state(SimpleNamespace(**assignment))
        assert (int(pre.slot) + 1) % int(spec.SLOTS_PER_ROUND) == 0
        observed = provider.recover(pre, None)
        assert observed == assignment
        records.append(observed)
        rewards = assert_execution(spec, pre)
        if any(rewards):
            paid += 1
        else:
            unpaid += 1
        if assignment["delay_vs_leak"] in ("GT_1", "GT_FAR"):
            assert rewards == [0, 0, 0]
        if not assignment["active_validator"] or assignment["validator_slashed"]:
            assert rewards[0] == 0
    assert paid > 0
    assert unpaid > 0
    for name in target.profiles:
        report = score(target, records, target.profiles[name])
        assert report.covered == report.total > 0
        assert report.unexpected == report.uncovered == []


@pytest.mark.parametrize("spec", [minimal, mainnet], ids=["minimal", "mainnet"])
def test_disk_generation_validation_and_replay(tmp_path, spec):
    assert run("rewards", tmp_path, "smoke", spec.config.PRESET_BASE, fork="eip8453") == 0
    cases = list(gather_tests(tmp_path, kind="state_transition"))
    assert cases
    assert {(c.fork, c.runner, c.handler) for c in cases} == {
        ("eip8453", "round_processing", "rewards")
    }
    assert validate_handler(tmp_path, "rewards") == 0
    for case in cases:
        run_test(case)


def baseline(spec):
    return RewardsMaterializer(spec).build_pre_state(
        SimpleNamespace(
            active_validator=True,
            validator_slashed=False,
            progress_flag=True,
            finality_flag=True,
            balance_increments="MANY",
            other_progress_participants=True,
            other_finality_participants=True,
            delay_vs_leak="EQ",
        )
    )


def test_clamped_finality_delay_and_empty_active_balance():
    spec = minimal
    state = baseline(spec)
    state.finalized_slot = state.slot
    assert observe_attributes(Context(spec, state, None, None, {}))["finality_delay"] == 0
    assert any(assert_execution(spec, state))
    for validator in state.validators:
        validator.activation_epoch = spec.FAR_FUTURE_EPOCH
    assert assert_execution(spec, state) == [0, 0, 0]
    for validator in state.validators:
        validator.activation_epoch = 0
        validator.effective_balance = 0
    assert assert_execution(spec, state) == [0, 0, 0]


def test_supporting_flags_and_slashed_balances_are_not_ignored():
    spec = minimal
    state = baseline(spec)
    full = assert_execution(spec, state)
    state.round_participation[1] = spec.ParticipationFlags()
    partial = assert_execution(spec, state)
    assert partial[0] < full[0]
    assert partial[1] == 0
    state.validators[2].slashed = spec.Boolean(1)
    reduced = assert_execution(spec, state)
    assert reduced[0] < partial[0]
    assert reduced[2] == 0
    # Inactivity scores affect finality weights, not reward balances.
    for i in range(len(state.inactivity_scores)):
        state.inactivity_scores[i] = spec.EJECTION_INACTIVITY_SCORE
    assert assert_execution(spec, state) == reduced


def test_rejects_wrong_fork_and_missing_dimensions():
    with pytest.raises(ValueError, match="requires the EIP-8453 fork"):
        RewardsMaterializer(gloas)
    with pytest.raises(ValueError, match="missing active dimension"):
        RewardsMaterializer(minimal).build_pre_state(SimpleNamespace())


def test_each_flag_is_rounded_before_adding_rewards():
    spec = minimal
    state = baseline(spec)
    increment = int(spec.EFFECTIVE_BALANCE_INCREMENT)
    state.validators[0].effective_balance = 4 * increment
    total = sum(int(v.effective_balance) for v in state.validators)
    per_increment = increment * int(spec.ROUND_REWARD_FACTOR) // isqrt(total)
    combined = [
        (int(v.effective_balance) // increment)
        * per_increment
        * sum(int(w) for w in spec.PARTICIPATION_FLAG_WEIGHTS_EIP8453)
        // int(spec.WEIGHT_DENOMINATOR)
        for v in state.validators
    ]
    # Everyone participates in both flags here. Combining their numerators
    # before division incorrectly rounds up at least one validator's reward.
    separate = assert_execution(spec, state)
    assert any(
        actual < combined_reward for actual, combined_reward in zip(separate, combined, strict=True)
    )
