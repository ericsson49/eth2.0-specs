from eth2spec.test.helpers.validator_state_profiles import validator_state_profiles
from eth2spec.test.helpers.withdrawals import (
    set_compounding_withdrawal_credential,
)
from eth2spec.test.helpers.state import (
    next_epoch_with_full_participation,
)


def compute_validator_epochs(spec, state, profile: dict) -> dict:
    # defaults
    activation_eligibility_epoch = spec.FAR_FUTURE_EPOCH
    activation_epoch = spec.FAR_FUTURE_EPOCH
    exit_epoch = spec.FAR_FUTURE_EPOCH
    withdrawable_epoch = spec.FAR_FUTURE_EPOCH

    if profile['activation_eligibility_epoch_finalized']:
        activation_eligibility_epoch = state.finalized_checkpoint.epoch
    elif profile['activation_eligibility_epoch_set']:
        activation_eligibility_epoch = state.finalized_checkpoint.epoch + 1

    if profile['activation_epoch_set']:
        assert activation_eligibility_epoch != spec.FAR_FUTURE_EPOCH
        # min 2 epochs to finalize activation_eligibility_epoch
        activation_epoch = spec.compute_activation_exit_epoch(activation_eligibility_epoch + 2)

    if profile['exit_epoch_set']:
        if profile['slashed']:
            if activation_epoch != spec.FAR_FUTURE_EPOCH:
                if (profile['shard_committee_period_lte_current_epoch']
                        and profile['exit_epoch_to_current_epoch'] in ['>', '=']):
                    # if (activation_epoch + spec.config.SHARD_COMMITTEE_PERIOD <= current_epoch <= exit_epoch)
                    # then (exit_epoch > activation_epoch + spec.config.SHARD_COMMITTEE_PERIOD)
                    exit_epoch = spec.compute_activation_exit_epoch(
                        activation_epoch + spec.config.SHARD_COMMITTEE_PERIOD)
                else:
                    # assume validator was slashed after activation
                    exit_epoch = spec.compute_activation_exit_epoch(activation_epoch)
            else:
                # use get_current_epoch(state) if activation epoch isn't set
                exit_epoch = spec.compute_activation_exit_epoch(spec.get_current_epoch(state))
        else:
            assert activation_epoch != spec.FAR_FUTURE_EPOCH
            # must satisfy voluntary exit condition
            exit_epoch = spec.compute_activation_exit_epoch(activation_epoch + spec.config.SHARD_COMMITTEE_PERIOD)

    if profile['withdrawable_epoch_set']:
        assert exit_epoch != spec.FAR_FUTURE_EPOCH
        if profile['slashed']:
            withdrawable_epoch = exit_epoch + spec.EPOCHS_PER_SLASHINGS_VECTOR
        else:
            withdrawable_epoch = exit_epoch + spec.config.MIN_VALIDATOR_WITHDRAWABILITY_DELAY

    return {
        'activation_eligibility_epoch': spec.Epoch(activation_eligibility_epoch),
        'activation_epoch': spec.Epoch(activation_epoch),
        'exit_epoch': spec.Epoch(exit_epoch),
        'withdrawable_epoch': spec.Epoch(withdrawable_epoch),
    }


def compute_current_epoch(spec, state, profile, validator) -> int:
    constraints = [
        (validator.activation_epoch, profile['activation_epoch_to_current_epoch']),
        (validator.exit_epoch, profile['exit_epoch_to_current_epoch']),
        (validator.withdrawable_epoch, profile['withdrawable_epoch_to_current_epoch']),
    ]
    if validator.activation_epoch != spec.FAR_FUTURE_EPOCH:
        constraints.append(
            (validator.activation_epoch + spec.config.SHARD_COMMITTEE_PERIOD - 1,
                '<' if profile['shard_committee_period_lte_current_epoch'] else '>')
        )

    lower_boundary = spec.get_current_epoch(state)
    upper_boundary = spec.FAR_FUTURE_EPOCH
    for value, op in constraints:
        if op == '=':
            # strict equality
            return value
        # lower boundary
        if op == '<' and lower_boundary < value:
            lower_boundary = value + 1
        # upper boundary
        if op == '>' and upper_boundary > value:
            upper_boundary = value - 1

    if upper_boundary < spec.FAR_FUTURE_EPOCH:
        return upper_boundary
    else:
        return lower_boundary


def compute_effective_balance(spec, profile: dict) -> int:
    if profile['balance_is_zero'] and profile['balance_to_effective_balance'] == '=':
        return spec.Gwei(0)

    if profile['effective_balance_lte_ejection_balance']:
        return spec.config.EJECTION_BALANCE

    if profile['effective_balance_to_min_activation_balance'] == '<':
        return spec.MIN_ACTIVATION_BALANCE - spec.EFFECTIVE_BALANCE_INCREMENT

    if profile['effective_balance_to_min_activation_balance'] == '=':
        return spec.MIN_ACTIVATION_BALANCE

    # profile['effective_balance_to_min_activation_balance'] == '>'
    # must be compounding credentials
    assert profile['withdrawal_credential_type'] == 'COMP'

    if profile['effective_balance_to_max_effective_balance'] == '<':
        return spec.MIN_ACTIVATION_BALANCE + spec.EFFECTIVE_BALANCE_INCREMENT

    # profile['effective_balance_to_max_effective_balance'] == '='
    return spec.MAX_EFFECTIVE_BALANCE_ELECTRA


def compute_balance(spec, profile: dict, effective_balance) -> int:
    if profile['balance_is_zero']:
        return spec.Gwei(0)

    # Use value that doesn't affect effective balance
    BALANCE_DEVIATION = spec.EFFECTIVE_BALANCE_INCREMENT // 10

    if profile['balance_to_effective_balance'] == '<':
        return effective_balance - BALANCE_DEVIATION

    if profile['balance_to_effective_balance'] == '=':
        return effective_balance

    # profile['balance_to_effective_balance'] == '>'
    return effective_balance + BALANCE_DEVIATION


def compute_withdrawal_credentials(spec, profile: dict, withdrawal_credentials) -> bytes:
    assert withdrawal_credentials[:1] == spec.BLS_WITHDRAWAL_PREFIX

    if profile['withdrawal_credential_type'] == 'ETH1':
        return spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b'\x00' * 11 + withdrawal_credentials[12:]

    if profile['withdrawal_credential_type'] == 'COMP':
        return spec.COMPOUNDING_WITHDRAWAL_PREFIX + b'\x00' * 11 + withdrawal_credentials[12:]

    if profile['withdrawal_credential_type'] == 'UNKNOWN':
        return spec.Bytes1("0xfe") + withdrawal_credentials[1:]

    return withdrawal_credentials


def prepare_validator_and_beacon_state(spec, state, profile: dict, validator_index) -> None:
    validator_epochs = compute_validator_epochs(spec, state, profile)

    # validator state and balance
    validator = state.validators[validator_index]
    validator.activation_eligibility_epoch = validator_epochs['activation_eligibility_epoch']
    validator.activation_epoch = validator_epochs['activation_epoch']
    validator.exit_epoch = validator_epochs['exit_epoch']
    validator.withdrawable_epoch = validator_epochs['withdrawable_epoch']
    validator.slashed = profile['slashed']
    validator.effective_balance = compute_effective_balance(spec, profile)
    validator.withdrawal_credentials = compute_withdrawal_credentials(spec, profile, validator.withdrawal_credentials)
    state.balances[validator_index] = compute_balance(spec, profile, validator.effective_balance)

    # advance current_epoch
    current_epoch = compute_current_epoch(spec, state, profile, validator)
    state.slot = spec.Slot(spec.SLOTS_PER_EPOCH * current_epoch)

    # partial withdrawal
    if profile['has_pending_withdrawal_request']:
        assert validator.withdrawal_credentials[:1] in (
            spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX, spec.COMPOUNDING_WITHDRAWAL_PREFIX)

        if profile['excess_balance_gt_pending_withdrawal_balance']:
            # half of the excess is pending withdrawal
            amount = (state.balances[validator_index] - validator.effective_balance) // 2
            assert amount > 0
        else:
            # entire excess balance is pending withdrawal
            amount = spec.EFFECTIVE_BALANCE_INCREMENT // 10

        state.pending_partial_withdrawals.append(spec.PendingPartialWithdrawal(
            validator_index=validator_index,
            amount=amount,
            # Make withdrawable in the current_epoch
            withdrawable_epoch=spec.get_current_epoch(state),
        ))

    # consolidation
    if profile['has_pending_consolidation_request']:
        assert validator.withdrawal_credentials[:1] in (
            spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX, spec.COMPOUNDING_WITHDRAWAL_PREFIX)
        assert validator.exit_epoch != spec.FAR_FUTURE_EPOCH

        target_index = (validator_index + 1) % len(state.validators)
        set_compounding_withdrawal_credential(spec, state, target_index)
        state.pending_consolidations.append(spec.PendingConsolidation(
            source_index=validator_index,
            target_index=target_index,
        ))


def get_validator_state_profile(spec, state, validator_index) -> dict:
    # Set defaults
    profile = {
        'activation_eligibility_epoch_set': False,
        'activation_eligibility_epoch_finalized': False,
        'activation_epoch_set': False,
        'exit_epoch_set': False,
        'withdrawable_epoch_set': False,
        'slashed': False,
        'activation_epoch_gt_activation_eligibility_epoch': False,
        'withdrawable_epoch_gt_exit_epoch': False,
        'shard_committee_period_lte_current_epoch': False,
        'activation_epoch_to_current_epoch': '>',
        'exit_epoch_to_current_epoch': '>',
        'withdrawable_epoch_to_current_epoch': '>',
        'balance_is_zero': False,
        'balance_to_effective_balance': '=',
        'effective_balance_lte_ejection_balance': False,
        'effective_balance_to_min_activation_balance': '=',
        'effective_balance_to_max_effective_balance': '=',
        'withdrawal_credential_type': 'BLS',
        'has_pending_withdrawal_request': False,
        'excess_balance_gt_pending_withdrawal_balance': False,
        'has_pending_consolidation_request': False,
    }

    def get_comparison_op(value1, value2) -> str:
        if value1 < value2:
            return '<'
        elif value1 == value2:
            return '='
        else:
            return '>'

    validator = state.validators[validator_index]
    balance = state.balances[validator_index]
    max_effective_balace = spec.get_max_effective_balance(validator)
    current_epoch = spec.get_current_epoch(state)

    if validator.activation_eligibility_epoch != spec.FAR_FUTURE_EPOCH:
        profile['activation_eligibility_epoch_set'] = True

    if validator.activation_eligibility_epoch <= state.finalized_checkpoint.epoch:
        profile['activation_eligibility_epoch_finalized'] = True

    if validator.activation_epoch != spec.FAR_FUTURE_EPOCH:
        profile['activation_epoch_set'] = True

    if validator.exit_epoch != spec.FAR_FUTURE_EPOCH:
        profile['exit_epoch_set'] = True

    if validator.withdrawable_epoch != spec.FAR_FUTURE_EPOCH:
        profile['withdrawable_epoch_set'] = True

    if validator.slashed:
        profile['slashed'] = True

    if validator.activation_epoch > validator.activation_eligibility_epoch:
        profile['activation_epoch_gt_activation_eligibility_epoch'] = True

    if validator.withdrawable_epoch > validator.exit_epoch:
        profile['withdrawable_epoch_gt_exit_epoch'] = True

    if (validator.activation_epoch != spec.FAR_FUTURE_EPOCH
            and validator.activation_epoch + spec.config.SHARD_COMMITTEE_PERIOD <= current_epoch):
        profile['shard_committee_period_lte_current_epoch'] = True

    profile['activation_epoch_to_current_epoch'] = get_comparison_op(validator.activation_epoch, current_epoch)
    profile['exit_epoch_to_current_epoch'] = get_comparison_op(validator.exit_epoch, current_epoch)
    profile['withdrawable_epoch_to_current_epoch'] = get_comparison_op(validator.withdrawable_epoch, current_epoch)

    if balance == spec.Gwei(0):
        profile['balance_is_zero'] = True

    if validator.effective_balance <= spec.config.EJECTION_BALANCE:
        profile['effective_balance_lte_ejection_balance'] = True

    profile['balance_to_effective_balance'] = get_comparison_op(balance, validator.effective_balance)
    profile['effective_balance_to_min_activation_balance'] = get_comparison_op(
        validator.effective_balance, spec.MIN_ACTIVATION_BALANCE)
    profile['effective_balance_to_max_effective_balance'] = get_comparison_op(
        validator.effective_balance, max_effective_balace)

    if validator.withdrawal_credentials[:1] == spec.BLS_WITHDRAWAL_PREFIX:
        profile['withdrawal_credential_type'] = 'BLS'
    elif validator.withdrawal_credentials[:1] == spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX:
        profile['withdrawal_credential_type'] = 'ETH1'
    elif validator.withdrawal_credentials[:1] == spec.COMPOUNDING_WITHDRAWAL_PREFIX:
        profile['withdrawal_credential_type'] = 'COMP'
    else:
        profile['withdrawal_credential_type'] = 'UNKNOWN'

    if any(w for w in state.pending_partial_withdrawals if w.validator_index == validator_index):
        profile['has_pending_withdrawal_request'] = True

    if any(c for c in state.pending_consolidations if c.source_index == validator_index):
        profile['has_pending_consolidation_request'] = True

    if (profile['has_pending_withdrawal_request']
            and validator.effective_balance >= spec.MIN_ACTIVATION_BALANCE
            and balance > validator.effective_balance
            and spec.get_pending_balance_to_withdraw(state, validator_index) < balance - validator.effective_balance):
        profile['excess_balance_gt_pending_withdrawal_balance'] = True

    return profile


# ---------
# Test runs
# ---------


def run_validator_state_test(spec, state, test_case_fn):
    # skip 4 epochs to finalize epoch > GENESIS_EPOCH
    for _ in range(4):
        next_epoch_with_full_participation(spec, state)
    clean_state = state

    for i, profile in enumerate(validator_state_profiles):
        state = clean_state.copy()

        validator_index = i % len(state.validators)
        prepare_validator_and_beacon_state(spec, state, profile, validator_index)

        # Run test case
        try:
            test_case_fn(spec, state, i, validator_index)
        except Exception as e:
            print(f"Failed profile {i}: {profile}")
            raise e

        # Check that validator is in the correct state after processing
        post_validator_state_profile = get_validator_state_profile(spec, state, validator_index)
        assert post_validator_state_profile in validator_state_profiles, (
            f"Failed profile {i}: pre={profile}, post={post_validator_state_profile}")


def with_all_validator_states(fn):
    def entry(*args, spec, **kw):
        state = kw.pop('state')
        return run_validator_state_test(spec, state, fn)
    return entry
