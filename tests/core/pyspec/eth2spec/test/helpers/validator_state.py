from dataclasses import dataclass
from eth2spec.test.helpers.validator_state_profiles import (
    ValidatorStateProfile,
    validator_state_profiles,
)
from eth2spec.test.helpers.state import (
    next_epoch_with_full_participation,
)


@dataclass
class ValidatorAndBeaconState:
    # state.validators
    withdrawal_credential_type: str
    activation_eligibility_epoch: int
    activation_epoch: int
    exit_epoch: int
    withdrawable_epoch: int
    slashed: bool
    effective_balance: int
    # balance
    balance: int
    # operations
    has_pending_withdrawal_request: bool
    has_pending_consolidation_request: bool
    # current_epoch
    current_epoch: int


def compute_epochs(spec, state, profile: dict) -> tuple[int, int, int, int, int]:
    if profile['activation_eligibility_epoch_finalized']:
        activation_eligibility_epoch = state.finalized_checkpoint.epoch
    elif profile['activation_eligibility_epoch_set']:
        activation_eligibility_epoch = state.finalized_checkpoint.epoch + 1
    else:
        activation_eligibility_epoch = spec.FAR_FUTURE_EPOCH

    if profile['activation_epoch_set']:
        # 2 epochs to finalize
        activation_epoch = spec.compute_activation_exit_epoch(activation_eligibility_epoch + 2)
    else:
        activation_epoch = spec.FAR_FUTURE_EPOCH

    if profile['exit_epoch_set']:
        if activation_epoch != spec.FAR_FUTURE_EPOCH:
            exit_epoch = spec.compute_activation_exit_epoch(activation_epoch + spec.config.SHARD_COMMITTEE_PERIOD + 1)
        elif activation_eligibility_epoch <= state.finalized_checkpoint.epoch:
            exit_epoch = spec.compute_activation_exit_epoch(activation_eligibility_epoch + 2)
        elif activation_eligibility_epoch != spec.FAR_FUTURE_EPOCH:
            exit_epoch = spec.compute_activation_exit_epoch(activation_eligibility_epoch)
        else:
            exit_epoch = spec.compute_activation_exit_epoch(spec.get_current_epoch(state))
    else:
        exit_epoch = spec.FAR_FUTURE_EPOCH

    if profile['withdrawable_epoch_set']:
        if profile['slashed']:
            withdrawable_epoch = exit_epoch + spec.EPOCHS_PER_SLASHINGS_VECTOR
        else:
            withdrawable_epoch = exit_epoch + spec.config.MIN_VALIDATOR_WITHDRAWABILITY_DELAY
    else:
        withdrawable_epoch = spec.FAR_FUTURE_EPOCH

    if profile['withdrawable_epoch_set'] and profile['withdrawable_epoch_to_current_epoch'] != 'LT':
        if profile['withdrawable_epoch_to_current_epoch'] == 'EQ':
            current_epoch = withdrawable_epoch
        else:
            current_epoch = withdrawable_epoch + 1
    elif profile['exit_epoch_set'] and profile['exit_epoch_to_current_epoch'] != 'LT':
        if profile['exit_epoch_to_current_epoch'] == 'EQ':
            current_epoch = exit_epoch
        else:
            current_epoch = exit_epoch + 1
    elif profile['activation_epoch_set'] and profile['activation_epoch_to_current_epoch'] != 'LT':
        if profile['activation_epoch_to_current_epoch'] == 'EQ':
            current_epoch = activation_epoch
        else:
            current_epoch = activation_epoch + 1
    elif profile['activation_eligibility_epoch_finalized']:
        current_epoch = activation_eligibility_epoch + 2
    elif profile['activation_eligibility_epoch_set']:
        current_epoch = activation_eligibility_epoch
    else:
        current_epoch = spec.get_current_epoch(state)

    return (
        activation_eligibility_epoch,
        activation_epoch,
        exit_epoch,
        withdrawable_epoch,
        current_epoch
    )


def compute_balances(spec, profile: dict) -> tuple[int, int]:
    if profile['balance_is_zero'] and profile['balance_to_effective_balance'] == 'EQ':
        return (0, 0)

    if profile['effective_balance_lte_ejection_balance']:
        effective_balance = spec.config.EJECTION_BALANCE
    elif profile['effective_balance_to_min_activation_balance'] == 'LT':
        effective_balance = spec.MIN_ACTIVATION_BALANCE - spec.EFFECTIVE_BALANCE_INCREMENT
    elif profile['effective_balance_to_min_activation_balance'] == 'EQ':
        effective_balance = spec.MIN_ACTIVATION_BALANCE
    elif profile['effective_balance_to_max_effective_balance'] == 'EQ':
        if profile['withdrawal_credential_type'] == 'COMP':
            effective_balance = spec.MAX_EFFECTIVE_BALANCE_ELECTRA
        else:
            effective_balance = spec.MIN_ACTIVATION_BALANCE
    else:
        effective_balance = spec.MIN_ACTIVATION_BALANCE + spec.EFFECTIVE_BALANCE_INCREMENT

    if profile['balance_is_zero']:
        balance = 0
    elif profile['balance_to_effective_balance'] == 'LT':
        balance = effective_balance - spec.EFFECTIVE_BALANCE_INCREMENT // 5
    elif profile['balance_to_effective_balance'] == 'EQ':
        balance = effective_balance
    else:
        balance = effective_balance + spec.EFFECTIVE_BALANCE_INCREMENT // 5

    return (balance, effective_balance)


def compute_validator_and_beacon_state(spec, state, profile: dict) -> ValidatorAndBeaconState:
    epochs = compute_epochs(spec, state, profile)
    balance, effective_balance = compute_balances(spec, profile)

    return ValidatorAndBeaconState(
        withdrawal_credential_type=profile['withdrawal_credential_type'],
        activation_eligibility_epoch=epochs[0],
        activation_epoch=epochs[1],
        exit_epoch=epochs[2],
        withdrawable_epoch=epochs[3],
        current_epoch=epochs[4],
        balance=balance,
        effective_balance=effective_balance,
        slashed=profile['slashed'],
        has_pending_withdrawal_request=profile['has_pending_withdrawal_request'],
        has_pending_consolidation_request=profile['has_pending_consolidation_request'],
    )


def prepare_validator_and_beacon_state(spec, state, profile: dict, validator_index):
    # skip 4 epochs to finalize a new epoch
    next_epoch_with_full_participation(spec, state)
    next_epoch_with_full_participation(spec, state)
    next_epoch_with_full_participation(spec, state)
    next_epoch_with_full_participation(spec, state)

    validator_and_beacon_state = compute_validator_and_beacon_state(spec, state, profile)

    # validator state and balance
    validator = state.validators[validator_index]
    validator.activation_eligibility_epoch = spec.Epoch(validator_and_beacon_state.activation_eligibility_epoch)
    validator.activation_epoch = spec.Epoch(validator_and_beacon_state.activation_epoch)
    validator.exit_epoch = spec.Epoch(validator_and_beacon_state.exit_epoch)
    validator.withdrawable_epoch = spec.Epoch(validator_and_beacon_state.withdrawable_epoch)
    validator.slashed = validator_and_beacon_state.slashed
    validator.effective_balance = spec.Gwei(validator_and_beacon_state.effective_balance)
    state.balances[validator_index] = spec.Gwei(validator_and_beacon_state.balance)

    # withdrawal credential
    if validator_and_beacon_state.withdrawal_credential_type == 'ETH1':
        validator.withdrawal_credentials = (
            spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX + b'\x00' * 11 + validator.withdrawal_credentials[12:])
    elif validator_and_beacon_state.withdrawal_credential_type == 'COMP':
        validator.withdrawal_credentials = (
            spec.COMPOUNDING_WITHDRAWAL_PREFIX + b'\x00' * 11 + validator.withdrawal_credentials[12:])
    elif validator_and_beacon_state.withdrawal_credential_type == 'UNKNOWN':
        validator.withdrawal_credentials = spec.Bytes1("0xfe") + validator.withdrawal_credentials[1:]

    # advance current_epoch
    state.slot = spec.Slot(spec.SLOTS_PER_EPOCH * validator_and_beacon_state.current_epoch)

    # operations
    
