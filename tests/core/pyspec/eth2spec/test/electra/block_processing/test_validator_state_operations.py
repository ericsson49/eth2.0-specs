from eth2spec.test.helpers.constants import (
    MINIMAL,
)
from eth2spec.test.context import (
    spec_state_test,
    with_electra_and_later,
    with_presets,
    single_phase,
    spec_test,
    with_custom_state,
    scaled_churn_balances_exceed_activation_exit_churn_limit,
    default_activation_threshold,
    expect_assertion_error,
    never_bls,
)
from eth2spec.test.helpers.validator_state import (
    with_all_validator_states,
)
from eth2spec.test.helpers.withdrawals import (
    is_valid_withdrawal_request,
    set_compounding_withdrawal_credential,
)
from eth2spec.test.helpers.consolidations import is_valid_consolidation_request
from eth2spec.test.helpers.bls_to_execution_changes import get_signed_address_change
from eth2spec.test.helpers.keys import privkeys
from eth2spec.test.helpers.voluntary_exits import (
    sign_voluntary_exit,
    is_valid_voluntary_exit,
)
from eth2spec.test.helpers.attester_slashings import get_valid_attester_slashing_by_indices
from eth2spec.test.helpers.proposer_slashings import get_valid_proposer_slashing
from eth2spec.test.helpers.execution_payload import build_empty_execution_payload


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], reason="too many tests")
@with_all_validator_states
def test_vs_process_partial_withdrawal_request(spec, state, profile_index, validator_index):
    # Create a partial withdrawal request
    validator = state.validators[validator_index]
    withdrawal_request = spec.WithdrawalRequest(
        source_address=validator.withdrawal_credentials[12:],
        validator_pubkey=validator.pubkey,
        amount=spec.Gwei(1),  # any excess balance works
    )

    # Run request processing
    pre_len_pending_partial_withdrawals = len(state.pending_partial_withdrawals)
    success = is_valid_withdrawal_request(spec, state, withdrawal_request)
    spec.process_withdrawal_request(state, withdrawal_request)

    # Check the operation is processed
    if success:
        assert len(state.pending_partial_withdrawals) == pre_len_pending_partial_withdrawals + 1
    else:
        assert len(state.pending_partial_withdrawals) == pre_len_pending_partial_withdrawals


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], reason="too many tests")
@with_all_validator_states
def test_vs_process_full_exit_request(spec, state, profile_index, validator_index):
    # Create a full exit
    validator = state.validators[validator_index]
    withdrawal_request = spec.WithdrawalRequest(
        source_address=validator.withdrawal_credentials[12:],
        validator_pubkey=validator.pubkey,
        amount=spec.Gwei(0),  # full exit request
    )

    # Run request processing
    pre_exit_epoch = state.validators[validator_index].exit_epoch
    success = is_valid_withdrawal_request(spec, state, withdrawal_request)
    spec.process_withdrawal_request(state, withdrawal_request)

    # Check the operation is processed
    if success:
        assert state.validators[validator_index].exit_epoch != spec.FAR_FUTURE_EPOCH
    else:
        assert state.validators[validator_index].exit_epoch == pre_exit_epoch


@with_electra_and_later
@with_presets([MINIMAL], "need sufficient consolidation churn limit")
@with_custom_state(
    balances_fn=scaled_churn_balances_exceed_activation_exit_churn_limit,
    threshold_fn=default_activation_threshold,
)
@spec_test
@single_phase
@with_all_validator_states
def test_vs_process_consolidation_request(spec, state, profile_index, validator_index):
    # Create a consolidation request
    validator = state.validators[validator_index]
    target_index = (validator_index + 1) % len(state.validators)
    consolidation_request = spec.ConsolidationRequest(
        source_address=validator.withdrawal_credentials[12:],
        source_pubkey=state.validators[validator_index].pubkey,
        target_pubkey=state.validators[target_index].pubkey,
    )

    # Set target to compounding credentials
    set_compounding_withdrawal_credential(spec, state, target_index)

    # Run request processing
    pre_len_pending_consolidations = len(state.pending_consolidations)
    success = is_valid_consolidation_request(spec, state, consolidation_request)
    spec.process_consolidation_request(state, consolidation_request)

    # Check the operation is processed
    if success:
        assert len(state.pending_consolidations) == pre_len_pending_consolidations + 1
    else:
        assert len(state.pending_consolidations) == pre_len_pending_consolidations


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], reason="too many tests")
@with_all_validator_states
def test_vs_process_switch_to_compounding_request(spec, state, profile_index, validator_index):
    # Create a switch to compounding request
    validator = state.validators[validator_index]
    consolidation_request = spec.ConsolidationRequest(
        source_address=validator.withdrawal_credentials[12:],
        source_pubkey=state.validators[validator_index].pubkey,
        target_pubkey=state.validators[validator_index].pubkey,
    )

    # Run request processing
    success = spec.is_valid_switch_to_compounding_request(state, consolidation_request)
    pre_withdrawal_credentials_type = state.validators[validator_index].withdrawal_credentials[:1]
    spec.process_consolidation_request(state, consolidation_request)

    # Check the operation is processed
    if success:
        assert state.validators[validator_index].withdrawal_credentials[:1] == spec.COMPOUNDING_WITHDRAWAL_PREFIX
    else:
        assert state.validators[validator_index].withdrawal_credentials[:1] == pre_withdrawal_credentials_type


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], reason="too many tests")
@with_all_validator_states
def test_vs_process_top_up(spec, state, profile_index, validator_index):
    # Create a top-up request
    validator = state.validators[validator_index]
    deposit_request = spec.DepositRequest(
        pubkey=validator.pubkey,
        withdrawal_credentials=validator.withdrawal_credentials,
        amount=spec.EFFECTIVE_BALANCE_INCREMENT // 10,
        signature=spec.bls.G2_POINT_AT_INFINITY,
        index=state.eth1_deposit_index
    )

    # Run request processing
    spec.process_deposit_request(state, deposit_request)

    # Check the operation is processed
    pending_deposit = state.pending_deposits[len(state.pending_deposits) - 1]
    assert pending_deposit.pubkey == validator.pubkey
    assert pending_deposit.amount == deposit_request.amount


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], reason="too many tests")
@with_all_validator_states
def test_vs_process_bls_to_execution_change(spec, state, profile_index, validator_index):
    # Create a valid message
    signed_change = get_signed_address_change(spec, state, validator_index)

    # Run processing
    validator = state.validators[validator_index]
    if validator.withdrawal_credentials[:1] == spec.BLS_WITHDRAWAL_PREFIX:
        spec.process_bls_to_execution_change(state, signed_change)
        post_withdrawal_credentials = state.validators[validator_index].withdrawal_credentials
        assert post_withdrawal_credentials == (
            spec.ETH1_ADDRESS_WITHDRAWAL_PREFIX
            + b'\x00' * 11
            + signed_change.message.to_execution_address
        )
    else:
        expect_assertion_error(lambda: spec.process_bls_to_execution_change(state, signed_change))


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], reason="too many tests")
@with_all_validator_states
@never_bls
def test_vs_process_voluntary_exit(spec, state, profile_index, validator_index):
    # Create a valid message
    voluntary_exit = spec.VoluntaryExit(
        epoch=spec.get_current_epoch(state),
        validator_index=validator_index,
    )
    signed_voluntary_exit = sign_voluntary_exit(spec, state, voluntary_exit, privkeys[validator_index])

    # Run processing
    if is_valid_voluntary_exit(spec, state, voluntary_exit):
        spec.process_voluntary_exit(state, signed_voluntary_exit)
        assert state.validators[validator_index].exit_epoch < spec.FAR_FUTURE_EPOCH
    else:
        expect_assertion_error(lambda: spec.process_voluntary_exit(state, signed_voluntary_exit))


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], reason="too many tests")
@with_all_validator_states
@never_bls
def test_vs_process_attester_slashing(spec, state, profile_index, validator_index):
    # Create a valid message
    attester_slashing = get_valid_attester_slashing_by_indices(spec, state, [validator_index])

    # Run processing
    if spec.is_slashable_validator(state.validators[validator_index], spec.get_current_epoch(state)):
        spec.process_attester_slashing(state, attester_slashing)
        assert state.validators[validator_index].slashed
    else:
        expect_assertion_error(lambda: spec.process_attester_slashing(state, attester_slashing))


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], reason="too many tests")
@with_all_validator_states
@never_bls
def test_vs_process_proposer_slashing(spec, state, profile_index, validator_index):
    # Create a valid message
    proposer_slashing = get_valid_proposer_slashing(spec, state, slashed_index=validator_index)

    # Run processing
    if spec.is_slashable_validator(state.validators[validator_index], spec.get_current_epoch(state)):
        spec.process_proposer_slashing(state, proposer_slashing)
        assert state.validators[validator_index].slashed
    else:
        expect_assertion_error(lambda: spec.process_proposer_slashing(state, proposer_slashing))


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], reason="too many tests")
@with_all_validator_states
def test_vs_process_withdrawals(spec, state, profile_index, validator_index):
    # Set next_withdrawal_validator_index and build a payload
    state.next_withdrawal_validator_index = validator_index
    execution_payload = build_empty_execution_payload(spec, state)

    # Read predicates from pre-state
    validator = state.validators[validator_index]
    balance = state.balances[validator_index]
    is_fully_withdrawable = spec.is_fully_withdrawable_validator(
        validator, balance, spec.get_current_epoch(state)
    )
    is_partially_withdrawable = spec.is_partially_withdrawable_validator(validator, balance)
    pending_balance_to_withdraw = spec.get_pending_balance_to_withdraw(state, validator_index)
    pre_next_withdrawal_index = state.next_withdrawal_index
    if balance >= spec.MIN_ACTIVATION_BALANCE and balance > validator.effective_balance:
        excess = balance - validator.effective_balance
    else:
        excess = 0

    # Run request processing
    spec.process_withdrawals(state, execution_payload)

    # Check withdrawals processing
    index = 0
    if is_fully_withdrawable:
        assert execution_payload.withdrawals[index] == spec.Withdrawal(
            index=pre_next_withdrawal_index + index,
            validator_index=validator_index,
            address=validator.withdrawal_credentials[12:],
            amount=balance,
        )
        index += 1
    else:
        if (pending_balance_to_withdraw > 0
                and validator.exit_epoch == spec.FAR_FUTURE_EPOCH
                and excess > 0):
            assert execution_payload.withdrawals[index] == spec.Withdrawal(
                index=pre_next_withdrawal_index + index,
                validator_index=validator_index,
                address=validator.withdrawal_credentials[12:],
                amount=min(excess, pending_balance_to_withdraw),
            )
            excess = excess - min(excess, pending_balance_to_withdraw)
            index += 1

        if is_partially_withdrawable and excess > 0:
            assert execution_payload.withdrawals[index] == spec.Withdrawal(
                index=(pre_next_withdrawal_index + index),
                validator_index=validator_index,
                address=validator.withdrawal_credentials[12:],
                amount=excess,
            )
            index += 1
