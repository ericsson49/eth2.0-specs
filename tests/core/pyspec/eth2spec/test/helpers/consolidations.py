from eth2spec.test.helpers.withdrawals import (
    set_eth1_withdrawal_credential_with_balance
)


def prepare_switch_to_compounding_request(spec, state, validator_index, address=None):
    validator = state.validators[validator_index]
    if not spec.has_execution_withdrawal_credential(validator):
        set_eth1_withdrawal_credential_with_balance(spec, state, validator_index, address=address)

    return spec.ConsolidationRequest(
        source_address=state.validators[validator_index].withdrawal_credentials[12:],
        source_pubkey=state.validators[validator_index].pubkey,
        target_pubkey=state.validators[validator_index].pubkey,
    )


def is_valid_consolidation_request(spec, state, consolidation_request) -> bool:
    # Verify that source != target, so a consolidation cannot be used as an exit.
    if consolidation_request.source_pubkey == consolidation_request.target_pubkey:
        return False
    # If the pending consolidations queue is full, consolidation requests are ignored
    if len(state.pending_consolidations) == spec.PENDING_CONSOLIDATIONS_LIMIT:
        return False
    # If there is too little available consolidation churn limit, consolidation requests are ignored
    if spec.get_consolidation_churn_limit(state) <= spec.MIN_ACTIVATION_BALANCE:
        return False

    validator_pubkeys = [v.pubkey for v in state.validators]
    # Verify pubkeys exists
    request_source_pubkey = consolidation_request.source_pubkey
    request_target_pubkey = consolidation_request.target_pubkey
    if request_source_pubkey not in validator_pubkeys:
        return False
    if request_target_pubkey not in validator_pubkeys:
        return False
    source_index = spec.ValidatorIndex(validator_pubkeys.index(request_source_pubkey))
    target_index = spec.ValidatorIndex(validator_pubkeys.index(request_target_pubkey))
    source_validator = state.validators[source_index]
    target_validator = state.validators[target_index]

    # Verify source withdrawal credentials
    has_correct_credential = spec.has_execution_withdrawal_credential(source_validator)
    is_correct_source_address = (
        source_validator.withdrawal_credentials[12:] == consolidation_request.source_address
    )
    if not (has_correct_credential and is_correct_source_address):
        return False

    # Verify that target has compounding withdrawal credentials
    if not spec.has_compounding_withdrawal_credential(target_validator):
        return False

    # Verify the source and the target are active
    current_epoch = spec.get_current_epoch(state)
    if not spec.is_active_validator(source_validator, current_epoch):
        return False
    if not spec.is_active_validator(target_validator, current_epoch):
        return False
    # Verify exits for source and target have not been initiated
    if source_validator.exit_epoch != spec.FAR_FUTURE_EPOCH:
        return False
    if target_validator.exit_epoch != spec.FAR_FUTURE_EPOCH:
        return False
    # Verify the source has been active long enough
    if current_epoch < source_validator.activation_epoch + spec.config.SHARD_COMMITTEE_PERIOD:
        return False
    # Verify the source has no pending withdrawals in the queue
    if spec.get_pending_balance_to_withdraw(state, source_index) > 0:
        return False

    return True
