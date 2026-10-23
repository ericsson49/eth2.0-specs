"""Recover the first deposit and its suffix without executing deposit processing."""


def observe_attributes(ctx):
    s, pre = ctx.spec, ctx.pre
    if not pre.pending_deposits:
        raise ValueError("pending-deposit target requires a first deposit")
    deposit = pre.pending_deposits[0]
    validator = next(v for v in pre.validators if v.pubkey == deposit.pubkey)
    next_epoch = int(pre.slot) // int(s.SLOTS_PER_EPOCH) + 1
    role = (
        "WITHDRAWN"
        if int(validator.withdrawable_epoch) < next_epoch
        else "EXITING"
        if int(validator.exit_epoch) < int(s.FAR_FUTURE_EPOCH)
        else "ACTIVE"
    )
    offset = int(pre.finalized_slot) % int(s.SLOTS_PER_EPOCH)
    return {
        "deposit_slot": int(deposit.slot),
        "finalized_slot": int(pre.finalized_slot),
        "position": "START"
        if offset == 0
        else "END"
        if offset == int(s.SLOTS_PER_EPOCH) - 1
        else "INTERIOR",
        "role": role,
        "amount": int(deposit.amount),
        # Churn calculation is inherited and outside this target's coverage scope.
        "available": int(pre.deposit_balance_to_consume) + int(s.get_activation_churn_limit(pre)),
        "carry": bool(pre.deposit_balance_to_consume),
        "observed_suffix": "NONE"
        if len(pre.pending_deposits) == 1
        else "UNFINALIZED"
        if pre.pending_deposits[1].slot > pre.finalized_slot
        else "FINALIZED",
    }
