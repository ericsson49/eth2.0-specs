"""Observe validator zero and its reward context without invoking spec helpers."""


def observe_attributes(ctx):
    spec, state = ctx.spec, ctx.pre
    if not state.validators:
        raise ValueError("rewards target requires validator zero")
    epoch = int(state.slot) // int(spec.SLOTS_PER_EPOCH)
    active = [int(v.activation_epoch) <= epoch < int(v.exit_epoch) for v in state.validators]
    validator = state.validators[0]
    flags = int(state.round_participation[0])
    progress = 1 << int(spec.PROGRESS_FLAG_INDEX)
    finality = 1 << int(spec.FINALITY_FLAG_INDEX)
    previous_round = max(0, int(state.slot) // int(spec.SLOTS_PER_ROUND) - 1)
    delay = max(0, previous_round - int(state.finalized_slot) // int(spec.SLOTS_PER_ROUND))

    def other_participants(mask):
        return any(
            active[i] and not v.slashed and int(state.round_participation[i]) & mask
            for i, v in enumerate(state.validators)
            if i != 0
        )

    return {
        "active": active[0],
        "slashed": bool(validator.slashed),
        "progress": bool(flags & progress),
        "finality": bool(flags & finality),
        "increments": int(validator.effective_balance) // int(spec.EFFECTIVE_BALANCE_INCREMENT),
        "other_progress": other_participants(progress),
        "other_finality": other_participants(finality),
        "finality_delay": delay,
    }
