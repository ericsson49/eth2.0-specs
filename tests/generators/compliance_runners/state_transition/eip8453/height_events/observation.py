"""Recover height-event inputs without calling the handler or its quorum helpers."""


def observe_attributes(ctx):
    spec, state = ctx.spec, ctx.pre
    epoch = int(state.slot) // int(spec.SLOTS_PER_EPOCH)
    active = [
        i
        for i, validator in enumerate(state.validators)
        if int(validator.activation_epoch) <= epoch < int(validator.exit_epoch)
    ]
    cap = int(spec.EJECTION_INACTIVITY_SCORE)
    weights = {i: cap * cap - min(int(state.inactivity_scores[i]), cap) ** 2 for i in active}
    total = sum(weights.values())

    def participating(bits):
        return sum(
            weight for i, weight in weights.items() if bits[i] and not state.validators[i].slashed
        )

    pending_weights = [participating(p.participation) for p in state.pending_finality]
    qualifying = [
        p
        for p, weight in zip(state.pending_finality, pending_weights, strict=True)
        if weight > 0 and weight * 3 >= total * 2
    ]
    # The processor selects the last qualifying entry in list order.
    finalized_height = int(
        qualifying[-1].checkpoint.height if qualifying else state.finalized_checkpoint.height
    )
    return {
        "target_weight": participating(state.target_participation),
        "progress_weight": participating(state.progress_participation),
        "total_weight": total,
        "timeout_delta": int(state.slot) - int(state.target_slot) - int(spec.TIMEOUT_DELAY_SLOTS),
        "pending_count": len(state.pending_finality),
        "qualifying_count": len(qualifying),
        "retained_count": sum(
            int(p.checkpoint.height) > finalized_height for p in state.pending_finality
        ),
    }
