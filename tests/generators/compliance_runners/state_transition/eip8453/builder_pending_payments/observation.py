"""Read payment slot zero, the tail slot, and both rotation buffers."""


def observe_attributes(ctx):
    s, pre = ctx.spec, ctx.pre
    n = int(s.SLOTS_PER_EPOCH)
    return {
        "amount": bool(pre.builder_pending_payments[0].withdrawal.amount),
        "seats": sum(bool(x) for x in pre.head_vote_participation[0]),
        "legacy_weight": bool(pre.builder_pending_payments[0].weight),
        "observed_tail_payment": bool(pre.builder_pending_payments[n - 1].withdrawal.amount),
        "observed_next_payments": any(
            p.withdrawal.amount for p in pre.builder_pending_payments[n:]
        ),
        "observed_next_votes": any(any(bits) for bits in pre.head_vote_participation[n:]),
        "observed_existing_withdrawals": bool(pre.builder_pending_withdrawals),
    }
