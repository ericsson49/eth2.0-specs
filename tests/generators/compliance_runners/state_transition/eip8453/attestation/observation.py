"""Read finality-gadget inputs without invoking acceptance/processing helpers."""

from tests.generators.compliance_runners.state_transition.eip8453.operation_materializer import (
    verify,
)


def participants(s, pre, operation):
    active = [
        i
        for i, v in enumerate(pre.validators)
        if v.activation_epoch <= operation.epoch < v.exit_epoch
    ]
    selected = []
    for c, bit in enumerate(operation.committee_bits):
        if bit:
            start = len(active) * c // int(s.ATTESTATION_COMMITTEES_PER_ROUND)
            end = len(active) * (c + 1) // int(s.ATTESTATION_COMMITTEES_PER_ROUND)
            selected.extend(
                active[(p + int(operation.epoch)) % len(active)] for p in range(start, end)
            )
    return selected


def observe_attributes(ctx):
    s, pre, op = ctx.spec, ctx.pre, ctx.operation
    data = op.data
    committee = participants(s, pre, op)
    indices = [v for v, bit in zip(committee, op.aggregation_bits, strict=False) if bit]
    if (
        not data.target.height
        or data.target.root == s.Root()
        or (
            data.finality != s.HeightCheckpoint.empty()
            and (data.finality.root == s.Root() or data.finality.height >= data.target.height)
        )
    ):
        case = "MALFORMED"
    elif op.epoch > int(pre.slot) // int(s.SLOTS_PER_EPOCH):
        case = "FUTURE"
    elif op.epoch < int(pre.finalized_slot) // int(s.SLOTS_PER_EPOCH):
        case = "STALE"
    elif len(op.aggregation_bits) != len(committee):
        case = "BITS"
    elif not indices:
        case = "EMPTY"
    elif not verify(
        s, pre, data, s.DOMAIN_FINALITY_ATTESTER, op.epoch, sorted(set(indices)), op.signature
    ):
        case = "SIGNATURE"
    elif data.target == pre.target_checkpoint:
        case = "NO_FINALITY" if data.finality == s.HeightCheckpoint.empty() else "CURRENT"
    elif any(p.checkpoint == data.finality for p in pre.pending_finality) and (
        data.finality.height < data.target.height < pre.target_checkpoint.height
    ):
        case = "OLD"
    else:
        case = "UNKNOWN"
    return {
        "observed_case": case,
        "observed_timeout": bool(data.timeout),
        "observed_progress_recorded": bool(pre.progress_participation[0]),
        "observed_finality_recorded": bool(pre.pending_finality[0].participation[0]),
    }
