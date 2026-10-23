"""Recover vote timing, roots, seats and signature without running the handler."""

from eth_consensus_specs.utils import bls
from tests.generators.compliance_runners.state_transition.eip8453.operation_materializer import (
    verify,
)


def observe_attributes(ctx):
    s, pre, vote = ctx.spec, ctx.pre, ctx.operation
    slot, n = int(vote.data.slot), int(s.SLOTS_PER_EPOCH)
    epoch, current = slot // n, int(pre.slot) // n
    timing = (
        "TOO_EARLY"
        if slot + int(s.MIN_ATTESTATION_INCLUSION_DELAY) > int(pre.slot)
        else "GENESIS"
        if slot == 0
        else "CURRENT"
        if epoch == current
        else "PREVIOUS"
        if epoch == current - 1
        else "TOO_OLD"
    )
    root = pre.block_roots[slot % s.SLOTS_PER_HISTORICAL_ROOT]
    previous = pre.block_roots[(slot - 1) % s.SLOTS_PER_HISTORICAL_ROOT]
    seats = [i for i, bit in enumerate(vote.aggregation_bits) if bit]
    committee = s.get_head_vote_committee(pre, vote.data.slot) if seats else []
    valid = (
        verify(
            s,
            pre,
            vote.data,
            s.DOMAIN_HEAD_VOTE,
            epoch,
            [committee[i] for i in seats],
            vote.signature,
        )
        if seats
        else bytes(vote.signature) == bls.G2_POINT_AT_INFINITY
    )
    index = (n if epoch == current else 0) + slot % n
    return {
        "observed_timing": timing,
        "observed_root_relation": "OTHER"
        if vote.data.beacon_block_root != root
        else "SKIPPED"
        if root == previous
        else "SAME_SLOT",
        "observed_voters": "EMPTY" if not seats else "ONE" if len(seats) == 1 else "MULTIPLE",
        "observed_already_recorded": bool(pre.head_vote_participation[index][0]),
        "observed_signature_valid": valid,
    }
