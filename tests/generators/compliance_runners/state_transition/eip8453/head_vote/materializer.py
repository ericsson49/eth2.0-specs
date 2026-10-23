"""Real signed votes, including duplicate committee seats."""

from eth_consensus_specs.utils import bls
from tests.generators.compliance_runners.state_transition.eip8453.operation_materializer import (
    base_state,
    OperationMaterializer,
    sign,
)

from .target import TARGET


class HeadVoteMaterializer(OperationMaterializer):
    handler_name = input_name = "head_vote"
    test_provider = "eip8453_head_vote"
    target = TARGET

    def build_inputs(self, solution):
        s = self.spec
        pre = base_state(s)
        n = int(s.SLOTS_PER_EPOCH)
        if solution.timing == "GENESIS":
            pre.slot = s.Slot(1)
        slot = {
            "GENESIS": 0,
            "PREVIOUS": 3 * n - 1,
            "CURRENT": 3 * n + 1,
            "TOO_OLD": n,
            "TOO_EARLY": int(pre.slot),
        }[solution.timing]
        root = b"\x01" * 32
        # Even rejected same-slot votes have a concrete history entry for observation.
        pre.block_roots[slot % s.SLOTS_PER_HISTORICAL_ROOT] = root
        if slot:
            pre.block_roots[(slot - 1) % s.SLOTS_PER_HISTORICAL_ROOT] = (
                root if solution.root_relation == "SKIPPED" else b"\x02" * 32
            )
        else:
            # Genesis bypasses root comparison; retain a separate sentinel for root coverage.
            pre.block_roots[-1] = root if solution.root_relation == "SKIPPED" else b"\x02" * 32
        vote = s.HeadVote(
            data=s.HeadVoteData(
                slot=slot,
                beacon_block_root=b"\x03" * 32 if solution.root_relation == "OTHER" else root,
            )
        )
        count = {"EMPTY": 0, "ONE": 1, "MULTIPLE": 3}[solution.voters]
        epoch = slot // n
        index = (n if epoch == int(pre.slot) // n else 0) + slot % n
        if solution.already_recorded:
            pre.head_vote_participation[index][0] = True
        if count:
            committee = s.get_head_vote_committee(pre, s.Slot(slot))
            positions = [0]
            if count > 1:
                # Guaranteed duplicate validator, but distinct seats; deduplicating
                # signature participants or participation bits would be incorrect.
                first, repeated = next(
                    (i, j)
                    for j in range(1, len(committee))
                    for i in range(j)
                    if committee[i] == committee[j]
                )
                positions = [first]
                positions.extend(
                    [repeated, next(i for i in range(len(committee)) if i not in (first, repeated))]
                )
            for i in positions:
                vote.aggregation_bits[i] = True
            vote.signature = sign(
                s, pre, vote.data, s.DOMAIN_HEAD_VOTE, epoch, [committee[i] for i in positions]
            )
        elif solution.signature_valid:
            # No participants is separately rejected; a well-formed empty aggregate sentinel.
            vote.signature = bls.G2_POINT_AT_INFINITY
        if not solution.signature_valid:
            vote.signature = s.BLSSignature()
        return pre, vote


MATERIALIZER = HeadVoteMaterializer
