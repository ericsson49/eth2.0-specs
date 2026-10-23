"""Block bodies exercising new guards and actual operation dispatch."""

from types import SimpleNamespace

from tests.generators.compliance_runners.state_transition.eip8453.attestation.materializer import (
    AttestationMaterializer,
)
from tests.generators.compliance_runners.state_transition.eip8453.attester_slashing.materializer import (
    Materializer as SlashingMaterializer,
)
from tests.generators.compliance_runners.state_transition.eip8453.head_vote.materializer import (
    HeadVoteMaterializer,
)
from tests.generators.compliance_runners.state_transition.eip8453.legacy_attester_slashing.materializer import (
    Materializer as LegacyMaterializer,
)
from tests.generators.compliance_runners.state_transition.eip8453.operation_materializer import (
    base_state,
    OperationMaterializer,
    real_bls,
    sign,
)

from .target import TARGET


class OperationsMaterializer(OperationMaterializer):
    runner_name = "sanity"
    handler_name = "blocks"
    input_name = "body"
    test_provider = "eip8453_operations"
    target = TARGET

    def build_inputs(self, solution):
        s, case = self.spec, solution.case
        pre, body = base_state(s), s.BeaconBlockBody()
        if case in ("ATTESTATION", "MIXED"):
            pre, att = AttestationMaterializer(s).build_inputs(
                SimpleNamespace(
                    case="CURRENT", timeout=False, progress_recorded=False, finality_recorded=False
                )
            )
            body.attestations.append(att)
        if case in (
            "HEAD",
            "OLD_SUPPORTED",
            "OLD_UNSUPPORTED",
            "SHORT_SUPPORT",
            "LONG_SUPPORT",
            "MAX_HEAD",
            "OVER_HEAD",
            "MIXED",
        ):
            head_pre, vote = HeadVoteMaterializer(s).build_inputs(
                SimpleNamespace(
                    timing="PREVIOUS" if case.startswith("OLD_") else "CURRENT",
                    root_relation="SAME_SLOT",
                    voters="MULTIPLE",
                    already_recorded=False,
                    signature_valid=True,
                )
            )
            pre.block_roots = head_pre.block_roots
            count = (
                int(s.MAX_HEAD_VOTES) + (case == "OVER_HEAD")
                if case in ("MAX_HEAD", "OVER_HEAD")
                else 1
            )
            body.head_votes = s.HeadVotes(data=[vote.copy() for _ in range(count)])
            support_count = count + (case == "LONG_SUPPORT") - (case == "SHORT_SUPPORT")
            body.head_vote_support_bits = s.HeadVoteSupportBits(
                data=[case != "OLD_UNSUPPORTED"] * support_count
            )
        if case in ("LEGACY", "OVER_LEGACY", "MIXED", "FINALITY_SLASHING"):
            cls = SlashingMaterializer if case == "FINALITY_SLASHING" else LegacyMaterializer
            _, slashing = cls(s).build_inputs(
                SimpleNamespace(kind="DOUBLE", fault="NONE", multiple=True)
            )
            if case == "FINALITY_SLASHING":
                body.attester_slashings.append(slashing)
            else:
                count = int(s.MAX_ATTESTER_SLASHINGS_ELECTRA) + 1 if case == "OVER_LEGACY" else 1
                body.legacy_attester_slashings = s.LegacyAttesterSlashings(
                    data=[slashing.copy() for _ in range(count)]
                )
        return pre, body

    def build_block_inputs(self, solution):
        """Wrap the operation scenario in a block with a valid surrounding transition."""
        s = self.spec
        pre, body = self.build_inputs(solution)
        slot = pre.slot
        pre.slot = s.Slot(int(slot) - 1)
        pre.latest_block_header = s.BeaconBlockHeader(slot=pre.slot)
        # An empty parent bypasses payload application and withdrawals.
        pre.latest_execution_payload_bid.block_hash = s.Hash32(b"\x44" * 32)
        pre.current_sync_committee = s.SyncCommittee(
            pubkeys=s.SyncCommitteePubkeys(data=[pre.validators[0].pubkey] * s.SYNC_COMMITTEE_SIZE)
        )
        advanced = pre.copy()
        s.process_slots(advanced, slot)
        for vote in body.head_votes:
            if vote.data.slot == pre.slot:
                vote.data.beacon_block_root = s.get_block_root_at_slot(advanced, vote.data.slot)
                committee = s.get_head_vote_committee(advanced, vote.data.slot)
                vote.signature = sign(
                    s,
                    advanced,
                    vote.data,
                    s.DOMAIN_HEAD_VOTE,
                    s.compute_epoch_at_slot(vote.data.slot),
                    [
                        index
                        for index, bit in zip(committee, vote.aggregation_bits, strict=True)
                        if bit
                    ],
                )
        body.randao_reveal = sign(
            s,
            advanced,
            s.get_current_epoch(advanced),
            s.DOMAIN_RANDAO,
            s.get_current_epoch(advanced),
            [s.get_beacon_proposer_index(advanced)],
        )
        body.sync_aggregate.sync_committee_signature = s.G2_POINT_AT_INFINITY
        body.signed_execution_payload_bid = s.SignedExecutionPayloadBid(
            message=s.ExecutionPayloadBid(
                slot=slot,
                builder_index=s.BUILDER_INDEX_SELF_BUILD,
                parent_block_hash=advanced.latest_block_hash,
                block_hash=s.Hash32(b"\x55" * 32),
                parent_block_root=s.get_block_root_at_slot(advanced, slot - 1),
                prev_randao=s.get_randao_mix(advanced, s.get_current_epoch(advanced)),
                execution_requests_root=s.hash_tree_root(s.ExecutionRequests()),
            ),
            signature=s.G2_POINT_AT_INFINITY,
        )
        block = s.BeaconBlock(
            slot=slot,
            proposer_index=s.get_beacon_proposer_index(advanced),
            parent_root=s.hash_tree_root(advanced.latest_block_header),
            body=body,
        )
        return pre, s.SignedBeaconBlock(message=block)

    def materialize_solution(self, solution):
        claimed = self.claimed_record(self.target, solution)
        with real_bls():
            pre, block = self.build_block_inputs(solution)
            post = pre.copy()
            try:
                self.spec.state_transition(post, block, validate_result=False)
            except AssertionError:
                post = None
        if (post is not None) != claimed["accepted"]:
            raise ValueError(f"Unexpected block outcome: {claimed}")
        parts = [("pre", "ssz", pre.encode_bytes()), ("blocks_0", "ssz", block.encode_bytes())]
        if post is not None:
            parts.append(("post", "ssz", post.encode_bytes()))
        return {
            "description": "process_operations through state_transition",
            "bls_setting": 1,
            "blocks_count": 1,
            "claimed": claimed,
        }, parts


MATERIALIZER = OperationsMaterializer
