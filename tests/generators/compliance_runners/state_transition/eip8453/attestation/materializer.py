"""Signed aggregates across all attestation committees."""

from tests.generators.compliance_runners.state_transition.eip8453.operation_materializer import (
    base_state,
    checkpoint,
    OperationMaterializer,
    sign,
)

from .target import TARGET


class AttestationMaterializer(OperationMaterializer):
    handler_name = input_name = "attestation"
    test_provider = "eip8453_attestation"
    target = TARGET

    def build_inputs(self, solution):
        s = self.spec
        pre = base_state(s)
        pre.finalized_slot = s.Slot(2 * s.SLOTS_PER_EPOCH)
        pre.target_checkpoint = checkpoint(s, 10)
        pre.pending_finality.append(
            s.PendingFinality(
                checkpoint=checkpoint(s, 5),
                slot=s.Slot(s.SLOTS_PER_EPOCH),
                participation=s.HeightParticipation(
                    data=[s.Boolean(solution.finality_recorded)] * 8
                ),
            )
        )
        pre.progress_participation = s.HeightParticipation(
            data=[s.Boolean(solution.progress_recorded)] * 8
        )
        pre.round_participation = s.RoundParticipation(
            data=[
                (int(solution.progress_recorded) << int(s.PROGRESS_FLAG_INDEX))
                | (int(solution.finality_recorded) << int(s.FINALITY_FLAG_INDEX))
            ]
            * 8
        )
        case = solution.case
        data = s.AttestationData(
            target=checkpoint(s, 10), finality=checkpoint(s, 5), timeout=solution.timeout
        )
        if case in ("OLD", "UNKNOWN"):
            data.target = checkpoint(s, 9)
        if case in ("NO_FINALITY", "UNKNOWN"):
            data.finality = s.HeightCheckpoint.empty()
        if case == "MALFORMED":
            data.target.root = s.Root()
        epoch = 4 if case == "FUTURE" else 1 if case == "STALE" else 3
        operation = s.Attestation(
            data=data,
            epoch=epoch,
            committee_bits=s.AttestationCommitteeBits(
                data=[True] * s.ATTESTATION_COMMITTEES_PER_ROUND
            ),
            aggregation_bits=s.AggregationBits(data=[case != "EMPTY"] * 8),
        )
        operation.signature = sign(s, pre, data, s.DOMAIN_FINALITY_ATTESTER, epoch, range(8))
        if case == "BITS":
            operation.aggregation_bits.append(True)
        if case == "SIGNATURE":
            # Valid BLS bytes, wrong domain: catches accidental legacy-domain use.
            operation.signature = sign(s, pre, data, s.DOMAIN_BEACON_ATTESTER, epoch, range(8))
        return pre, operation


MATERIALIZER = AttestationMaterializer
