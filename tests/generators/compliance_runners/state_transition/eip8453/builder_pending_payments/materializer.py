"""Construct distinct payment/vote buffers to expose rotation and append bugs."""

from tests.generators.compliance_runners.state_transition.eip8453.epoch_materializer import (
    EpochMaterializer,
)

from .target import TARGET


class BuilderPendingPaymentsMaterializer(EpochMaterializer):
    handler_name = "builder_pending_payments"
    test_provider = "eip8453_builder_pending_payments"
    target = TARGET

    def build_pre_state(self, solution):
        self.claimed_record(TARGET, solution)
        s = self.spec
        n = int(s.SLOTS_PER_EPOCH)
        pre = s.BeaconState(slot=3 * n - 1)
        threshold = TARGET.for_spec(s).bound_constants["required_seats"]
        seats = (
            threshold
            + {"LT_FAR": -2, "LT_1": -1, "EQ": 0, "GT_1": 1, "GT_FAR": 2}[
                solution.seats_vs_required
            ]
        )

        def bits(count):
            return s.HeadVoteCommitteeBits(
                data=[i < count for i in range(s.HEAD_VOTE_COMMITTEE_SIZE)]
            )

        def withdrawal(amount, marker):
            return s.BuilderPendingWithdrawal(
                amount=amount, builder_index=marker, fee_recipient=marker.to_bytes(20, "little")
            )

        def payment(amount, marker, weight=0):
            return s.BuilderPendingPayment(
                withdrawal=withdrawal(amount, marker), weight=weight, proposer_index=marker
            )

        pre.builder_pending_payments[0] = payment(
            11 if solution.amount_nonzero else 0, 1, 10**12 if solution.legacy_weight_nonzero else 0
        )
        pre.head_vote_participation[0] = bits(seats)
        if solution.tail_payment:
            pre.builder_pending_payments[n - 1] = payment(22, 2)
            pre.head_vote_participation[n - 1] = bits(threshold)
        if solution.next_payments:
            pre.builder_pending_payments[n] = payment(33, 3, 17)
            pre.builder_pending_payments[2 * n - 1] = payment(44, 4, 23)
        if solution.next_votes:
            pre.head_vote_participation[n] = bits(1)
            pre.head_vote_participation[2 * n - 1] = bits(threshold)
        if solution.existing_withdrawals:
            pre.builder_pending_withdrawals.append(withdrawal(55, 5))
        return pre


MATERIALIZER = BuilderPendingPaymentsMaterializer
