"""Existing-validator deposits straddling exact finalization boundaries."""

from tests.generators.compliance_runners.state_transition.eip8453.epoch_materializer import (
    EpochMaterializer,
)

from .target import TARGET


class PendingDepositsMaterializer(EpochMaterializer):
    handler_name = "pending_deposits"
    test_provider = "eip8453_pending_deposits"
    target = TARGET

    def build_pre_state(self, solution):
        self.claimed_record(TARGET, solution)
        s = self.spec
        n = int(s.SLOTS_PER_EPOCH)
        finalized = (
            2 * n + {"START": 0, "INTERIOR": n // 2, "END": n - 1}[solution.finalized_position]
        )
        role = solution.primary_role
        validators = [
            s.Validator(
                pubkey=(i + 1).to_bytes(48, "little"),
                activation_epoch=0,
                exit_epoch=s.FAR_FUTURE_EPOCH,
                withdrawable_epoch=s.FAR_FUTURE_EPOCH,
                effective_balance=s.MIN_ACTIVATION_BALANCE,
            )
            for i in range(2)
        ]
        if role != "ACTIVE":
            validators[0].exit_epoch = s.Epoch(2)
            validators[0].withdrawable_epoch = s.Epoch(3 if role == "WITHDRAWN" else 4)
        pre = s.BeaconState(
            slot=4 * n - 1,
            finalized_slot=finalized,
            validators=s.Validators(data=validators),
            balances=s.Balances(data=[s.MIN_ACTIVATION_BALANCE] * 2),
            inactivity_scores=s.InactivityScores(data=[0, 0]),
            deposit_balance_to_consume=7 if solution.carry_nonzero else 0,
        )
        available = int(pre.deposit_balance_to_consume) + int(s.get_activation_churn_limit(pre))
        amount = available + {"LT": -1, "EQ": 0, "GT": 1}[solution.amount_vs_churn]
        slot = finalized + {"LT": -1, "EQ": 0, "GT": 1}[solution.deposit_vs_finalized_slot]
        pre.pending_deposits.append(
            s.PendingDeposit(pubkey=validators[0].pubkey, slot=slot, amount=amount)
        )
        if solution.suffix != "NONE":
            pre.pending_deposits.append(
                s.PendingDeposit(
                    pubkey=validators[1].pubkey,
                    amount=1,
                    slot=finalized + (1 if solution.suffix == "UNFINALIZED" else 0),
                )
            )
        return pre


MATERIALIZER = PendingDepositsMaterializer
