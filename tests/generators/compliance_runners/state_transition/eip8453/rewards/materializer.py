"""Construct deterministic reward vectors with unequal supporting balances."""

from tests.generators.compliance_runners.state_transition.eip8453.round_materializer import (
    RoundMaterializer,
)

from .target import TARGET


class RewardsMaterializer(RoundMaterializer):
    handler_name = "rewards"
    test_provider = "eip8453_rewards"
    target = TARGET

    def build_pre_state(self, solution):
        spec = self.spec
        # Validate domains as well as feasibility before constructing SSZ inputs.
        self.claimed_record(TARGET, solution)
        if not TARGET.for_spec(spec).feasible(vars(solution)):
            raise ValueError("infeasible rewards coverage assignment")
        offset = {"LT_FAR": -2, "LT_1": -1, "EQ": 0, "GT_1": 1, "GT_FAR": 2}[solution.delay_vs_leak]
        delay = int(spec.MIN_ROUNDS_TO_INACTIVITY_LEAK) + offset
        current_round = delay + 3
        slot = (current_round + 1) * int(spec.SLOTS_PER_ROUND) - 1
        epoch = slot // int(spec.SLOTS_PER_EPOCH)
        increment = int(spec.EFFECTIVE_BALANCE_INCREMENT)
        focused_increments = {
            "ZERO": 0,
            "ONE": 1,
            "MANY": int(spec.MIN_ACTIVATION_BALANCE) // increment,
        }[solution.balance_increments]
        progress_mask = 1 << int(spec.PROGRESS_FLAG_INDEX)
        finality_mask = 1 << int(spec.FINALITY_FLAG_INDEX)
        focused_flags = (progress_mask if solution.progress_flag else 0) | (
            finality_mask if solution.finality_flag else 0
        )
        peer_flags = (progress_mask if solution.other_progress_participants else 0) | (
            finality_mask if solution.other_finality_participants else 0
        )
        validators = [
            spec.Validator(
                activation_epoch=0 if solution.active_validator else epoch + 1,
                exit_epoch=spec.FAR_FUTURE_EPOCH,
                withdrawable_epoch=spec.FAR_FUTURE_EPOCH,
                effective_balance=focused_increments * increment,
                slashed=solution.validator_slashed,
            )
        ]
        validators.extend(
            spec.Validator(
                activation_epoch=0,
                exit_epoch=spec.FAR_FUTURE_EPOCH,
                effective_balance=n * increment,
            )
            for n in (1, 2)
        )
        return spec.BeaconState(
            slot=slot,
            finalized_slot=2 * int(spec.SLOTS_PER_ROUND),
            validators=spec.Validators(data=validators),
            balances=spec.Balances(data=[spec.MIN_ACTIVATION_BALANCE] * 3),
            # Nonzero inactivity scores must not act as virtual reward weights.
            inactivity_scores=spec.InactivityScores(data=[7, 11, 0]),
            round_participation=spec.RoundParticipation(
                data=[spec.ParticipationFlags(f) for f in (focused_flags, peer_flags, peer_flags)]
            ),
            progress_participation=spec.HeightParticipation(data=[spec.Boolean()] * 3),
            target_participation=spec.HeightParticipation(data=[spec.Boolean()] * 3),
        )


MATERIALIZER = RewardsMaterializer
