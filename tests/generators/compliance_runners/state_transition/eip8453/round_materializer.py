"""Shared state-only vector construction for per-validator round targets."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from tests.generators.compliance_runners.state_transition.materializer import Materializer

if TYPE_CHECKING:
    from tests.generators.compliance_runners.gen_base.gen_typing import TestCasePart


class RoundMaterializer(Materializer):
    runner_name = "round_processing"
    target: Any

    def __init__(
        self, spec: Any, fork_name: str = "eip8453", preset_name: str = "minimal", seed: int = 0
    ) -> None:
        if spec.fork != "eip8453" or fork_name != spec.fork:
            raise ValueError("round materializer requires the EIP-8453 fork")
        super().__init__(spec, fork_name, preset_name, seed)

    def base_state(self, solution: Any, *, score: int, flags: int, progress: bool) -> Any:
        spec = self.spec
        if not self.target.for_spec(spec).feasible(vars(solution)):
            raise ValueError("infeasible round-processing coverage assignment")
        epoch = 3
        exit_set = getattr(solution, "exit_already_set", False)
        active = solution.active_validator
        focused = spec.Validator(
            activation_epoch=0 if active or exit_set else epoch + 1,
            exit_epoch=(epoch + 1 if active else epoch) if exit_set else spec.FAR_FUTURE_EPOCH,
            withdrawable_epoch=spec.FAR_FUTURE_EPOCH,
            effective_balance=spec.MIN_ACTIVATION_BALANCE,
            slashed=solution.validator_slashed,
        )
        return spec.BeaconState(
            slot=epoch * int(spec.SLOTS_PER_EPOCH) + int(spec.SLOTS_PER_ROUND) - 1,
            validators=spec.Validators(data=[focused]),
            balances=spec.Balances(data=[spec.MIN_ACTIVATION_BALANCE]),
            inactivity_scores=spec.InactivityScores(data=[score]),
            round_participation=spec.RoundParticipation(data=[spec.ParticipationFlags(flags)]),
            progress_participation=spec.HeightParticipation(data=[spec.Boolean(progress)]),
            target_participation=spec.HeightParticipation(data=[spec.Boolean()]),
        )

    def build_pre_state(self, solution: Any) -> Any:
        raise NotImplementedError

    def materialize_solution(self, solution: Any) -> tuple[dict, list[TestCasePart]]:
        pre = self.build_pre_state(solution)
        post = pre.copy()
        getattr(self.spec, f"process_{self.handler_name}")(post)
        return {
            "description": f"process_{self.handler_name}",
            "claimed": self.claimed_record(self.target, solution),
        }, [("pre", "ssz", pre.encode_bytes()), ("post", "ssz", post.encode_bytes())]
