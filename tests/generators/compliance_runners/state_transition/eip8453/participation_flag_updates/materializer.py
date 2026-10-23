"""Materialize EIP-8453 round flag clearing and height-progress carry-over."""

from tests.generators.compliance_runners.state_transition.eip8453.round_materializer import (
    RoundMaterializer,
)

from .target import TARGET


class ParticipationFlagUpdatesMaterializer(RoundMaterializer):
    handler_name = "participation_flag_updates"
    test_provider = "eip8453_participation_flag_updates"
    target = TARGET

    def build_pre_state(self, solution):
        spec = self.spec
        flags = ((1 << int(spec.PROGRESS_FLAG_INDEX)) if solution.previous_progress_flag else 0) | (
            (1 << int(spec.FINALITY_FLAG_INDEX)) if solution.previous_other_flags else 0
        )
        return self.base_state(solution, score=7, flags=flags, progress=solution.carry_progress)


MATERIALIZER = ParticipationFlagUpdatesMaterializer
