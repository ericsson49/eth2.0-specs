"""Materialize EIP-8453 round inactivity and ejection boundaries."""

from tests.generators.compliance_runners.state_transition.eip8453.round_materializer import (
    RoundMaterializer,
)

from .target import TARGET


def _bucket(delta):
    return (
        "LT_FAR"
        if delta < -1
        else "LT_1"
        if delta == -1
        else "EQ"
        if delta == 0
        else "GT_1"
        if delta == 1
        else "GT_FAR"
    )


class InactivityUpdatesMaterializer(RoundMaterializer):
    handler_name = "inactivity_updates"
    test_provider = "eip8453_inactivity_updates"
    target = TARGET

    def build_pre_state(self, solution):
        spec = self.spec
        if not TARGET.for_spec(spec).feasible(vars(solution)):
            raise ValueError("infeasible round-processing coverage assignment")
        recovery = int(spec.config.INACTIVITY_SCORE_RECOVERY_RATE)
        ejection_input = int(spec.EJECTION_INACTIVITY_SCORE) - int(
            spec.config.INACTIVITY_SCORE_BIAS
        )
        # Both comparisons concern the same score. Choose a representative at
        # their partition boundaries rather than solving them independently.
        candidates = {0} | {
            threshold + offset
            for threshold in (recovery, ejection_input)
            for offset in (-2, -1, 0, 1, 2)
            if threshold + offset >= 0
        }
        score = next(
            value
            for value in sorted(candidates)
            if (value == 0) == solution.initial_score_zero
            and _bucket(value - recovery) == solution.score_vs_recovery
            and _bucket(value - ejection_input) == solution.increased_score_vs_ejection
        )
        flags = (1 << int(spec.PROGRESS_FLAG_INDEX)) if solution.round_progress_flag else 0
        # Height participation is distinct from this round's progress flag.
        return self.base_state(
            solution, score=score, flags=flags, progress=not solution.round_progress_flag
        )


MATERIALIZER = InactivityUpdatesMaterializer
