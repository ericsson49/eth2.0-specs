"""Materialize direct EIP-8453 ``process_height_events`` state vectors."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from tests.generators.compliance_runners.state_transition.materializer import Materializer

from .target import TARGET

if TYPE_CHECKING:
    from tests.generators.compliance_runners.gen_base.gen_typing import TestCasePart


def _count_bucket(value: int) -> str:
    return "ZERO" if value == 0 else "ONE" if value == 1 else "MANY"


def _boundary(votes: int, total: int) -> str:
    return "LT" if 3 * votes < 2 * total else "EQ" if 3 * votes == 2 * total else "GT"


class HeightEventsMaterializer(Materializer):
    # State-only operation: the immediate pre-state is after block attestations.
    runner_name = "operations"
    handler_name = "height_events"
    test_provider = "eip8453_height_events"

    def __init__(
        self, spec: Any, fork_name: str = "eip8453", preset_name: str = "minimal", seed: int = 0
    ) -> None:
        super().__init__(spec, fork_name, preset_name, seed)

    def build_pre_state(self, solution: Any) -> Any:
        """Realize a complete coverage witness as the handler's immediate input."""
        spec = self.spec
        if not TARGET.for_spec(spec).feasible(vars(solution)):
            raise ValueError("infeasible height-events coverage assignment")

        # Three equal-weight validators realize every LT/EQ/GT quorum boundary.
        # Zero total weight is separate: equality alone must not grant a quorum.
        total, target_votes, progress_votes = next(
            (total, target, progress)
            for total in (0, 3)
            for target in range(total + 1)
            for progress in range(target, total + 1)
            if (target > 0) == solution.target_positive
            and (progress > 0) == solution.progress_positive
            and _boundary(target, total) == solution.target_boundary
            and _boundary(progress, total) == solution.progress_boundary
        )
        window = int(spec.PENDING_FINALITY_WINDOW)
        pending_count, qualifying_count, retained_count = next(
            (pending, qualifying, retained)
            for pending in range(window + 1)
            for qualifying in range(pending + 1)
            for retained in (range(pending - qualifying + 1) if qualifying else (pending,))
            if _count_bucket(pending) == solution.pending
            and _count_bucket(qualifying) == solution.qualifying
            and _count_bucket(retained) == solution.retained
            and (retained == window) == solution.window_full_after_pruning
        )

        delta = {"LT_FAR": -2, "LT_1": -1, "EQ": 0, "GT_1": 1, "GT_FAR": 2}[solution.timeout]
        # Leave room for all pending slots before the target and all timeout offsets.
        target_slot = window + 2
        slot = target_slot + int(spec.TIMEOUT_DELAY_SLOTS) + delta

        def participation(votes):
            return spec.HeightParticipation(data=[spec.Boolean(i < votes) for i in range(3)])

        def checkpoint(height):
            return spec.HeightCheckpoint(height=height, root=height.to_bytes(32, "little"))

        # Earlier nonqualifying entries can be pruned too. Place the final quorum
        # immediately before the retained suffix, with any other quorums first.
        qualifying_indices = set(range(max(0, qualifying_count - 1)))
        if qualifying_count:
            qualifying_indices.add(pending_count - retained_count - 1)

        return spec.BeaconState(
            slot=slot,
            latest_block_header=spec.BeaconBlockHeader(slot=slot),
            validators=spec.Validators(
                data=[
                    spec.Validator(activation_epoch=0, exit_epoch=spec.FAR_FUTURE_EPOCH)
                    for _ in range(3)
                ]
            ),
            balances=spec.Balances(data=[spec.MIN_ACTIVATION_BALANCE] * 3),
            inactivity_scores=spec.InactivityScores(
                data=[0 if total else spec.EJECTION_INACTIVITY_SCORE] * 3
            ),
            finalized_checkpoint=checkpoint(0),
            finalized_slot=0,
            justified_checkpoint=checkpoint(pending_count),
            target_checkpoint=checkpoint(window + 1),
            target_slot=target_slot,
            target_participation=participation(target_votes),
            progress_participation=participation(progress_votes),
            pending_finality=spec.PendingFinalities(
                data=[
                    spec.PendingFinality(
                        checkpoint=checkpoint(i + 1),
                        slot=i + 1,
                        participation=participation(2 if i in qualifying_indices else 0),
                    )
                    for i in range(pending_count)
                ]
            ),
        )

    def materialize_solution(self, solution: Any) -> tuple[dict, list[TestCasePart]]:
        pre = self.build_pre_state(solution)
        post = pre.copy()
        self.spec.process_height_events(post)
        meta = {
            "description": "process_height_events",
            "claimed": self.claimed_record(TARGET, solution),
        }
        return meta, [("pre", "ssz", pre.encode_bytes()), ("post", "ssz", post.encode_bytes())]


MATERIALIZER = HeightEventsMaterializer
