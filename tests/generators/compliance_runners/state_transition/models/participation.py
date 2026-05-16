from dataclasses import dataclass
from enum import auto, Enum


class ParticipationShape(Enum):
    PARTICIPATION_NONE = auto()
    PARTICIPATION_TARGET_ONLY = auto()
    PARTICIPATION_FULL = auto()
    PARTICIPATION_POOR_SUPPORT = auto()


class FinalityShape(Enum):
    FINALITY_NONE = auto()
    FINALITY_CURRENT_JUSTIFIED = auto()
    FINALITY_PREVIOUS_JUSTIFIED = auto()
    FINALITY_FINALIZE_CURRENT = auto()


class ParticipationBranchTarget(Enum):
    PARTICIPATION_FINALITY_GENESIS_SKIP = auto()
    PARTICIPATION_FINALITY_CURRENT_JUSTIFIED = auto()
    PARTICIPATION_FINALITY_PREVIOUS_JUSTIFIED = auto()
    PARTICIPATION_FINALITY_POOR_SUPPORT = auto()
    PARTICIPATION_FINALITY_FINALIZE_CURRENT = auto()
    PARTICIPATION_FINALITY_FINALIZE_234 = auto()
    PARTICIPATION_FINALITY_FINALIZE_23 = auto()
    PARTICIPATION_FINALITY_FINALIZE_123 = auto()
    PARTICIPATION_INACTIVITY_GENESIS_SKIP = auto()
    PARTICIPATION_INACTIVITY_PARTICIPATING_RECOVERY = auto()
    PARTICIPATION_INACTIVITY_NON_PARTICIPATING_NO_LEAK = auto()
    PARTICIPATION_INACTIVITY_NON_PARTICIPATING_LEAK = auto()
    PARTICIPATION_REWARDS_GENESIS_SKIP = auto()
    PARTICIPATION_REWARDS_FULL_PARTICIPATION = auto()
    PARTICIPATION_REWARDS_EMPTY_PARTICIPATION = auto()
    PARTICIPATION_REWARDS_INACTIVITY_LEAK_PENALTY = auto()
    PARTICIPATION_REWARDS_INACTIVITY_LEAK_FULL_PARTICIPATION = auto()
    PARTICIPATION_FLAGS_ALL_ZERO = auto()
    PARTICIPATION_FLAGS_CURRENT_FILLED = auto()
    PARTICIPATION_FLAGS_PREVIOUS_FILLED = auto()


@dataclass
class ParticipationProfile:
    branch_target: ParticipationBranchTarget
    participation_shape: ParticipationShape
    finality_shape: FinalityShape
    inactivity_leak: bool


p: ParticipationProfile = ...


if p.finality_shape != FinalityShape.FINALITY_NONE:
    p.participation_shape != ParticipationShape.PARTICIPATION_NONE

if p.inactivity_leak:
    p.participation_shape != ParticipationShape.PARTICIPATION_POOR_SUPPORT
