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


@dataclass
class ParticipationProfile:
    participation_shape: ParticipationShape
    finality_shape: FinalityShape
    inactivity_leak: bool


p: ParticipationProfile = ...


if p.finality_shape != FinalityShape.FINALITY_NONE:
    p.participation_shape != ParticipationShape.PARTICIPATION_NONE

if p.inactivity_leak:
    p.participation_shape != ParticipationShape.PARTICIPATION_POOR_SUPPORT
