from dataclasses import dataclass
from enum import auto, Enum


class ExitEpochRelation(Enum):
    EXIT_EPOCH_CURRENT = auto()
    EXIT_EPOCH_FUTURE = auto()


@dataclass
class VoluntaryExitInputProfile:
    exit_epoch_relation: ExitEpochRelation


p: VoluntaryExitInputProfile = ...
