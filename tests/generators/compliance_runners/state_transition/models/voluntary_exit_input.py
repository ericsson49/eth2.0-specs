from dataclasses import dataclass
from enum import auto, Enum


class ExitEpochRelation(Enum):
    EXIT_EPOCH_CURRENT = auto()
    EXIT_EPOCH_FUTURE = auto()


class VoluntaryExitBranchTarget(Enum):
    VOLUNTARY_EXIT_SUCCESS = auto()
    VOLUNTARY_EXIT_INACTIVE = auto()
    VOLUNTARY_EXIT_ALREADY_EXITED = auto()
    VOLUNTARY_EXIT_FUTURE_EPOCH = auto()
    VOLUNTARY_EXIT_NOT_ACTIVE_LONG_ENOUGH = auto()
    VOLUNTARY_EXIT_PENDING_WITHDRAWAL = auto()


@dataclass
class VoluntaryExitInputProfile:
    branch_target: VoluntaryExitBranchTarget
    exit_epoch_relation: ExitEpochRelation


p: VoluntaryExitInputProfile = ...
