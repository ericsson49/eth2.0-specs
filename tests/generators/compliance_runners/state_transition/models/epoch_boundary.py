from dataclasses import dataclass
from enum import auto, Enum


class EpochBoundaryShape(Enum):
    GENESIS = auto()
    NORMAL = auto()
    PERIOD_BOUNDARY = auto()
    NON_BOUNDARY = auto()


class EpochBoundaryBranchTarget(Enum):
    EPOCH_GENESIS_SKIP = auto()
    ETH1_DATA_PERIOD_BOUNDARY = auto()
    ETH1_DATA_NON_BOUNDARY = auto()
    HISTORICAL_SUMMARIES_PERIOD_BOUNDARY = auto()
    HISTORICAL_SUMMARIES_NON_BOUNDARY = auto()
    SYNC_COMMITTEE_PERIOD_BOUNDARY = auto()
    SYNC_COMMITTEE_GENESIS_PERIOD_BOUNDARY = auto()
    SYNC_COMMITTEE_NON_BOUNDARY = auto()
    SLASHINGS_RESET_NONZERO = auto()
    SLASHINGS_RESET_ALREADY_ZERO = auto()
    RANDAO_RESET_TO_CURRENT_MIX = auto()
    RANDAO_ALREADY_CURRENT_MIX = auto()


@dataclass
class EpochBoundaryProfile:
    branch_target: EpochBoundaryBranchTarget
    epoch_boundary_shape: EpochBoundaryShape


p: EpochBoundaryProfile = ...
