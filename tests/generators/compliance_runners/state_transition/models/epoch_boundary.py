from dataclasses import dataclass
from enum import auto, Enum


class EpochBoundaryShape(Enum):
    GENESIS = auto()
    NORMAL = auto()
    PERIOD_BOUNDARY = auto()
    NON_BOUNDARY = auto()


@dataclass
class EpochBoundaryProfile:
    epoch_boundary_shape: EpochBoundaryShape


p: EpochBoundaryProfile = ...
