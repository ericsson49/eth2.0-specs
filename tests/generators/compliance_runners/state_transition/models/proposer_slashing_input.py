from dataclasses import dataclass
from enum import auto, Enum


class HeaderRelation(Enum):
    DIFFERENT_HEADERS = auto()
    SAME_HEADER = auto()


class ProposerRelation(Enum):
    SAME_PROPOSER = auto()
    DIFFERENT_PROPOSER = auto()


class ProposerStatus(Enum):
    PROPOSER_SLASHABLE = auto()
    PROPOSER_ALREADY_SLASHED = auto()


class ProposerSlashingBranchTarget(Enum):
    PROPOSER_SLASHING_SUCCESS = auto()
    PROPOSER_SLASHING_SAME_HEADER = auto()
    PROPOSER_SLASHING_PROPOSER_MISMATCH = auto()
    PROPOSER_SLASHING_ALREADY_SLASHED = auto()
    PROPOSER_SLASHING_BAD_SIGNATURE = auto()


@dataclass
class ProposerSlashingInputProfile:
    branch_target: ProposerSlashingBranchTarget
    header_relation: HeaderRelation
    proposer_relation: ProposerRelation
    proposer_status: ProposerStatus


p: ProposerSlashingInputProfile = ...
