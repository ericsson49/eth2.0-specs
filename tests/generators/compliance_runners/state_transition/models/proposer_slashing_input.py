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


@dataclass
class ProposerSlashingInputProfile:
    header_relation: HeaderRelation
    proposer_relation: ProposerRelation
    proposer_status: ProposerStatus


p: ProposerSlashingInputProfile = ...
