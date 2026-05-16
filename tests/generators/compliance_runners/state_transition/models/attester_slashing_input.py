from dataclasses import dataclass
from enum import auto, Enum


class AttesterOverlap(Enum):
    OVERLAPPING = auto()
    DISJOINT = auto()


class AttestationDataRelation(Enum):
    ATTESTATION_DATA_SLASHABLE = auto()
    ATTESTATION_DATA_SAME = auto()


class AttesterStatus(Enum):
    ATTESTER_SLASHABLE = auto()
    ATTESTER_ALREADY_SLASHED = auto()


class AttesterSlashingBranchTarget(Enum):
    ATTESTER_SLASHING_SUCCESS = auto()
    ATTESTER_SLASHING_NOT_SLASHABLE_DATA = auto()
    ATTESTER_SLASHING_NO_OVERLAP = auto()
    ATTESTER_SLASHING_ALREADY_SLASHED = auto()
    ATTESTER_SLASHING_BAD_SIGNATURE = auto()


@dataclass
class AttesterSlashingInputProfile:
    branch_target: AttesterSlashingBranchTarget
    attester_overlap: AttesterOverlap
    attestation_data_relation: AttestationDataRelation
    attester_status: AttesterStatus


p: AttesterSlashingInputProfile = ...
