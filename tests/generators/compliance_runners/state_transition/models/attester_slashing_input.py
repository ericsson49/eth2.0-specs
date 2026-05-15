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


@dataclass
class AttesterSlashingInputProfile:
    attester_overlap: AttesterOverlap
    attestation_data_relation: AttestationDataRelation
    attester_status: AttesterStatus


p: AttesterSlashingInputProfile = ...
