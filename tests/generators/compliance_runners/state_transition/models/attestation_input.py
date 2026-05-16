from dataclasses import dataclass
from enum import auto, Enum


class AttestationSlotRelation(Enum):
    CURRENT_EPOCH = auto()
    PREVIOUS_EPOCH = auto()
    FUTURE_SLOT = auto()


class TargetEpochRelation(Enum):
    MATCHES_SLOT_EPOCH = auto()
    WRONG_TARGET_EPOCH = auto()


class CommitteeIndexShape(Enum):
    COMMITTEE_VALID = auto()
    COMMITTEE_BAD_INDEX = auto()


class AggregationShape(Enum):
    AGGREGATION_NONEMPTY = auto()
    AGGREGATION_EMPTY = auto()


class AttestationBranchTarget(Enum):
    ATTESTATION_SUCCESS = auto()
    ATTESTATION_PREVIOUS_EPOCH_SUCCESS = auto()
    ATTESTATION_FUTURE_SLOT = auto()
    ATTESTATION_WRONG_TARGET_EPOCH = auto()
    ATTESTATION_BAD_COMMITTEE_INDEX = auto()
    ATTESTATION_EMPTY_AGGREGATION = auto()
    ATTESTATION_BAD_SIGNATURE = auto()


@dataclass
class AttestationInputProfile:
    branch_target: AttestationBranchTarget
    slot_relation: AttestationSlotRelation
    target_epoch_relation: TargetEpochRelation
    committee_index_shape: CommitteeIndexShape
    aggregation_shape: AggregationShape


p: AttestationInputProfile = ...
