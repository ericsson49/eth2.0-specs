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


@dataclass
class AttestationInputProfile:
    slot_relation: AttestationSlotRelation
    target_epoch_relation: TargetEpochRelation
    committee_index_shape: CommitteeIndexShape
    aggregation_shape: AggregationShape


p: AttestationInputProfile = ...
