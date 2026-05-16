from dataclasses import dataclass
from enum import auto, Enum


class SyncAggregateBranchTarget(Enum):
    SYNC_AGGREGATE_ALL_PARTICIPATE = auto()
    SYNC_AGGREGATE_MAJORITY_PARTICIPATE = auto()
    SYNC_AGGREGATE_MINORITY_PARTICIPATE = auto()
    SYNC_AGGREGATE_NONE_PARTICIPATE = auto()
    SYNC_AGGREGATE_BAD_SIGNATURE = auto()


@dataclass
class SyncAggregateInputProfile:
    branch_target: SyncAggregateBranchTarget


p: SyncAggregateInputProfile = ...
