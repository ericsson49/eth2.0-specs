from dataclasses import dataclass
from enum import auto, Enum


class PendingConsolidationSourceShape(Enum):
    SOURCE_WITHDRAWABLE = auto()
    SOURCE_NOT_WITHDRAWABLE = auto()
    SOURCE_SLASHED = auto()


class PendingConsolidationBalanceShape(Enum):
    BALANCE_EQUALS_EFFECTIVE_BALANCE = auto()
    BALANCE_LESS_THAN_EFFECTIVE_BALANCE = auto()
    BALANCE_GREATER_THAN_EFFECTIVE_BALANCE = auto()


class PendingConsolidationQueueShape(Enum):
    SINGLE_ITEM = auto()
    BLOCKED_AFTER_PROCESSED = auto()


@dataclass
class PendingConsolidationsInputProfile:
    source_shape: PendingConsolidationSourceShape
    balance_shape: PendingConsolidationBalanceShape
    queue_shape: PendingConsolidationQueueShape


p: PendingConsolidationsInputProfile = ...
