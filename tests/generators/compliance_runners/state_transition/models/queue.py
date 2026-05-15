from dataclasses import dataclass
from enum import auto, Enum


class QueueShape(Enum):
    EMPTY = auto()
    NONEMPTY = auto()
    FULL = auto()


class PendingRequestShape(Enum):
    REQUEST_NONE = auto()
    REQUEST_WITHDRAWAL = auto()
    REQUEST_CONSOLIDATION = auto()


@dataclass
class QueueProfile:
    pending_partial_withdrawals: QueueShape
    pending_consolidations: QueueShape
    pending_deposits: QueueShape
    pending_request: PendingRequestShape


p: QueueProfile = ...


if p.pending_partial_withdrawals == QueueShape.FULL:
    p.pending_request != PendingRequestShape.REQUEST_CONSOLIDATION

if p.pending_consolidations == QueueShape.FULL:
    p.pending_request != PendingRequestShape.REQUEST_WITHDRAWAL
