from dataclasses import dataclass
from enum import auto, Enum


class WithdrawalRequestKind(Enum):
    FULL_EXIT_REQUEST = auto()
    PARTIAL_WITHDRAWAL_REQUEST = auto()


@dataclass
class WithdrawalRequestInputProfile:
    request_kind: WithdrawalRequestKind


p: WithdrawalRequestInputProfile = ...
