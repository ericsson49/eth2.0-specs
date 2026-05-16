from dataclasses import dataclass
from enum import auto, Enum


class WithdrawalRequestKind(Enum):
    FULL_EXIT_REQUEST = auto()
    PARTIAL_WITHDRAWAL_REQUEST = auto()


class WithdrawalRequestBranchTarget(Enum):
    WITHDRAWAL_QUEUE_FULL_PARTIAL = auto()
    WITHDRAWAL_PUBKEY_MISSING = auto()
    WITHDRAWAL_BAD_SOURCE_ADDRESS = auto()
    WITHDRAWAL_SOURCE_INACTIVE = auto()
    WITHDRAWAL_SOURCE_EXITING = auto()
    WITHDRAWAL_NOT_ACTIVE_LONG_ENOUGH = auto()
    WITHDRAWAL_FULL_EXIT_PENDING_WITHDRAWAL = auto()
    WITHDRAWAL_FULL_EXIT_SUCCESS = auto()
    WITHDRAWAL_PARTIAL_CONDITIONS_NOT_MET = auto()
    WITHDRAWAL_PARTIAL_SUCCESS = auto()


@dataclass
class WithdrawalRequestInputProfile:
    request_kind: WithdrawalRequestKind
    branch_target: WithdrawalRequestBranchTarget


p: WithdrawalRequestInputProfile = ...
