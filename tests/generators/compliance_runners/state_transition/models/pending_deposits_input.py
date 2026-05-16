from dataclasses import dataclass
from enum import auto, Enum


class PendingDepositKind(Enum):
    EXISTING_ACTIVE_VALIDATOR = auto()
    NEW_VALIDATOR = auto()
    EXITING_VALIDATOR = auto()
    WITHDRAWABLE_VALIDATOR = auto()


class PendingDepositFinality(Enum):
    FINALIZED = auto()
    NOT_FINALIZED = auto()


class PendingDepositChurn(Enum):
    CHURN_AVAILABLE = auto()
    CHURN_LIMIT_REACHED = auto()


class PendingDepositBridgeState(Enum):
    ETH1_BRIDGE_APPLIED = auto()
    ETH1_BRIDGE_PENDING = auto()


class PendingDepositsBranchTarget(Enum):
    PENDING_DEPOSITS_SUCCESS_TOP_UP = auto()
    PENDING_DEPOSITS_NOT_FINALIZED = auto()
    PENDING_DEPOSITS_CHURN_LIMIT_REACHED = auto()
    PENDING_DEPOSITS_EXITED_VALIDATOR_POSTPONED = auto()
    PENDING_DEPOSITS_WITHDRAWABLE_VALIDATOR = auto()
    PENDING_DEPOSITS_ETH1_BRIDGE_BLOCKS_REQUEST = auto()
    PENDING_DEPOSITS_MAX_PER_EPOCH_REACHED = auto()
    PENDING_DEPOSITS_NEW_VALIDATOR = auto()


@dataclass
class PendingDepositsInputProfile:
    branch_target: PendingDepositsBranchTarget
    deposit_kind: PendingDepositKind
    finality_shape: PendingDepositFinality
    churn_shape: PendingDepositChurn
    bridge_state: PendingDepositBridgeState


p: PendingDepositsInputProfile = ...
