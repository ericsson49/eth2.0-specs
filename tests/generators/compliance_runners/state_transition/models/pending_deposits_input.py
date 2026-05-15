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


@dataclass
class PendingDepositsInputProfile:
    deposit_kind: PendingDepositKind
    finality_shape: PendingDepositFinality
    churn_shape: PendingDepositChurn
    bridge_state: PendingDepositBridgeState


p: PendingDepositsInputProfile = ...
