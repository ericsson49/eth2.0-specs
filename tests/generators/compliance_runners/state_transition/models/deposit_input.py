from dataclasses import dataclass
from enum import auto, Enum


class DepositRecipientShape(Enum):
    NEW_VALIDATOR = auto()
    TOP_UP_EXISTING_VALIDATOR = auto()


class DepositBranchTarget(Enum):
    DEPOSIT_NEW_VALIDATOR = auto()
    DEPOSIT_TOP_UP_EXISTING_VALIDATOR = auto()
    DEPOSIT_INVALID_PROOF = auto()
    DEPOSIT_INVALID_SIGNATURE = auto()


@dataclass
class DepositInputProfile:
    branch_target: DepositBranchTarget
    recipient_shape: DepositRecipientShape


p: DepositInputProfile = ...
