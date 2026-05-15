from dataclasses import dataclass
from enum import auto, Enum


class DepositRecipientShape(Enum):
    NEW_VALIDATOR = auto()
    TOP_UP_EXISTING_VALIDATOR = auto()


@dataclass
class DepositInputProfile:
    recipient_shape: DepositRecipientShape


p: DepositInputProfile = ...
