from dataclasses import dataclass
from enum import auto, Enum


class BLSCredentialShape(Enum):
    BLS_CREDENTIALS = auto()
    EXECUTION_CREDENTIALS = auto()


class WithdrawalPubkeyRelation(Enum):
    PUBKEY_MATCHES = auto()
    PUBKEY_MISMATCH = auto()


@dataclass
class BLSToExecutionChangeInputProfile:
    credential_shape: BLSCredentialShape
    withdrawal_pubkey_relation: WithdrawalPubkeyRelation


p: BLSToExecutionChangeInputProfile = ...
