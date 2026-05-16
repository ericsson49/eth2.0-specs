from dataclasses import dataclass
from enum import auto, Enum


class BLSCredentialShape(Enum):
    BLS_CREDENTIALS = auto()
    EXECUTION_CREDENTIALS = auto()


class WithdrawalPubkeyRelation(Enum):
    PUBKEY_MATCHES = auto()
    PUBKEY_MISMATCH = auto()


class BLSToExecutionChangeBranchTarget(Enum):
    BLS_TO_EXECUTION_SUCCESS = auto()
    BLS_TO_EXECUTION_OUT_OF_RANGE = auto()
    BLS_TO_EXECUTION_NOT_BLS_CREDENTIALS = auto()
    BLS_TO_EXECUTION_PUBKEY_MISMATCH = auto()
    BLS_TO_EXECUTION_BAD_SIGNATURE = auto()


@dataclass
class BLSToExecutionChangeInputProfile:
    branch_target: BLSToExecutionChangeBranchTarget
    credential_shape: BLSCredentialShape
    withdrawal_pubkey_relation: WithdrawalPubkeyRelation


p: BLSToExecutionChangeInputProfile = ...
