from dataclasses import dataclass
from enum import auto, Enum


class ConsolidationRequestKind(Enum):
    CONSOLIDATION_REQUEST = auto()
    SWITCH_TO_COMPOUNDING_REQUEST = auto()


class TargetLookupShape(Enum):
    TARGET_PRESENT = auto()
    TARGET_MISSING = auto()


class SourceActivityShape(Enum):
    SOURCE_ACTIVE = auto()
    SOURCE_INACTIVE = auto()
    SOURCE_EXITING = auto()
    SOURCE_NOT_ACTIVE_LONG_ENOUGH = auto()


class TargetActivityShape(Enum):
    TARGET_ACTIVE = auto()
    TARGET_INACTIVE = auto()
    TARGET_EXITING = auto()


class TargetCredentialShape(Enum):
    TARGET_COMPOUNDING = auto()
    TARGET_ETH1 = auto()


class ConsolidationChurnShape(Enum):
    CHURN_AVAILABLE = auto()
    CHURN_TOO_LOW = auto()


class ConsolidationRequestBranchTarget(Enum):
    CONSOLIDATION_SWITCH_SUCCESS = auto()
    CONSOLIDATION_SWITCH_PUBKEY_MISSING = auto()
    CONSOLIDATION_SWITCH_BAD_SOURCE_ADDRESS = auto()
    CONSOLIDATION_SWITCH_SOURCE_INACTIVE = auto()
    CONSOLIDATION_SWITCH_SOURCE_EXITING = auto()
    CONSOLIDATION_SOURCE_EQUALS_TARGET = auto()
    CONSOLIDATION_QUEUE_FULL = auto()
    CONSOLIDATION_CHURN_TOO_LOW = auto()
    CONSOLIDATION_SOURCE_MISSING = auto()
    CONSOLIDATION_TARGET_MISSING = auto()
    CONSOLIDATION_BAD_SOURCE_ADDRESS = auto()
    CONSOLIDATION_TARGET_NOT_COMPOUNDING = auto()
    CONSOLIDATION_SOURCE_INACTIVE = auto()
    CONSOLIDATION_TARGET_INACTIVE = auto()
    CONSOLIDATION_SOURCE_EXITING = auto()
    CONSOLIDATION_TARGET_EXITING = auto()
    CONSOLIDATION_SOURCE_NOT_ACTIVE_LONG_ENOUGH = auto()
    CONSOLIDATION_SOURCE_PENDING_WITHDRAWAL = auto()
    CONSOLIDATION_SUCCESS = auto()


@dataclass
class ConsolidationRequestInputProfile:
    branch_target: ConsolidationRequestBranchTarget
    request_kind: ConsolidationRequestKind
    target_lookup_shape: TargetLookupShape
    source_activity_shape: SourceActivityShape
    target_activity_shape: TargetActivityShape
    target_credential_shape: TargetCredentialShape
    churn_shape: ConsolidationChurnShape


p: ConsolidationRequestInputProfile = ...
