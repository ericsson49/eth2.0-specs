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


@dataclass
class ConsolidationRequestInputProfile:
    request_kind: ConsolidationRequestKind
    target_lookup_shape: TargetLookupShape
    source_activity_shape: SourceActivityShape
    target_activity_shape: TargetActivityShape
    target_credential_shape: TargetCredentialShape
    churn_shape: ConsolidationChurnShape


p: ConsolidationRequestInputProfile = ...
