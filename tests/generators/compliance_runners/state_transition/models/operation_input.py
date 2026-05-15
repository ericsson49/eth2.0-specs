from dataclasses import dataclass
from enum import auto, Enum


class SignatureShape(Enum):
    SIGNATURE_VALID = auto()
    SIGNATURE_INVALID = auto()


class ProofShape(Enum):
    PROOF_VALID = auto()
    PROOF_INVALID = auto()


class LookupShape(Enum):
    LOOKUP_PRESENT = auto()
    LOOKUP_MISSING = auto()


class SourceAddressShape(Enum):
    SOURCE_ADDRESS_VALID = auto()
    SOURCE_ADDRESS_INVALID = auto()


class SourceTargetRelation(Enum):
    SOURCE_TARGET_DISTINCT = auto()
    SOURCE_TARGET_SAME = auto()


@dataclass
class OperationInputProfile:
    signature_shape: SignatureShape
    proof_shape: ProofShape
    lookup_shape: LookupShape
    source_address_shape: SourceAddressShape
    source_target_relation: SourceTargetRelation


p: OperationInputProfile = ...
