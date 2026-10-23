"""Changed body-level guards and dispatch in process_operations."""

from tests.generators.compliance_runners.state_transition.eip8453.operation_target import (
    operation_target,
)

from .observation import observe_attributes

CASES = (
    "EMPTY",
    "HEAD",
    "OLD_UNSUPPORTED",
    "OLD_SUPPORTED",
    "SHORT_SUPPORT",
    "LONG_SUPPORT",
    "MAX_HEAD",
    "OVER_HEAD",
    "LEGACY",
    "OVER_LEGACY",
    "ATTESTATION",
    "FINALITY_SLASHING",
    "MIXED",
)
TARGET = operation_target(
    "operations",
    {"case": CASES},
    lambda x: (
        (x["case"] != "OLD_SUPPORTED")
        & (x["case"] != "SHORT_SUPPORT")
        & (x["case"] != "LONG_SUPPORT")
        & (x["case"] != "OVER_HEAD")
        & (x["case"] != "OVER_LEGACY")
    ),
    observe_attributes,
)
