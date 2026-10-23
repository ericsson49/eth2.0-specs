"""Finality-gadget attestation validation and first-credit behavior."""

from tests.generators.compliance_runners.state_transition.eip8453.operation_target import (
    operation_target,
)

from .observation import observe_attributes

VALID = ("CURRENT", "OLD", "NO_FINALITY")
TARGET = operation_target(
    "attestation",
    {
        "case": (*VALID, "UNKNOWN", "MALFORMED", "FUTURE", "STALE", "BITS", "EMPTY", "SIGNATURE"),
        "timeout": bool,
        "progress_recorded": bool,
        "finality_recorded": bool,
    },
    lambda x: (x["case"] == "CURRENT") | (x["case"] == "OLD") | (x["case"] == "NO_FINALITY"),
    observe_attributes,
)
