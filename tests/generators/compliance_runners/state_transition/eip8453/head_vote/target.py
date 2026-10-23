"""Head-vote acceptance and seat-recording coverage."""

from tests.generators.compliance_runners.state_transition.eip8453.operation_target import (
    operation_target,
)

from .observation import observe_attributes

TARGET = operation_target(
    "head_vote",
    {
        "timing": ("GENESIS", "PREVIOUS", "CURRENT", "TOO_OLD", "TOO_EARLY"),
        "root_relation": ("SAME_SLOT", "SKIPPED", "OTHER"),
        "voters": ("EMPTY", "ONE", "MULTIPLE"),
        "already_recorded": bool,
        "signature_valid": bool,
    },
    lambda x: (
        (x["timing"] != "TOO_OLD")
        & (x["timing"] != "TOO_EARLY")
        & (x["voters"] != "EMPTY")
        & x["signature_valid"]
    ),
    observe_attributes,
)
