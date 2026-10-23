"""EIP-8453 exact-slot finalization gate and its queue/churn interactions.

This target focuses on existing-validator deposits; inherited new-validator
signature validation and registry insertion are not claimed as covered here.
"""

from tests.generators.compliance_runners.state_transition.evaluation.declarations import (
    aspect,
    attribute,
    bind,
    Boolean,
    cmp3,
    coverage_spec,
    dimension,
    each,
    exhaustive,
    Finite,
    Integer,
    nwise,
    union,
)

from .observation import observe_attributes

slot = dimension(
    "deposit_vs_finalized_slot",
    cmp3(
        attribute("deposit_slot", Integer(min=0)),
        attribute("finalized_slot", Integer(min=0)),
        op="<=",
    ),
)
position = dimension(
    "finalized_position", attribute("position", Finite(("START", "INTERIOR", "END")))
)
role = dimension("primary_role", attribute("role", Finite(("ACTIVE", "EXITING", "WITHDRAWN"))))
amount = dimension(
    "amount_vs_churn",
    cmp3(attribute("amount", Integer(min=0)), attribute("available", Integer(min=1)), op="<="),
)
carry = dimension("carry_nonzero", attribute("carry", Boolean()))
suffix = dimension(
    "suffix", attribute("observed_suffix", Finite(("NONE", "FINALIZED", "UNFINALIZED")))
)
FINALIZATION = aspect("finalization", slot, position)
QUEUE = aspect("queue", role, suffix)
CHURN = aspect("churn", amount, carry)
ASPECTS = (FINALIZATION, QUEUE, CHURN)
FACTORS = tuple(f for a in ASPECTS for f in a.declarations)
STANDARD = union(
    nwise(FACTORS, 2),
    exhaustive((slot, position, role, amount)),
    exhaustive((slot, role, suffix, carry)),
)
COVERAGE = coverage_spec(
    "eip8453_pending_deposits",
    focus="process_pending_deposits: exact finalized slot and queue/churn interactions",
    record="first pending deposit belongs to an existing validator; optional suffix",
    aspects=ASPECTS,
    profiles={
        "smoke": each(FACTORS),
        "normal": STANDARD,
        "standard": STANDARD,
        "max": exhaustive(FACTORS),
    },
)
TARGET = bind(COVERAGE, observe_attributes=observe_attributes, backend="python")
