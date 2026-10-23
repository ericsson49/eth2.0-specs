"""Coverage of the EIP-8453 head-vote payment rewrite, focused on slot zero."""

from tests.generators.compliance_runners.state_transition.evaluation.declarations import (
    aspect,
    attribute,
    bind,
    Boolean,
    cmp5,
    constant,
    coverage_spec,
    dimension,
    each,
    exhaustive,
    Integer,
    nwise,
    union,
)

from .observation import observe_attributes

amount = dimension("amount_nonzero", attribute("amount", Boolean()))
seats = attribute("seats", Integer(min=0))
threshold = constant("required_seats", Integer(min=1))
quorum = dimension("seats_vs_required", cmp5(seats, threshold, op=">="))
legacy = dimension("legacy_weight_nonzero", attribute("legacy_weight", Boolean()))
tail = dimension("tail_payment", attribute("observed_tail_payment", Boolean()))
next_payments = dimension("next_payments", attribute("observed_next_payments", Boolean()))
next_votes = dimension("next_votes", attribute("observed_next_votes", Boolean()))
existing = dimension("existing_withdrawals", attribute("observed_existing_withdrawals", Boolean()))
GATE = aspect("head_vote_payment", amount, quorum, legacy)
ROTATION = aspect("rotation_and_append", tail, next_payments, next_votes, existing)
ASPECTS = (GATE, ROTATION)
FACTORS = tuple(f for a in ASPECTS for f in a.declarations)
STANDARD = union(nwise(FACTORS, 2), GATE.exhaustive(), ROTATION.exhaustive())
COVERAGE = coverage_spec(
    "eip8453_builder_pending_payments",
    focus="process_builder_pending_payments: seat quorum, zero amount, append order and buffer rotation",
    record="one epoch; payment and head-vote slot zero is the focal slot",
    aspects=ASPECTS,
    profiles={
        "smoke": each(FACTORS),
        "normal": STANDARD,
        "standard": STANDARD,
        "max": exhaustive(FACTORS),
    },
)
TARGET = bind(
    COVERAGE,
    observe_attributes=observe_attributes,
    backend="python",
    constants={
        "required_seats": lambda s: (
            (
                int(s.HEAD_VOTE_COMMITTEE_SIZE) * int(s.BUILDER_PAYMENT_THRESHOLD_NUMERATOR)
                + int(s.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR)
                - 1
            )
            // int(s.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR)
        ),
    },
)
