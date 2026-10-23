"""Input coverage for EIP-8453 process_height_events.

Spec: specs/_features/eip8453/beacon-chain.md#new-process_height_events.
Outcome dimensions below are predictions from inputs, not post-state checks.
"""

from tests.generators.compliance_runners.state_transition.evaluation.declarations import (
    aspect,
    attribute,
    bind,
    choose,
    cmp3,
    cmp5,
    constant,
    count,
    coverage_spec,
    coverage_value,
    dimension,
    each,
    exhaustive,
    implies,
    Integer,
    nwise,
    union,
)

from .observation import observe_attributes

target_weight = attribute("target_weight", Integer(min=0))
progress_weight = attribute("progress_weight", Integer(min=0))
total_weight = attribute("total_weight", Integer(min=0))
timeout_delta = attribute("timeout_delta", Integer())
pending_count = attribute("pending_count", Integer(min=0))
qualifying_count = attribute("qualifying_count", Integer(min=0))
retained_count = attribute("retained_count", Integer(min=0))
window = constant("window", Integer(min=1))

target_positive = dimension("target_positive", target_weight > 0)
progress_positive = dimension("progress_positive", progress_weight > 0)
target_boundary = dimension(
    "target_boundary",
    cmp3(target_weight + target_weight + target_weight, total_weight + total_weight, op=">="),
)
progress_boundary = dimension(
    "progress_boundary",
    cmp3(progress_weight + progress_weight + progress_weight, total_weight + total_weight, op=">="),
)
timeout = dimension("timeout", cmp5(timeout_delta, 0, op=">="))
target_quorum = (target_weight > 0) & (
    target_weight + target_weight + target_weight >= total_weight + total_weight
)
progress_quorum = (progress_weight > 0) & (
    progress_weight + progress_weight + progress_weight >= total_weight + total_weight
)
advance = dimension(
    "advance_reason",
    choose(
        (target_quorum, "JUSTIFICATION"),
        (progress_quorum & (timeout_delta >= 0), "TIMEOUT"),
        "NONE",
    ),
)
pending = dimension("pending", count(pending_count))
qualifying = dimension("qualifying", count(qualifying_count))
retained = dimension("retained", count(retained_count))
full = dimension("window_full_after_pruning", retained_count == window)
evict = dimension("evict_oldest", target_quorum & (retained_count == window))

QUORUM = aspect("quorum", target_positive, target_boundary, progress_positive, progress_boundary)
ADVANCEMENT = aspect("advancement", advance, timeout)
FINALIZATION = aspect("finalization", pending, qualifying, retained, full, evict)
ASPECTS = (QUORUM, ADVANCEMENT, FINALIZATION)
FACTORS = tuple(f for a in ASPECTS for f in a.declarations)
STANDARD = union(
    nwise(FACTORS, 2),
    exhaustive(
        (target_positive, target_boundary, progress_positive, progress_boundary, timeout, advance)
    ),
    FINALIZATION.exhaustive(),
)


def constant_feasibility(bound):
    def bucket(n):
        return "ZERO" if n == 0 else "ONE" if n == 1 else "MANY"

    combinations = {
        (bucket(n), bucket(q), bucket(r), r == bound["window"])
        for n in range(bound["window"] + 1)
        for q in range(n + 1)
        for r in (range(n - q + 1) if q else (n,))
    }

    def feasible(assignment):
        return (
            tuple(
                assignment[k]
                for k in ("pending", "qualifying", "retained", "window_full_after_pruning")
            )
            in combinations
        )

    return feasible


COVERAGE = coverage_spec(
    "eip8453_height_events",
    focus="process_height_events: quorum, pruning, justification, timeout, and window eviction",
    record="one direct process_height_events call; pre-state is after all block attestations",
    aspects=ASPECTS,
    constant_feasibility=constant_feasibility,
    constraints=(
        implies(~target_positive & (coverage_value(target_boundary) == "EQ"), qualifying == "ZERO"),
        advance
        == choose(
            (target_positive & target_boundary, "JUSTIFICATION"),
            (progress_positive & progress_boundary & timeout, "TIMEOUT"),
            "NONE",
        ),
        evict == (target_positive & target_boundary & full),
        implies(target_positive, progress_positive),
        implies(~target_positive, coverage_value(target_boundary) != "GT"),
        implies(~progress_positive, coverage_value(progress_boundary) != "GT"),
        implies(coverage_value(target_boundary) == "GT", coverage_value(progress_boundary) == "GT"),
        implies(coverage_value(target_boundary) == "EQ", coverage_value(progress_boundary) != "LT"),
        implies(
            ~target_positive & (coverage_value(target_boundary) == "EQ"),
            ~progress_positive & (coverage_value(progress_boundary) == "EQ"),
        ),
        implies(
            ~progress_positive, coverage_value(target_boundary) == coverage_value(progress_boundary)
        ),
    ),
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
    constants={"window": lambda spec: int(spec.PENDING_FINALITY_WINDOW)},
    backend="python",
)
