"""Coverage of EIP-8453 process_rewards and its flag-reward gates.

Spec: specs/_features/eip8453/beacon-chain.md#new-process_rewards.
Eligibility is input-derived; integer truncation can still produce zero rewards.
"""

from tests.generators.compliance_runners.state_transition.evaluation.declarations import (
    aspect,
    attribute,
    bind,
    Boolean,
    cmp5,
    constant,
    count,
    coverage_spec,
    dimension,
    each,
    exhaustive,
    Integer,
    nwise,
    union,
)

from .observation import observe_attributes

active_input = attribute("active", Boolean())
slashed_input = attribute("slashed", Boolean())
progress_input = attribute("progress", Boolean())
finality_input = attribute("finality", Boolean())
increments = attribute("increments", Integer(min=0))
other_progress_input = attribute("other_progress", Boolean())
other_finality_input = attribute("other_finality", Boolean())
delay = attribute("finality_delay", Integer(min=0))
leak_threshold = constant("leak_threshold", Integer(min=0))
active = dimension("active_validator", active_input)
slashed = dimension("validator_slashed", slashed_input)
progress = dimension("progress_flag", progress_input)
finality = dimension("finality_flag", finality_input)
balance = dimension("balance_increments", count(increments))
other_progress = dimension("other_progress_participants", other_progress_input)
other_finality = dimension("other_finality_participants", other_finality_input)
leak = dimension("delay_vs_leak", cmp5(delay, leak_threshold, op=">"))

LIFECYCLE = aspect("eligibility", active, slashed)
FLAGS = aspect("flags", progress, finality)
BALANCE = aspect("balance", balance)
POPULATION = aspect("other_participants", other_progress, other_finality)
LEAK = aspect("leak", leak)
ASPECTS = (LIFECYCLE, FLAGS, BALANCE, POPULATION, LEAK)
FACTORS = tuple(f for a in ASPECTS for f in a.declarations)
STANDARD = union(
    nwise(FACTORS, 2), exhaustive((active, slashed, progress, finality, balance, leak))
)
COVERAGE = coverage_spec(
    "eip8453_rewards",
    focus="process_rewards: eligibility, both flags, effective balance, population, and leak boundary",
    record="validator zero and the other validators in one round reward vector",
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
    constants={"leak_threshold": lambda spec: int(spec.MIN_ROUNDS_TO_INACTIVITY_LEAK)},
)
