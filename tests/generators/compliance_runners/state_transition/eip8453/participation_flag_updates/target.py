"""Input coverage for EIP-8453 process_participation_flag_updates.

Spec: specs/_features/eip8453/beacon-chain.md#modified-process_participation_flag_updates.
Activity and slashing must not filter progress carry-over. Effects are predicted.
"""

from tests.generators.compliance_runners.state_transition.evaluation.declarations import (
    aspect,
    attribute,
    bind,
    Boolean,
    choose,
    coverage_spec,
    dimension,
    each,
    exhaustive,
)

from .observation import observe_attributes

active_input = attribute("active", Boolean())
slashed_input = attribute("slashed", Boolean())
old_progress_input = attribute("old_progress", Boolean())
old_other_input = attribute("old_other_flags", Boolean())
carry_input = attribute("height_progress", Boolean())
active = dimension("active_validator", active_input)
slashed = dimension("validator_slashed", slashed_input)
old_progress = dimension("previous_progress_flag", old_progress_input)
old_other = dimension("previous_other_flags", old_other_input)
carry = dimension("carry_progress", carry_input)
effect = dimension(
    "expected_effect",
    choose(
        ((old_progress_input == carry_input) & ~old_other_input, "UNCHANGED"),
        (carry_input, "PROGRESS_ONLY"),
        "CLEARED",
    ),
)

LIFECYCLE = aspect("unfiltered_lifecycle", active, slashed)
FLAGS = aspect("round_flags", old_progress, old_other, carry, effect)
ASPECTS = (LIFECYCLE, FLAGS)
FACTORS = tuple(f for a in ASPECTS for f in a.declarations)
COVERAGE = coverage_spec(
    "eip8453_participation_flag_updates",
    focus="process_participation_flag_updates: clear prior flags and restore height progress for every validator",
    record="one validator selected by Context.meta['validator_index']",
    aspects=ASPECTS,
    constraints=(
        effect
        == choose(
            ((old_progress == carry) & ~old_other, "UNCHANGED"), (carry, "PROGRESS_ONLY"), "CLEARED"
        ),
    ),
    profiles={
        "smoke": each(FACTORS),
        "normal": exhaustive(FACTORS),
        "standard": exhaustive(FACTORS),
        "max": exhaustive(FACTORS),
    },
)
TARGET = bind(COVERAGE, observe_attributes=observe_attributes, backend="python")
