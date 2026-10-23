"""Input coverage for EIP-8453 process_inactivity_updates, one validator.

Spec: specs/_features/eip8453/beacon-chain.md#modified-process_inactivity_updates.
Decisions are input-derived expectations, not observations of execution.
"""

from tests.generators.compliance_runners.state_transition.evaluation.declarations import (
    aspect,
    attribute,
    bind,
    Boolean,
    choose,
    cmp5,
    constant,
    coverage_spec,
    dimension,
    each,
    exhaustive,
    Integer,
)

from .observation import observe_attributes

active_input = attribute("active", Boolean())
slashed_input = attribute("slashed", Boolean())
flag_input = attribute("progress_flag", Boolean())
score_input = attribute("score", Integer(min=0))
exit_input = attribute("exit_set", Boolean())
recovery = constant("recovery", Integer(min=1))
bias = constant("bias", Integer(min=1))
ejection = constant("ejection", Integer(min=1))

active = dimension("active_validator", active_input)
slashed = dimension("validator_slashed", slashed_input)
flag = dimension("round_progress_flag", flag_input)
exit_set = dimension("exit_already_set", exit_input)
score_zero = dimension("initial_score_zero", score_input == 0)
recover_boundary = dimension("score_vs_recovery", cmp5(score_input, recovery, op=">="))
eject_boundary = dimension(
    "increased_score_vs_ejection", cmp5(score_input + bias, ejection, op=">=")
)
mode = dimension(
    "score_action",
    choose((~active_input, "UNCHANGED"), (~slashed_input & flag_input, "RECOVER"), "INCREASE"),
)
exit_call = dimension(
    "exit_requested",
    active_input & (slashed_input | ~flag_input) & (score_input + bias >= ejection),
)
new_exit = dimension("new_exit", exit_call.value & ~exit_input)

LIFECYCLE = aspect("lifecycle", active, slashed, exit_set)
PARTICIPATION = aspect("participation", flag)
ARITHMETIC = aspect("arithmetic", score_zero, recover_boundary, eject_boundary)
DECISIONS = aspect("decisions", mode, exit_call, new_exit)
ASPECTS = (LIFECYCLE, PARTICIPATION, ARITHMETIC, DECISIONS)
FACTORS = tuple(f for a in ASPECTS for f in a.declarations)


def constant_feasibility(bound):
    # Each cmp5 partition changes only at threshold +/- 1. These representatives
    # include every nonnegative interval for both comparisons, without a score cap.
    thresholds = (bound["recovery"], bound["ejection"] - bound["bias"])
    scores = {0} | {
        t + offset for t in thresholds for offset in (-2, -1, 0, 1, 2) if t + offset >= 0
    }

    def bucket(delta):
        return (
            "LT_FAR"
            if delta < -1
            else "LT_1"
            if delta == -1
            else "EQ"
            if delta == 0
            else "GT_1"
            if delta == 1
            else "GT_FAR"
        )

    pairs = {(s == 0, bucket(s - thresholds[0]), bucket(s - thresholds[1])) for s in scores}
    return lambda row: (
        (row["initial_score_zero"], row["score_vs_recovery"], row["increased_score_vs_ejection"])
        in pairs
    )


COVERAGE = coverage_spec(
    "eip8453_inactivity_updates",
    focus="process_inactivity_updates: active/unslashed participation, saturation, and ejection",
    record="one validator selected by Context.meta['validator_index']",
    aspects=ASPECTS,
    constant_feasibility=constant_feasibility,
    constraints=(
        mode == choose((~active, "UNCHANGED"), (~slashed & flag, "RECOVER"), "INCREASE"),
        exit_call == (active & (slashed | ~flag) & eject_boundary),
        new_exit == (exit_call & ~exit_set),
    ),
    profiles={
        "smoke": each(FACTORS),
        "normal": exhaustive(FACTORS),
        "standard": exhaustive(FACTORS),
        "max": exhaustive(FACTORS),
    },
)
TARGET = bind(
    COVERAGE,
    observe_attributes=observe_attributes,
    backend="python",
    constants={
        "recovery": lambda spec: int(spec.config.INACTIVITY_SCORE_RECOVERY_RATE),
        "bias": lambda spec: int(spec.config.INACTIVITY_SCORE_BIAS),
        "ejection": lambda spec: int(spec.EJECTION_INACTIVITY_SCORE),
    },
)
