"""Read inactivity inputs without invoking participation or inactivity helpers."""

from tests.generators.compliance_runners.state_transition.eip8453.observation import (
    focused_validator,
    is_active,
)


def observe_attributes(ctx):
    index, validator = focused_validator(ctx)
    spec, state = ctx.spec, ctx.pre
    return {
        "active": is_active(spec, state, validator),
        "slashed": bool(validator.slashed),
        "progress_flag": bool(
            int(state.round_participation[index]) & (1 << int(spec.PROGRESS_FLAG_INDEX))
        ),
        "score": int(state.inactivity_scores[index]),
        "exit_set": int(validator.exit_epoch) != int(spec.FAR_FUTURE_EPOCH),
    }
