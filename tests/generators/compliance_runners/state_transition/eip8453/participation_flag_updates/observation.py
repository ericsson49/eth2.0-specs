"""Read prior round flags and height progress independently of spec helpers."""

from tests.generators.compliance_runners.state_transition.eip8453.observation import (
    focused_validator,
    is_active,
)


def observe_attributes(ctx):
    index, validator = focused_validator(ctx)
    spec, state = ctx.spec, ctx.pre
    flags = int(state.round_participation[index])
    return {
        "active": is_active(spec, state, validator),
        "slashed": bool(validator.slashed),
        "old_progress": bool(flags & (1 << int(spec.PROGRESS_FLAG_INDEX))),
        "old_other_flags": bool(flags & ~(1 << int(spec.PROGRESS_FLAG_INDEX))),
        "height_progress": bool(state.progress_participation[index]),
    }
