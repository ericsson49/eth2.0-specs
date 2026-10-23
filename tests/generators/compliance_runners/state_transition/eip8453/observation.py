"""Explicit focus selection for per-validator EIP-8453 observations."""


def focused_validator(ctx):
    """Require an index for multi-validator vectors instead of silently picking one."""
    index = ctx.meta.get("validator_index")
    if index is None and len(ctx.pre.validators) == 1:
        index = 0
    if type(index) is not int or not 0 <= index < len(ctx.pre.validators):
        raise ValueError("validator_index must select an existing validator")
    return index, ctx.pre.validators[index]


def is_active(spec, state, validator):
    epoch = int(state.slot) // int(spec.SLOTS_PER_EPOCH)
    return int(validator.activation_epoch) <= epoch < int(validator.exit_epoch)
