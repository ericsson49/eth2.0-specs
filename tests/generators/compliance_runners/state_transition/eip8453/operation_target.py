"""Common profile construction for finite operation targets."""

from tests.generators.compliance_runners.state_transition.evaluation.declarations import (
    aspect,
    assertion,
    assertion_set,
    attribute,
    bind,
    Boolean,
    coverage_spec,
    dimension,
    each,
    exhaustive,
    Finite,
    fix,
)


def operation_target(name, fields, accepted, observer):
    inputs = {
        key: attribute("observed_" + key, Boolean() if values is bool else Finite(values))
        for key, values in fields.items()
    }
    factors = [dimension(key, value) for key, value in inputs.items()]
    factors.append(assertion("inputs_valid", accepted(inputs)))
    factors.append(dimension("accepted", attribute("post_present", Boolean())))
    group = aspect("operation", *factors)
    return bind(
        coverage_spec(
            "eip8453_" + name,
            focus="process_" + name,
            record="one operation and its immediate pre-state; finite local-handler model",
            aspects=(group,),
            constraints=(
                factors[-2] == accepted(dict(zip(fields, factors[:-2], strict=True))),
                factors[-1] == factors[-2],
            ),
            assertions=assertion_set(group, observed=group.ref("accepted"), complete=True),
            profiles={
                "smoke": each(factors),
                "normal": fix(accepted=True) * exhaustive(factors),
                "exceptional": fix(accepted=False) * exhaustive(factors),
                "standard": exhaustive(factors),
                "max": exhaustive(factors),
            },
        ),
        observe_attributes=lambda ctx: {**observer(ctx), "post_present": ctx.post is not None},
        backend="python",
    )
