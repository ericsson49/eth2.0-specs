"""Recover slashing inputs independently of processing."""

from tests.generators.compliance_runners.state_transition.eip8453.slashing import observe


def observe_attributes(ctx):
    return observe(ctx, legacy=False)
