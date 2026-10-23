"""Signed slashing operation materializer."""

from tests.generators.compliance_runners.state_transition.eip8453.slashing import (
    SlashingMaterializer,
)

from .target import TARGET


class Materializer(SlashingMaterializer):
    handler_name = input_name = "legacy_attester_slashing"
    test_provider = "eip8453_legacy_attester_slashing"
    target = TARGET
    legacy = True


MATERIALIZER = Materializer
