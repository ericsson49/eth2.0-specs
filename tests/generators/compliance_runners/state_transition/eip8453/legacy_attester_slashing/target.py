"""EIP-8453 legacy_attester_slashing coverage."""

from tests.generators.compliance_runners.state_transition.eip8453.slashing import target

from .observation import observe_attributes

TARGET = target(legacy=True, observer=observe_attributes)
