import random
import pytest
from eth2spec.test.helpers.constants import (
    MINIMAL,
)
from eth2spec.test.context import (
    spec_state_test,
    with_electra_and_later,
    with_presets,
    spec_state_test,
    single_phase,
)
from eth2spec.test.helpers.validator_state_profiles import validator_state_profiles
from eth2spec.test.helpers.validator_state import prepare_validator_and_beacon_state



@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], reason="too many tests")
@single_phase
def test_validator_state_withdrawal_request(spec, state):
    validator_index = 0
    pre_state = state.copy()
    for i in range(100, 110):
        state = pre_state.copy()
        prepare_validator_and_beacon_state(spec, state, validator_state_profiles[i], validator_index)
        print(state.validators[validator_index])

    assert False

