from eth2spec.test.helpers.constants import (
    MINIMAL,
)
from eth2spec.test.context import (
    spec_state_test,
    with_electra_and_later,
    with_presets,
    single_phase,
)
from eth2spec.test.helpers.validator_state_profiles import validator_state_profiles
from eth2spec.test.helpers.validator_state import (
    prepare_validator_and_beacon_state,
    get_validator_state_profile,
)
from eth2spec.test.helpers.state import (
    next_epoch_with_full_participation,
)


@with_electra_and_later
@spec_state_test
@with_presets([MINIMAL], reason="too many tests")
@single_phase
def test_basic_validator_state_computation(spec, state):
    # skip 4 epochs to finalize epoch > GENESIS_EPOCH
    for _ in range(4):
        next_epoch_with_full_participation(spec, state)

    clean_state = state
    for i, profile in enumerate(validator_state_profiles):
        state = clean_state.copy()

        validator_index = i % len(state.validators)
        prepare_validator_and_beacon_state(spec, state, profile, validator_index)
        computed_profile = get_validator_state_profile(spec, state, validator_index)

        assert computed_profile == profile, f"Computed profile {i} doesn't match"
