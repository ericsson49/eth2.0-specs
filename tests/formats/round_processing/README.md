# Round processing tests

The round sub-transitions are tested individually with test handlers. Each
sub-transition takes only a `BeaconState` as input, so the format is shared
between handlers, as in [epoch processing](../epoch_processing/README.md).

## Test case format

### `meta.yaml`

```yaml
description: string    -- Optional description of test case, purely for debugging purposes.
                          Tests should use the directory name of the test case as identifier, not the description.
bls_setting: int       -- see general test-format spec.
```

### `pre.ssz_snappy`

An SSZ-snappy encoded `BeaconState`, the state immediately before running the
round sub-transition of focus.

### `post.ssz_snappy`

An SSZ-snappy encoded `BeaconState`, the state after applying the round
sub-transition. No value if the sub-transition processing is aborted.

### `pre_round.ssz_snappy`

Optional. An SSZ-snappy encoded `BeaconState`, the state before running the
entire round transition.

Some test cases mutate the state between sub-transitions to exercise a specific
one in isolation. If this is the case, this file is omitted, since
`process_round(pre_round)` would not reproduce the mutated state.

### `post_round.ssz_snappy`

Optional. An SSZ-snappy encoded `BeaconState`, the state after applying the
entire round transition. No value if the transition processing is aborted.
Omitted in the same circumstances as `pre_round.ssz_snappy`.

## Condition

A handler of the `round_processing` test-runner should call the corresponding
processing implementation (same name, prefixed with `process_`). This excludes
the other parts of the round transition. The provided pre-state is already
transitioned to just before the specific sub-transition of focus.

Sub-transitions, in execution order:

- `inactivity_updates` (EIP-8453)
- `rewards` (EIP-8453)
- `participation_flag_updates` (EIP-8453)

The resulting state should match the expected `post` state in its entirety. An
absent `post` requires the sub-transition to abort.

Round processing is invoked by `process_slots` after `process_slot`, when
`(state.slot + 1) % SLOTS_PER_ROUND == 0`, before incrementing `state.slot`.
When a round boundary coincides with an epoch boundary, round processing runs
before epoch processing. Replaying an individual sub-transition must neither
increment the slot nor execute the enclosing slot or epoch transitions.

Handler names may also exist under `epoch_processing` on other forks. The runner
name determines which transition is being tested; these round handlers require a
fork that supports `process_round`.

## Condition (alternative)

Instead of having a different handler for each sub-transition, a single handler
for all cases may load `pre_round`, call `process_round`, and compare the result
with `post_round`. Because these files are optional, this handler must skip
cases that do not provide `pre_round`. An absent `post_round` for a provided
`pre_round` requires the round transition to abort.

This alternative supports single-pass round implementations and can detect
dependencies between inactivity updates, rewards, and participation flag
updates. It requires a constant amount of additional work per vector.
