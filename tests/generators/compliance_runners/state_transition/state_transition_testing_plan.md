# State Transition Test Generation Plan

This document tracks the current state-transition test-generation approach,
what is already implemented, and the main remaining extensions.

## Goal

Generate reproducible state-transition reference vectors from compact abstract
models, then evaluate them with both code coverage and semantic coverage.

The current implementation focuses on Electra/Fulu-style state transition
testing, with Electra used as the first concrete target.

## Current Approach

The generator has four layers:

1. **Abstract case generation**
   - A small Python subset is transpiled to MiniZinc.
   - MiniZinc solutions describe abstract validator/state profiles.
   - Guided ontology intents expand those profiles into targeted cases.

2. **Materialization**
   - Abstract cases are converted into concrete reference-test vectors.
   - Outputs use existing `operations` and `epoch_processing` formats:
     `pre`, operation input when applicable, `post`, and `meta.yaml`.
   - Positive, assertion-failure, and no-state-change cases are supported.

3. **Validation**
   - Generated vectors are executed against the pyspec.
   - The runner checks that the resulting post-state matches the emitted
     `post`, or that assertion-failure cases raise.

4. **Coverage and reporting**
   - `coverage.py` provides source and branch coverage.
   - Ontology target coverage reports coverage of relevant spec functions.
   - Semantic coverage checks generated intents against expected outcomes.
   - Suite summaries report local suite health.
   - Campaign summaries aggregate multiple suites under one ontology-level
     coverage report.

## Implemented Coverage

### Operations

Implemented operation materializers:

- `deposit`
- `deposit_request`
- `withdrawal_request`
- `consolidation_request`
- `voluntary_exit`
- `bls_to_execution_change`
- `proposer_slashing`
- `attester_slashing`
- `attestation`

### Validator Lifecycle Stage

Implemented stage: `validator_lifecycle`

Handlers:

- `registry_updates`
- `slashings`
- `effective_balance_updates`
- `pending_deposits`
- `pending_consolidations`

This stage focuses on activation, ejection, slashing penalties, effective-balance
hysteresis, pending deposit queues, and pending consolidation queues.

### Participation and Finality Stage

Implemented stage: `participation_finality`

Handlers:

- `justification_and_finalization`
- `inactivity_updates`
- `rewards_and_penalties`
- `participation_flag_updates`

This stage focuses on finality movement, inactivity leak behavior,
participation flags, rewards, penalties, and participation flag rotation.

### Rotating and Reset Stage

Implemented stage: `rotating_resets`

Handlers:

- `slashings_reset`
- `randao_mixes_reset`
- `eth1_data_reset`
- `historical_summaries_update`

This stage focuses on epoch-indexed ring buffers, reset boundaries, and
historical accumulator updates. It covers both boundary and non-boundary cases
where applicable.

### Tooling

Implemented:

- deterministic suite configs
- campaign configs over multiple suites
- aggregate campaign coverage over multiple generated directories
- semantic intent/outcome reporting
- target-function coverage reporting
- suite and campaign summaries
- reproducibility checks
- deterministic distribution controls for generated suite shape
- stage-aware campaign summaries
- observed pairwise semantic interaction coverage

### Committee and Sync Stage

Implemented stage: `committee_sync`

Handlers:

- `sync_committee_updates`
- `sync_aggregate`

Purpose:

- Cover sync committee period boundaries.
- Cover current/next committee rotation.
- Cover sync aggregate reward and penalty behavior.

The stage intentionally mixes an epoch-processing handler with an operation
handler. `sync_committee_updates` covers period boundary rotation. The
`sync_aggregate` operation covers all, majority, minority, empty, and invalid
signature aggregate cases.

## Remaining Stages

### Block and Execution Stage

Potential handlers and sub-transitions:

- `process_block_header`
- `process_randao`
- `process_eth1_data`
- `process_withdrawals`
- `process_execution_payload`
- selected `process_block` paths

Purpose:

- Extend beyond isolated operation and epoch-processing handlers.
- Cover block-level interactions and execution payload consistency checks.

This stage is larger and should probably be split further before
implementation.

## Coverage Extensions

### Current Coverage Types

Current reporting includes:

- Source/branch coverage via `coverage.py`.
- Target-function coverage from ontology target declarations.
- Semantic intent coverage:
  `runner / handler / intent -> expected outcome`.
- Suite/campaign shape summaries.

### Semantic Interaction Coverage

Semantic interaction coverage is a planned extension. The idea is to measure
coverage of meaningful pairs of semantic dimensions, not only individual
intents.

Examples:

- `handler x outcome`
- `stage x outcome`
- `intent x outcome`
- `withdrawal_credential_type x outcome`
- `finality_leak_state x participation_state`
- `validator_lifecycle_state x balance_relation`
- `queue_pressure x churn_limit_state`

This can help answer whether generated tests cover interactions between
important concepts, instead of only covering each concept independently.

#### Possible Modes

1. **Observed pair coverage**
   - Report every pair observed in generated vectors.
   - Useful for exploration and debugging model shape.

2. **Expected pair coverage**
   - Ontology declares expected pairs.
   - Missing expected pairs become actionable report items.

3. **Constrained pair coverage**
   - Only count meaningful or feasible pairs.
   - Avoids reporting impossible combinations as missing coverage.

4. **Per-stage pair coverage**
   - Compute pairs within a stage-specific semantic space.
   - Example for `participation_finality`:
     `leak_state x participation_state`.
   - Example for `validator_lifecycle`:
     `validator_status x balance_relation`.

#### Possible Ontology Shape

```yaml
coverage_dimensions:
  common:
    - handler
    - intent
    - outcome
    - stage

  validator_lifecycle:
    - withdrawal_credential_type
    - slashed
    - exit_epoch_set
    - balance_to_effective_balance

  participation_finality:
    - finality_leak_state
    - participation_state
    - inactivity_score_state

expected_pairs:
  participation_finality:
    - [finality_leak_state, participation_state]
    - [participation_state, outcome]
  validator_lifecycle:
    - [withdrawal_credential_type, outcome]
    - [slashed, outcome]
```

The first implementation should probably start with observed pair coverage,
then add expected/constrained pairs once the useful dimensions are clearer.

## Suggested Next Steps

1. Decide which observed interaction pairs should become expected ontology
   targets.
2. Extend observed interaction reporting to sampled triples once randomized
   generation starts using interaction targets.
3. Start splitting the block/execution stage into smaller sub-stages.

## Open Questions

- Should stage-specific models eventually replace the single validator-state
  model for epoch-processing stages?
- How much overlap between suites should campaigns allow before deduplication
  is useful?
- Should campaigns report both raw vectors and unique semantic intents?
- Which semantic dimensions are stable enough to encode in ontology now?
