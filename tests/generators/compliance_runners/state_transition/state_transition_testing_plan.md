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

## Ontological Lens

The state-transition functions under test have two basic execution shapes:

- Epoch-processing handlers: `State -> State`
- Operation handlers: `(State, Input) -> State`

The test-generation ontology describes those executions with a small set of
related concepts:

- **Profile**: an abstract state aspect. A profile is a projection of the full
  `BeaconState` into facts relevant to a behavior area, such as validator
  lifecycle status, balance relations, credentials, queue state, participation
  state, or sync committee boundary state.
- **Stage**: a behavioral aspect or protocol area. A stage groups handlers that
  together express one part of the state transition, such as validator
  lifecycle, participation/finality, rotating resets, or committee/sync
  behavior.
- **Handler**: the concrete spec function under test. Each generated case calls
  one handler, either directly as an epoch-processing function or with an
  operation input.
- **Intent**: a semantic behavior class for a handler. Intents describe the
  behavior the test is trying to exercise, such as `queue_full`,
  `bad_signature`, `success_partial_withdrawal`, or `period_boundary`.
- **Outcome**: the coarse effect class of executing the handler. Current
  outcomes are `changed`, `no_change`, and `assertion_failure`.

These concepts answer different questions:

- Profile: what abstract state aspect is being materialized?
- Stage: which protocol behavior area owns this handler?
- Handler: which spec function is executed?
- Intent: which semantic behavior class is targeted?
- Outcome: what high-level effect is expected or observed?

An outcome can be viewed as effect-shaped semantic information, but it is kept
separate from intent because it describes the result rather than the behavior
goal. For example, `withdrawal_request / pubkey_missing -> no_change` and
`consolidation_request / source_missing -> no_change` have the same outcome but
different intents.

The current implementation assigns one primary intent to each generated case.
Real executions can touch multiple semantic facts at once, such as credential
type, queue capacity, active status, amount class, and outcome. The observed
interaction report is the first step toward measuring these combinations. Over
time, materializers and profiles can expose more semantic dimensions so
coverage can move from only `handler x intent` toward richer pairs or sampled
triples such as `handler x credential_type x outcome`.

Another useful view is to separate input-side knobs from output-side coverage
items.

Input-side concepts are the things generation can choose or construct before
execution:

- profiles and profile dimensions
- operation input shapes
- selected handler and stage context
- materializer parameters
- random seed and mutation choices

Output-side concepts are the things expected or observed after execution:

- semantic intents or observed behaviors
- outcomes
- touched target functions
- code statements and branches
- exceptions or post-state effects

Input coverage asks whether the generated suite covers the abstract state and
operation-input shapes we intended to construct. Output coverage asks whether
the suite triggers the intended behaviors, outcomes, and code coverage items.
Interaction coverage connects the two, for example `credential_type x outcome`,
`queue_state x branch`, or `handler x intent x outcome`.

This gives models three related roles:

1. **Define input coverage.** A model solution corresponds to an abstract test
   case or profile, and therefore to a point in an input coverage space.
2. **Approximate the input-output relation.** A model can estimate which input
   aspects are likely to trigger desired intents, outcomes, or branches.
3. **Guide input-output coverage.** Coverage feedback can identify missing
   input/output combinations, then models and materializers can try to produce
   inputs that cover them.

Intents are especially related to this inverse problem. An intent names a
desired behavior, then generation asks which state and input aspects should be
constructed to trigger it. Today that inverse map is mostly hand-coded in
materializers: `queue_full` fills a queue, `source_inactive` changes activation
status, `period_boundary` chooses a boundary slot, and `bad_signature` corrupts
a signature. Future models can make this more declarative by constraining or
scoring profiles that are expected to trigger the desired behavior. When the
model is approximate, the pyspec runner remains the oracle and coverage
feedback tells us whether the suggested inputs actually reached the target.

## Generation Modes

This gives two complementary generation modes.

**Simple generation** is input-first. It samples or enumerates input-side
aspects, such as profiles, operation input classes, handlers, stages, seeds, or
mutation choices. The generated vectors are then run against the pyspec, and
the resulting outcomes, semantic behaviors, and code coverage are observed.
This mode is useful for broad exploration and input coverage.

**Handler-touch generation** is the first shallow rung of simple generation. It
tries to produce one materialized vector for each handler in scope. Its goal is
not deep semantic coverage; it establishes that each handler can be generated,
serialized, executed, and measured.

**Profile-partition generation** is the next simple-generation rung. It still
uses the shared MiniZinc validator-state profile model, but it chooses cases to
cover values of configured input-side profile dimensions such as credential
type, lifecycle relation, balance relation, slashing status, and pending queue
flags. This gives deterministic input coverage before introducing semantic
intent targets.

**Profile-interaction generation** keeps the same input-first posture, but
samples combinations of profile dimensions. The first implemented form is
pairwise coverage, for example `withdrawal_credential_type x slashed` or
`exit_epoch_set x has_pending_withdrawal_request`. This is intended primarily
as an implementation-diversity rung: two states may touch similar spec code
while still stressing very different client cache keys, indexes, and fast
paths.

**Input-profile generation** promotes knobs learned from guided materializers
into small reusable MiniZinc models, such as operation input validity, queue
shape, epoch-boundary shape, and participation/finality shape. Each handler
declares the profile models that can affect it, and the sampler covers values
from those models without running the heavier pairwise interaction suite. This
is still input-first, but it reaches more protocol behavior because the sampled
inputs include operation and epoch-processing shapes, not only validator state.
The same mode can use `input_profile_order: 2` to sample pairwise combinations
over the handler-specific input profile dimensions. Pairwise and higher-order
input-profile sampling filters combinations against the solved MiniZinc profile
models, removing same-model tuples that cannot exist together while keeping
cross-model compatibility as a separate materialization concern. Case metadata
separates the sampled constraints from completed supporting profile rows, so
coverage reports can stay focused on the sampled target while materializers get
the fuller abstract input shape.

**Guided generation** is target-first. It samples an output-side coverage item
or an input-output interaction, then uses an inverse map to construct input
aspects likely to trigger it. Targets may be semantic intents, outcomes,
branches, target functions, or interactions such as `queue_state x branch`,
`profile_dimension x outcome`, or `handler x credential_type x outcome`.
Materializers currently implement much of this inverse map directly; future
models can make the map more declarative or approximate.

### Branch-Reachability Models

A useful future refinement is to model handler branching logic directly, but
only at the abstract guard level. These models should not try to translate the
full handler implementation, full SSA form, SSZ containers, hashes, BLS
verification, or exact post-state mutation. Instead, they should encode bounded
path conditions over profile dimensions.

In this view, MiniZinc is a target for branch-reachability models, not a target
for faithful handler transpilation:

```text
handler branch target
  -> abstract guard formula
  -> solved input-profile constraints
  -> concrete test vector
  -> real pyspec runner validates behavior
```

For example, a `withdrawal_request` branch model can describe the early-return
chain with predicates such as:

- partial-withdrawal queue is full
- request pubkey is present or missing
- credentials and source address match
- source validator is active
- source validator is not exiting
- source validator has been active long enough
- request is a full exit or a partial withdrawal
- partial withdrawal conditions are met

Each branch target becomes a small constraint formula over those predicates.
The materializer then realizes the solved predicates as a concrete pre-state
and operation input, while the pyspec runner remains the oracle for whether the
case actually reaches the expected outcome and code branch.

This keeps the modeling problem intentionally lean. The model approximates the
input-output relation enough to guide generation, and coverage feedback tells
us when the approximation or materializer needs to be refined. It also gives a
clean path from semantic coverage items to input-profile constraints without
embedding all of the branch recipes imperatively in materializers.

The first concrete prototypes apply this shape to `withdrawal_request` and
`consolidation_request`. Their input profiles include a `branch_target`
dimension for the relevant guard chain: queue guards, lookup failures,
credential/source-address checks, activity and exit checks, pending-request
checks, and success paths. For consolidation requests, the branch target also
covers the preliminary switch-to-compounding helper. The materializer treats
that dimension as authoritative and constructs a compatible pre-state plus
request, then the normal runner validates the outcome and the coverage report
verifies that the intended branch was reached.

Both modes fit the same feedback loop:

1. Choose generation mode and deterministic configuration.
2. Produce vectors from sampled inputs or sampled targets.
3. Run vectors against the pyspec.
4. Measure input coverage, semantic coverage, interaction coverage, and code
   coverage.
5. Use gaps to adjust models, intents, materializers, or sampled targets.

The new evolution campaign is intended to make this ladder explicit. It
currently starts from scratch with a handler-touch suite and then adds a
profile-partition suite, an input-profile suite, and a pairwise input-profile
interaction suite. The broader validator-state profile-interaction suite is
available as an implementation-diversity experiment, but it is not in the
default evolution campaign for now because it is heavier. Future campaign
phases can add intent-guided suites, mutation suites, and state-corpus reuse
suites without changing the basic reporting loop.

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
- non-overlapping campaign suite ownership

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

Semantic interaction coverage measures combinations of semantic dimensions, not
only individual intents. Observed pairwise reporting is implemented; expected,
constrained, and sampled higher-order interaction targets remain future
extensions.

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

#### Modes

1. **Observed pair coverage** (implemented)
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
- Should campaigns ever allow intentional suite overlap, or should overlap stay
  limited to ad hoc/manual comparisons?
- Should campaigns report both raw vectors and unique semantic intents?
- Which semantic dimensions are stable enough to encode in ontology now?
