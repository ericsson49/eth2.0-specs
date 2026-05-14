# State Transition Vector Generation with MiniZinc

## Goal

Use Python-authored constraint models, transpiled to MiniZinc, to generate
abstract state-transition test cases. Python then materializes each MiniZinc
solution into concrete consensus-spec objects and writes standard test vectors.

The goal is not to model the full `BeaconState` in MiniZinc. Instead, MiniZinc
enumerates compact abstract profiles that select interesting edge cases.

The primary target formats are:

- [`operations`](../../../formats/operations/README.md)
- [`epoch_processing`](../../../formats/epoch_processing/README.md)

Electra/Fulu are good examples because Fulu inherits most beacon-chain behavior
from Electra and directly modifies execution payload processing.

## Pipeline

```text
Python constraint model
  -> MiniZinc model
  -> MiniZinc solutions
  -> Python materializer
  -> concrete BeaconState / operation / payload
  -> spec transition function
  -> standard test-vector files
```

MiniZinc should answer questions like:

- Which validator lifecycle states are possible?
- Which request/state combinations should be ignored, accepted, or fail?
- Which epoch-processing branches can be reached?
- Which finality/checkpoint/cache-sensitive state shapes should be tested?

Python should handle:

- SSZ object construction.
- BLS signatures and mocked execution validity.
- Merkle roots, block roots, state roots, and fork versions.
- Running the actual pyspec transition function.
- Dumping `*.ssz_snappy` and `meta.yaml`.

## Target: Operations Vectors

The operations vector format is:

```text
tests/<preset>/<fork>/operations/<operation-name>/<suite>/<case>/
  meta.yaml
  pre.ssz_snappy
  <input-name>.ssz_snappy
  post.ssz_snappy       # absent/empty when the operation is invalid
```

Relevant Electra/Fulu operation handlers include:

- `deposit`
- `deposit_request`
- `withdrawal_request`
- `consolidation_request`
- `voluntary_exit`
- `withdrawals`
- `execution_payload`

For each generated abstract case:

```text
solution -> pre BeaconState + operation object
         -> process_<operation>(state, operation)
         -> post BeaconState or expected failure
         -> vector parts
```

## Target: Epoch Processing Vectors

The epoch-processing vector format is:

```text
tests/<preset>/<fork>/epoch_processing/<sub-transition>/<suite>/<case>/
  meta.yaml
  pre.ssz_snappy
  post.ssz_snappy
  pre_epoch.ssz_snappy
  post_epoch.ssz_snappy
```

Relevant Electra/Fulu sub-transitions include:

- `justification_and_finalization`
- `registry_updates`
- `slashings`
- `effective_balance_updates`
- `pending_deposits`
- `pending_consolidations`

For each generated abstract case:

```text
solution -> pre_epoch BeaconState
         -> transition to just before selected sub-transition
         -> pre BeaconState
         -> process_<sub_transition>(state)
         -> post BeaconState
         -> also run full process_epoch(pre_epoch)
         -> post_epoch BeaconState
         -> vector parts
```

The full-epoch `pre_epoch` / `post_epoch` pair is useful because it can catch
bugs in dependencies between epoch sub-transitions.

## Core Abstract State Groups

The model should be composed from small finite profiles. Each profile summarizes
one part of `BeaconState` that is important for branch coverage.

### Epoch Context

Describes the current slot/epoch and churn conditions.

Useful fields:

- current epoch relation to activation, exit, and withdrawable epochs
- whether the state is at an epoch boundary
- activation/exit churn remaining: none, partial, enough
- consolidation churn remaining: none, at threshold, enough
- finalized epoch relation to activation eligibility

Targets:

- `process_registry_updates`
- `process_pending_deposits`
- `process_pending_consolidations`
- `process_voluntary_exit`
- `process_withdrawal_request`
- `process_consolidation_request`

### Validator Lifecycle Profile

Summarizes the lifecycle fields of one or more validators.

Useful fields:

- activation eligibility epoch set
- activation eligibility finalized
- activation epoch set
- active now
- active for `SHARD_COMMITTEE_PERIOD`
- exit epoch set
- withdrawable epoch set
- withdrawable now
- slashed

Targets:

- voluntary exits
- withdrawal requests
- consolidation requests
- registry updates
- slashings

### Balance Profile

Summarizes balances without forcing MiniZinc to reason about full Gwei values.

Useful fields:

- balance is zero
- balance relation to effective balance: `<`, `=`, `>`
- effective balance relation to `MIN_ACTIVATION_BALANCE`
- effective balance relation to max effective balance
- excess balance over `MIN_ACTIVATION_BALANCE`
- excess balance over max effective balance
- hysteresis update direction: down, unchanged, up

Targets:

- `process_effective_balance_updates`
- `process_pending_deposits`
- partial withdrawals
- consolidation and exit churn
- withdrawal sweep behavior

### Withdrawal Credential Profile

Withdrawal credentials are a small but high-impact state dimension.

Useful values:

- BLS credential
- ETH1 execution credential
- compounding credential
- unknown prefix
- execution address matches request source address

Targets:

- `process_withdrawal_request`
- `process_consolidation_request`
- `process_withdrawals`
- `get_max_effective_balance`
- full and partial withdrawability

### Pending Queues Profile

Summarizes Electra pending queues.

Useful fields:

- pending deposits: empty, non-empty, at per-epoch limit, blocked by churn
- pending partial withdrawals: empty, has entry for validator, full
- pending consolidations: empty, non-empty, full
- pending withdrawal amount relation to excess balance
- pending deposit source: genesis deposit, deposit request, consolidation excess
- pending deposit slot finalized

Targets:

- `process_pending_deposits`
- `process_pending_consolidations`
- `process_withdrawal_request`
- `process_consolidation_request`
- `process_voluntary_exit`

## Operation-Specific Profiles

### Deposit Flow Profile

Useful fields:

- pubkey is new or existing
- signature valid or invalid
- amount below, at, or above `MIN_ACTIVATION_BALANCE`
- withdrawal credential type
- deposit request start index unset or set
- eth1 deposit index before, at, or after deposit-request start index

Targets:

- `process_deposit`
- `process_deposit_request`
- `process_pending_deposits`
- `process_operations` deposit disabling logic

### Withdrawal Request Profile

Useful fields:

- request pubkey exists
- request source address matches validator credentials
- request is full exit marker or partial amount
- validator is active
- validator is already exiting
- validator is active long enough
- pending partial queue is full
- validator has pending withdrawal balance
- validator has compounding credentials
- validator has sufficient effective balance
- validator has enough excess balance

Expected outcomes:

- request ignored
- full exit initiated
- pending partial withdrawal appended

Target:

- `operations/withdrawal_request`

### Consolidation Request Profile

Useful fields:

- source equals target
- source exists
- target exists
- source credential type
- target credential type
- source address matches
- source active
- target active
- source already exiting
- target already exiting
- source active long enough
- source has pending withdrawal balance
- pending consolidation queue is full
- consolidation churn is available
- request is switch-to-compounding or source-to-target consolidation

Expected outcomes:

- request ignored
- source switched to compounding and excess active balance queued
- source exit initiated and pending consolidation appended

Target:

- `operations/consolidation_request`

### Withdrawal Sweep Profile

Useful fields:

- next withdrawal validator index position
- pending partial withdrawals: none, below sweep limit, at sweep limit
- pending partial withdrawal withdrawable now
- validator fully withdrawable
- validator partially withdrawable
- sweep reaches `MAX_WITHDRAWALS_PER_PAYLOAD`
- sweep wraps around validator registry

Targets:

- `operations/withdrawals`
- execution payload withdrawal correctness

### Fulu Execution Payload Profile

Fulu directly modifies `process_execution_payload` by checking blob commitments
against `get_max_blobs_per_block(get_current_epoch(state))`.

Useful fields:

- parent hash matches latest execution payload header
- `prev_randao` matches current epoch randao mix
- timestamp matches state slot
- blob commitment count is below, equal to, or above the Fulu epoch limit
- execution engine accepts or rejects payload
- execution requests are empty, deposits-only, withdrawals-only,
  consolidations-only, or mixed

Expected outcomes:

- assertion failure before engine call
- assertion failure from engine rejection
- latest execution payload header updated

Target:

- `operations/execution_payload`

## Finality and Cache-Sensitive Profiles

Finality state should be modeled explicitly. It can expose implementation bugs
where clients cache derived data by epoch only, while the correct key also
depends on fork version, checkpoint root, state root, finalized checkpoint, or
other state inputs.

### Finality Profile

Useful fields:

- previous justified checkpoint relation:
  genesis, previous epoch, current epoch, old epoch, future epoch
- current justified checkpoint relation:
  genesis, previous epoch, current epoch, old epoch, future epoch
- finalized checkpoint relation:
  genesis, previous epoch, current epoch, old epoch
- justification bits pattern:
  none, current only, previous only, current and previous, four in a row,
  alternating, stale
- checkpoint root shape:
  all same, finalized equals previous justified, finalized equals current
  justified, all distinct, conflicting root at same epoch

Targets:

- `epoch_processing/justification_and_finalization`
- full `process_epoch` vectors
- fork-boundary regression tests when transition-format generation is added

### Fork Cache Profile

Useful fields:

- pre-fork, at fork epoch, post-fork same epoch, post-fork next epoch
- fork version changed
- same epoch under different fork
- same checkpoint epoch with different root
- reused validator registry
- reused block roots or state roots

Targets:

- cache invalidation around fork boundaries
- finality and committee cache bugs
- future `transition` runner vectors

Even while focusing on `epoch_processing` and `operations`, this profile is
valuable for constructing `pre_epoch` states that differ only in
finality/fork/cache-relevant dimensions.

## Example Abstract Model Shape

```python
class RequestKind(Enum):
    NONE = auto()
    WITHDRAWAL = auto()
    CONSOLIDATION = auto()
    DEPOSIT = auto()


@dataclass
class TransitionCase:
    epoch: EpochContext
    validator: ValidatorLifecycleProfile
    balance: BalanceProfile
    credentials: WithdrawalCredentialProfile
    queues: PendingQueuesProfile
    finality: FinalityProfile
    request_kind: RequestKind
    expect_valid: bool


@constraint
def withdrawal_request_rules(c: TransitionCase):
    if c.request_kind == RequestKind.WITHDRAWAL:
        if c.validator.exit_epoch_set:
            assert not c.expect_valid

        if c.queues.has_pending_withdrawal_for_validator:
            assert not c.expect_valid

        if c.credentials.kind == WithdrawalCredentialType.COMP:
            assert c.balance.has_excess_over_min_activation_balance or not c.expect_valid
```

The MiniZinc solution is not the test vector. It is a recipe for Python to build
one.

## Materialization Strategy

Each abstract profile needs a deterministic materializer:

```text
abstract profile -> concrete BeaconState / operation
```

The materializer should:

- start from an existing pyspec base state
- choose stable validator indices for source, target, proposer, and helper
  validators
- mutate only the fields required by the profile
- use existing helpers for signatures, credentials, exits, and pending queues
- compute roots and indexes normally through pyspec helpers
- keep case names deterministic and derived from profile labels

MiniZinc should avoid producing raw numeric values where possible. For example,
`balance_relation = GT` is better than asking MiniZinc for a concrete Gwei
amount. Python can then choose canonical values for `GT`, `EQ`, and `LT`.

## Vector Generation Strategy

For `operations`:

```text
1. solve abstract operation cases
2. materialize pre state and operation object
3. run process_<operation>
4. dump pre, input operation, and post/invalid marker
```

For `epoch_processing`:

```text
1. solve abstract epoch cases
2. materialize pre_epoch state
3. copy and advance to sub-transition pre state
4. run process_<sub_transition>
5. run full process_epoch on pre_epoch copy
6. dump pre, post, pre_epoch, and post_epoch
```

## Recommended First Slice

Start with a narrow surface where the abstract profiles are already close to
the existing helper work:

1. `operations/withdrawal_request`
2. `operations/consolidation_request`
3. `epoch_processing/pending_deposits`
4. `epoch_processing/pending_consolidations`
5. `epoch_processing/justification_and_finalization`

Then add Fulu-specific execution payload cases:

6. `operations/execution_payload` with blob-count relation to
   `get_max_blobs_per_block(current_epoch)`

This order proves the approach on small finite models before touching the
broader execution payload surface.

## Initial Implementation

The first implementation checkpoint lives under:

```text
tests/generators/compliance_runners/state_transition/
```

It includes:

- `models/validator_state.py`: a Python-authored finite validator-state profile
  model.
- `abstract_cases.py`: helpers to transpile the model to MiniZinc, solve it,
  and classify solved profiles by target handler.
- `generate_abstract_cases.py`: a small CLI for inspecting the transpiled
  MiniZinc model and selected abstract cases.

Useful commands:

```bash
uv run --extra test python -m tests.generators.compliance_runners.state_transition.generate_abstract_cases --emit-mzn /tmp/validator_state.mzn
uv run --extra test python -m tests.generators.compliance_runners.state_transition.generate_abstract_cases --per-handler-limit 5
uv run --extra test python -m tests.generators.compliance_runners.state_transition.generate_abstract_cases --handler withdrawal_request --per-handler-limit 10
```

The current output is intentionally abstract YAML. The next implementation step
is to materialize these profiles into concrete `pre`, operation input, and
`post` SSZ parts for the `operations` runner.

The first materializers support:

- `operations/deposit`
- `operations/bls_to_execution_change`
- `operations/deposit_request`
- `operations/proposer_slashing`
- `operations/attester_slashing`
- `operations/attestation`
- `operations/voluntary_exit`
- `operations/withdrawal_request`
- `operations/consolidation_request`
- `epoch_processing/justification_and_finalization`
- `epoch_processing/registry_updates`
- `epoch_processing/pending_deposits`
- `epoch_processing/pending_consolidations`
- `epoch_processing/effective_balance_updates`

```bash
uv run --extra test python -m tests.generators.compliance_runners.state_transition.generate_vectors --output /tmp/state-transition-vectors --per-handler-limit 5
uv run --extra test python -m tests.generators.compliance_runners.state_transition.generate_vectors --output /tmp/state-transition-vectors --per-handler-limit 5 --changed-only
uv run --extra test python -m tests.generators.compliance_runners.state_transition.generate_vectors --output /tmp/state-transition-vectors --per-handler-limit 5 --unchanged-only
uv run --extra test python -m tests.generators.compliance_runners.state_transition.generate_vectors --output /tmp/state-transition-vectors --per-handler-limit 5 --invalid-only
uv run --extra test python -m tests.generators.compliance_runners.state_transition.generate_vectors --output /tmp/state-transition-vectors --handler all --per-handler-limit 20 --guided
uv run --extra test python -m tests.generators.compliance_runners.state_transition.generate_vectors --output /tmp/state-transition-vectors --handler consolidation_request --per-handler-limit 5 --changed-only
uv run --extra test python -m tests.generators.compliance_runners.state_transition.generate_vectors --output /tmp/state-transition-vectors --handler all --per-handler-limit 5 --changed-only
```

The generator records `operation_valid` and `post_state_changed` in `meta.yaml`.
When a `process_*` function raises an assertion, the vector is negative and
omits `post.ssz_snappy`. For `deposit_request`, `withdrawal_request`, and
`consolidation_request`, many invalid preconditions are specified as ignored
requests or append-only behavior rather than assertions, so those cases are
emitted as valid vectors. Ignored requests have `post.ssz_snappy` equal to
`pre.ssz_snappy`. As a result, `--invalid-only` may emit no vectors for these
handlers until a materialized operation targets a `process_*` function with
assertion failures.

Most generated cases keep BLS verification disabled for speed, matching the
local runner default. Cases that intentionally exercise signature-sensitive
branches may set `bls_setting: 1` in `meta.yaml`; the runner enables BLS only for
that case and restores the previous setting afterwards. The first use of this is
`operations/deposit` for an invalid proof-of-possession signature.

The `--guided` mode keeps the shared validator profile model, but overlays
handler-specific guard intents. These intents are written to `meta.yaml` as
`coverage_tags`, making it possible to compare the intended guard coverage with
runtime coverage and refine the model or materializer when a tag does not reach
the expected branch.

It writes standard compliance-style operation cases:

```text
<output>/<preset>/<fork>/operations/withdrawal_request/minizinc_abstract/<case>/
  manifest.yaml
  meta.yaml
  pre.ssz_snappy
  <operation>.ssz_snappy
  post.ssz_snappy
```

The generated vectors can be checked against the pyspec with:

```bash
uv run --extra test pytest tests/generators/compliance_runners/state_transition/runner/test_run.py --test-dir /tmp/state-transition-vectors
```

Coverage evaluation can run the same vectors under `coverage.py` and write
compact text, full missing-line text, function summary, target-coverage
summary, semantic-coverage summary, JSON, HTML, and annotated-source reports:

```bash
uv run --extra test python -m tests.generators.compliance_runners.state_transition.measure_coverage --test-dir /tmp/state-transition-vectors --output /tmp/state-transition-coverage
```

By default, the command infers the pyspec source files to report from generated
`manifest.yaml` files. Use `--source-file` one or more times to override the
focused report targets. It also infers target functions, guide intents, and
expected outcomes from `test_ontology.yaml`, then writes `target_coverage.txt`
and `semantic_coverage.txt`. Use `--ontology` to provide an explicit ontology
YAML.

For reproducible suite generation, use a checked-in suite config:

```bash
uv run --extra test python -m tests.generators.compliance_runners.state_transition.run_suite --suite electra_operations_guided --coverage
uv run --extra test python -m tests.generators.compliance_runners.state_transition.run_suite --suite electra_operations_guided --coverage --summary
uv run --extra test python -m tests.generators.compliance_runners.state_transition.run_suite --suite electra_validator_lifecycle_guided --coverage --summary
uv run --extra test python -m tests.generators.compliance_runners.state_transition.run_suite --suite electra_participation_finality_guided --coverage --summary
uv run --extra test python -m tests.generators.compliance_runners.state_transition.run_suite --suite electra_rotating_resets_guided --coverage --summary
uv run --extra test python -m tests.generators.compliance_runners.state_transition.run_suite --suite electra_committee_sync_guided --coverage --summary
uv run --extra test python -m tests.generators.compliance_runners.state_transition.run_suite --suite electra_operations_guided --check-reproducible
uv run --extra test python -m tests.generators.compliance_runners.state_transition.run_campaign --campaign electra_state_transition
```

The default guided Electra operations profile lives in
`suite_configs/electra_operations_guided.yaml` and fixes the fork, preset,
non-stage operation handlers, generation mode, and coverage settings. Stage
owned handlers, including epoch-processing handlers and `sync_aggregate`, live
in stage-specific suite configs instead. If `run_suite --output` is not
provided, the suite runner derives the vector output directory as
`state_transition_tests/<suite-name>`. If `run_suite --coverage-output` is not
provided, it derives the coverage output directory as
`state_transition_coverage/<suite-name>`.
When `keep_existing` is false, the suite runner removes the configured vector
output directory before regenerating, preventing stale cases from affecting
validation or semantic coverage reports.

Suite configs may list handlers directly or name ontology stages. The first
stage profile is `validator_lifecycle`, expanding to `registry_updates`,
`slashings`, `effective_balance_updates`, `pending_deposits`, and
`pending_consolidations`. The `participation_finality` stage expands to
`justification_and_finalization`, `inactivity_updates`, `rewards_and_penalties`,
and `participation_flag_updates`. The `rotating_resets` stage expands to
`slashings_reset`, `randao_mixes_reset`, `eth1_data_reset`, and
`historical_summaries_update`. The `committee_sync` stage expands to
`sync_committee_updates` and `sync_aggregate`:

```yaml
generation:
  stages:
    - validator_lifecycle
```

Suite configs can optionally add deterministic distribution quotas. The
generator still materializes candidates in stable order, but only writes cases
that fit every configured dimension:

```yaml
generation:
  distribution:
    outcomes:
      changed: 12
      no_change: 12
      assertion_failure: 12
    runners:
      operations: 24
      epoch_processing: 12
```

Supported dimensions are `outcomes`, `runners`, and `handlers`. Within a
configured dimension, omitted labels are excluded. If candidates run out before
a quota is filled, generation still succeeds and `run_suite --summary` reports
the unmet quota.

The suite health summary can also be run directly over an existing generated
suite:

```bash
uv run --extra test python -m tests.generators.compliance_runners.state_transition.summarize_suite --test-dir /tmp/state-transition-vectors --coverage-dir /tmp/state-transition-coverage --suite electra_operations_guided
```

It reports generated handlers, intents per handler, missing ontology intents,
observed outcome counts, semantic outcome mismatches, and target coverage totals
when a coverage directory is provided.

The reproducibility check generates the same suite config twice into fresh
temporary directories and compares all emitted files byte-for-byte. Use
`--keep-reproducibility-temp` to preserve those directories when debugging a
difference.

Coverage campaigns aggregate multiple generated suites under one ontology-level
reporting unit. Campaign configs live in `campaign_configs/` and list
non-overlapping suite configs plus aggregate coverage settings:

```yaml
name: electra_state_transition
output: state_transition_tests/electra_state_transition
suites:
  - electra_operations_guided
  - electra_validator_lifecycle_guided
  - electra_committee_sync_guided
coverage:
  output: state_transition_coverage_campaign
  ontology: tests/generators/compliance_runners/state_transition/test_ontology.yaml
```

The campaign runner generates each suite, validates all generated directories,
measures coverage in one `coverage.py` session, and prints a combined health
summary. `run_campaign --output` overrides the campaign vector output root;
otherwise the campaign uses its configured `output` path, or
`state_transition_tests/<campaign-name>` when no path is configured. All suites
are generated into that single root; the reference-test path already includes
the suite name under each handler, so the vectors remain separated without an
extra filesystem layer. Campaign summaries include a stage view derived from
`test_ontology.yaml`, with case counts, outcome counts, semantic intent totals,
and target coverage totals per ontology stage. The lower-level coverage tool
also accepts repeated `--test-dir` arguments for direct aggregate reporting.

The coverage command also writes `interaction_coverage.txt`, an observed
semantic interaction report. The default ontology configuration records
pairwise combinations over `stage`, `runner`, `handler`, `intent`, and
`outcome`. The human suite and campaign summaries keep this compact by showing
only the number of observed combinations per dimension pair; the detailed file
lists each observed combination and its count. The same schema can later raise
`max_order` to `3` for triple-wise reporting or add stage-specific dimensions
from generated case profiles.
