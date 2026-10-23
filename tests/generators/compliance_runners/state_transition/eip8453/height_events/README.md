# EIP-8453 height-event coverage

`TARGET` describes one direct `process_height_events` call. Supply its immediate
pre-state, after all block attestations, through `Context.pre`. The observer
reads concrete state fields and independently computes voting weights; it does
not call `has_quorum` or execute the handler. `Context.post` is not needed.

```python
from eth_consensus_specs.eip8453 import minimal as spec
from types import SimpleNamespace
from tests.generators.compliance_runners.state_transition.eip8453.height_events.materializer import (
    HeightEventsMaterializer,
)
from tests.generators.compliance_runners.state_transition.eip8453.height_events.target import TARGET
from tests.generators.compliance_runners.state_transition.evaluation.coverage_dsl import (
    Context,
    score,
)

target = TARGET.for_spec(spec)
obligations = target.profiles["standard"].run()
materializer = HeightEventsMaterializer(spec)
assignment = next(iter(target.profiles["standard"].generation_obligations()))
meta, parts = materializer.materialize_solution(SimpleNamespace(**dict(assignment)))
pre = spec.BeaconState.decode_bytes(parts[0][2])
record = target.record(target.observation(Context(spec, pre, None, None, {})))
report = score(target, [record], target.profiles["standard"])
```

The declaration covers positive-weight guards, three-way weighted quorum
boundaries, five-way timeout boundaries, justification precedence, pending and
qualifying entry counts, retained entries after pruning, and window eviction.
`advance_reason` and `evict_oldest` are input-derived predictions, not evidence
that an implementation produced the correct post-state. Post-state checking
remains a separate responsibility.

Scope assumes protocol-valid pre-states: target participation is a subset of
progress participation; pending entries are ordered by increasing height, above
the finalized height, and bounded by `PENDING_FINALITY_WINDOW`. Feasibility
constraints preserve these relationships at the abstract level. Count buckets
are ZERO/ONE/MANY. The preset-bound feasibility callback enumerates possible
pending/qualifying/retained count combinations, including pruning nonqualifying
entries below the final qualifying entry. It imposes no fixture-size limit.

`smoke` covers each dimension. `normal` and `standard` combine pairwise coverage
with exhaustive quorum/timeout interactions and exhaustive finalization
interactions. `max` covers all feasible combinations. No exceptional profile is
declared: this handler has no rejecting assertion on valid pre-states.

The registered `HeightEventsMaterializer` accepts complete coverage assignments
and emits SSZ `pre`/`post` parts with claimed dimensions. It uses three
equal-weight validators to realize quorum boundaries, capped inactivity scores
for zero total weight, and ordered pending entries to realize pruning and window
eviction. Materialization is deterministic. The default fork is `eip8453`; pass
`preset_name="mainnet"` when writing mainnet vectors with `materialize_reps`.

Generate and replay through the standard infrastructure:

```bash
make comptests kind=state_transition fork=eip8453 handler=height_events profile=smoke \
  comptests_dir=/tmp/eip8453-comptests
uv run pytest tests/generators/compliance_runners/runner/test_run.py \
  --kind=state_transition --test-dir /tmp/eip8453-comptests
```

The provider writes `operations/height_events` vectors and requires a
post-state. `fork=eip8453` with `handler=all` selects only registered EIP-8453
providers. Semantic scoring and code-coverage CLI integration remain separate
work.

Tests check every feasible assignment against the observer and expected
post-state on both presets, plus SSZ round trips for the smoke profile.
Pending-entry-specific quorum boundary coverage, weight-distribution coverage,
and cross-block sequencing are not claimed by this initial target.
