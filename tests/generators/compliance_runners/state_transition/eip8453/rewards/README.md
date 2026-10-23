# EIP-8453 round rewards

This provider generates `round_processing/rewards` vectors for
`process_rewards`, starting at its immediate input after inactivity updates.
Fixtures are constructed in isolation; optional whole-round snapshots are
omitted.

```bash
make comptests kind=state_transition fork=eip8453 handler=rewards profile=smoke \
  comptests_dir=/tmp/eip8453-rewards
uv run pytest tests/generators/compliance_runners/runner/test_run.py \
  --kind=state_transition --test-dir /tmp/eip8453-rewards
```

## Coverage contract

Each record observes validator **zero**, plus the surrounding population. The
index is fixed by the target, not recovered from claimed dimensions or metadata.
Unlike the other per-validator round targets, this observer accepts
multi-validator states directly and never requires a selector. Empty registries
are outside its record scope.

Dimensions cover:

- Current-epoch activity and slashing of validator zero.
- Its progress and finality flags, independently.
- Its effective balance in zero, one, or multiple balance increments.
- Presence of other active, unslashed participants for each flag, independently.
- Finality delay below, immediately below, equal to, immediately above, or above
  the inactivity-leak threshold. Equality still pays rewards.

`smoke` covers each dimension. `normal` and `standard` combine pairwise coverage
with exhaustive activity/slashing/flag/balance/leak interactions. `max` exhausts
all 960 combinations on the current presets. No rejecting cases are declared.
The observer reads pre-state fields without using reward or participation
helpers. These dimensions characterize inputs; they do not claim that a nonzero
reward was observed, or classify every possible integer-rounding boundary.

## Materialization and validation

The materializer builds three validators at a round boundary. Validator zero
realizes the requested lifecycle, flags, and balance bucket. Two always-active,
unslashed peers have unequal effective balances of one and two increments. Their
flags realize the other-participant dimensions. All balances start nonzero so
unexpected penalties are visible. Inactivity scores differ to expose accidental
use of finality weights in reward accounting. Height participation arrays are
empty of votes: rewards must come from round flags.

Fixtures exercise local processor behavior, including low/zero effective
balances and synthetic lifecycle combinations; full-chain reachability is not
asserted. The materializer preserves the selected assignment and uses the
executable spec only to compute the vector's expected post-state. Construction
is deterministic; seed-dependent variation is deferred.

Tests recover every max-profile assignment on minimal and mainnet and score the
observed corpus against every profile. An independent Python integer/isqrt
oracle checks exact reward amounts, including per-flag division and the
effective-balance minimum used in denominators. Whole-state comparison checks
that only balances change. Additional cases test zero active balance, clamped
finality delay, slashed supporting validators, and independence from inactivity
scores. Smoke vectors are written, discovered, validated, and replayed through
the common runner.

Empty-validator loops, arbitrary population sizes, systematic square-root and
rounding-boundary coverage, whole-round sequencing, and randomization remain
separate extensions. These limits are not hidden feasibility constraints.
