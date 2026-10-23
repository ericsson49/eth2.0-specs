# EIP-8453 coverage targets

`height_events` is registered under `operations`. The round targets below have
materializers registered under `round_processing` for EIP-8453. The Gloas
providers for inactivity and participation flag updates remain under
`epoch_processing`. The new `rewards` provider is described in
[rewards/README.md](rewards/README.md); it uses a multi-validator record and
fixture rather than the single-validator contract below.

| Target module                       | Spec function                        | Record scope           |
| ----------------------------------- | ------------------------------------ | ---------------------- |
| `inactivity_updates.target`         | `process_inactivity_updates`         | One selected validator |
| `participation_flag_updates.target` | `process_participation_flag_updates` | One selected validator |

Both functions are called by `process_round` in EIP-8453. Supply the immediate
pre-state of the particular sub-transition, not the beginning of the entire
round: inactivity updates precede rewards, which precede flag updates.

## Observation contract

Bind `TARGET.for_spec(spec)` to the EIP-8453 minimal or mainnet spec. Observe a
`Context(spec, pre, None, post, {"validator_index": index})` and turn it into a
coverage record with `target.record(target.observation(ctx))`. `post` is
optional and is not read. A single-validator state may omit the index.
Multi-validator states must provide an explicit integer index; missing, invalid,
or out-of-range indices are errors. Empty registries have no per-validator
records.

For a complete corpus, observe each desired validator separately. Do not select
the index from claimed coverage dimensions. Multi-validator observations use an
explicit `Context` as above. Generated vectors contain a single validator, so
file validation and direct provider recovery (`recover(pre, None, post)`) use
the observer's implicit selection of index 0 without selector metadata.

The deterministic materializers use single-validator fixtures. Inactivity scores
are chosen jointly for the recovery and ejection boundaries. The states sit
immediately before the selected sub-transition at a round boundary; no earlier
sub-transitions are executed during fixture construction. Whole-round snapshots
are omitted because fixtures are constructed in isolation.

The adapters read fields directly and do not invoke the processors or their
participation helpers. All decision/effect dimensions are predictions from the
inputs, not measurements of successful execution. Validate actual post-state
scores, exits, flags, and unchanged fields independently.

## Inactivity updates

Coverage includes current-epoch activity, slashing, the current round's progress
flag, an already scheduled exit, zero initial score, and five-way comparison
boundaries around score recovery and the ejection threshold after adding bias.
This distinguishes recovery saturation, score increases, an exit request, and a
new exit versus a request for an already exiting validator. Height-level
progress participation is intentionally not the source of the current round's
flag.

The feasibility callback derives jointly possible score comparisons from the
selected preset constants. It does not treat comparisons against the same score
as independent. The model retains input boundaries for inactive and slashed
validators even when a branch is not executed.

The record scope is local handler behavior on structurally consistent inputs; it
does not impose full-chain reachability constraints on combinations of slashing,
activation, and scheduled exit. Exit churn allocation and the exact
exit/withdrawable epochs belong to separate coverage of the exit helper.

## Participation flag updates

Coverage crosses old progress/non-progress flags and height progress with
activity and slashing. The latter two are deliberately unfiltered: carry-over
applies to every validator. Expected effects distinguish no change, clearing all
flags, and leaving only the progress flag. Non-progress flags include finality;
the observer also recognizes any other set bits as needing removal.

## Profiles and validation

`smoke` covers each dimension. `normal`, `standard`, and `max` exhaust all
feasible combinations for these small models. There is no exceptional profile:
these are non-rejecting transitions on the modeled inputs.

`test_round_targets.py` builds concrete corpora independently of solver output,
checks complete coverage of all profiles on both presets, and checks the actual
spec transitions. Tests also cover explicit focus selection and the distinction
between round flags and height participation. Empty-loop behavior,
multi-validator interactions and whole-round ordering remain separate work.
Materializer tests check every feasible witness and full post-state on both
presets, plus serialized generation, coverage recovery, replay, and mixed-fork
validation. The materializers do not randomize their fixtures.

## Changed epoch handlers

`builder_pending_payments` and `pending_deposits` have EIP-8453-specific
providers under `epoch_processing`. Their targets, observers, and materializers
are separate from the Gloas providers. Fixtures represent immediate handler
pre-states at an epoch boundary, without executing earlier epoch handlers. Only
these two directly modified epoch sub-transitions are added; inherited handlers
are outside this addition's scope.

### Builder pending payments

The focal record is payment slot zero. Coverage crosses nonzero amount,
head-vote seat count around the minimum quorum, and the legacy payment weight
(which must not control EIP-8453 payment eligibility). Five seat-count buckets
use the ceiling of committee size times the threshold ratio: two below, one
below, exactly at, one above, and two above the minimum passing count. This
avoids demanding an unattainable fractional seat count.

Additional dimensions cover a payable final slot, existing withdrawals, and
populated next-epoch payment and vote buffers. Distinct buffer entries expose
append order, rotation, and clearing errors. `standard`/`normal` combine
pairwise coverage with exhaustive eligibility and rotation groups; `max`
exhausts all 320 combinations. `smoke` covers each dimension's values.

### Pending deposits

The focal record is the first queued deposit, belonging to an existing
validator. Coverage crosses its slot before/at/after `finalized_slot`,
finalization at the start/interior/end of an epoch, active/exiting/withdrawn
status, amount below/at/ above available churn, existing churn carry, and an
absent/finalized/unfinalized suffix. Same-epoch unfinalized deposits distinguish
the new exact-slot gate from the former epoch gate. Withdrawn deposits and
postponed exits exercise the gate's ordering relative to bypass and postponement
behavior.

`standard`/`normal` combine pairwise coverage with exhaustive finalization/role/
amount and gate/role/suffix/carry groups. `max` exhausts all 486 combinations;
`smoke` covers each dimension's values. The observer reads the pre-state and
reuses only the inherited churn-limit helper. New-validator signatures, registry
insertion, and churn-limit calculation itself are outside this target. The
observer requires a nonempty queue and a known first-deposit pubkey.

These are deterministic local-handler fixtures, not full-chain reachability
claims. Randomization is deferred. `test_epoch_providers.py` checks all modeled
combinations and complete post-states on both presets, independent recovery,
all-profile coverage, and serialized smoke generation/validation/replay. It also
checks the deposit processing cap, empty queue, and changed epoch dispatch that
excludes round work.
