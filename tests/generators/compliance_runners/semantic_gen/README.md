# Semantic Test Generation

`semantic_gen` contains a generic framework for ontology-guided compliance
test generation. It is intentionally domain-neutral: consensus state
transition tests, fork-choice tests, and execution-spec tests can all use the
same high-level shape while providing their own aspects, models,
materializers, and oracle runners.

## Core Idea

Treat a system under test as a relation:

```text
DomainRelation ⊆ Input × Output × Outcome × Trace
```

For a state transition function, `Input` may be `PreState` plus an optional
operation. For fork choice, it may be `PreStore` plus an event sequence. For an
execution-spec test, it may be pre-state plus a transaction or block.

An **aspect** is a named predicate or projection over this relation. Aspects
can describe input shape, output effect, execution trace, or relationships
between them.

Examples:

- input aspect: queue is full
- input aspect: validator is slashed
- output aspect: state changed
- outcome aspect: assertion failure
- trace aspect: branch target reached
- relational aspect: queue full leads to no state change

The generator does not have to construct concrete test vectors immediately.
It can first reason over aspect assignments and coverage goals.

## Relational Interpretation

The framework can be read as a small relational algebra over abstract test
case spaces.

An aspect model is a relation:

```text
AspectModel(columns, constraints)
```

The columns are semantic dimensions, such as handler, state shape, operation
shape, output effect, outcome, or trace target. The constraints are predicates
over those columns, similar to a SQL `WHERE` clause.

Combining aspects is a natural join over shared columns:

```text
Goal = AspectA ⋈ AspectB ⋈ AspectC
```

The join succeeds when shared columns agree and fails when constraints are
inconsistent. In solver terms, the join is constraint conjunction; a non-empty
join has at least one witness.

Completion can be interpreted the same way. A selected goal is often a partial
row over coverage columns. Completion joins that partial row with the remaining
input-aspect relations and selects one compatible witness:

```text
CompletedInput = SelectedGoal ⋈ RemainingInputAspects
```

If completion is modelled as an outer join, uncompletable goals can be kept as
rows with missing witness columns. This makes the funnel explicit:

```text
planned goals
  -> completable goals
  -> selected goals
  -> materialized vectors
  -> executed vectors
  -> observed coverage
```

This interpretation is useful for optimization:

- projection pushdown: keep rows narrow until more columns are needed
- selection pushdown: apply handler, fork, or stage constraints early
- join ordering: join selective aspect relations before broad ones
- semi-joins: test whether a goal has any completion without building all
  witnesses
- anti-joins: report missing planned goals
- distinct: avoid materializing duplicate goal IDs
- limit pushdown: enforce suite budgets before expensive materialization

The current Python interpreter is one physical backend for these logical
plans. MiniZinc, SAT, SMT, randomized samplers, or minimizers can be viewed as
alternative execution engines for the same relational strategy.

## Strategy Programs

The `Gen` type is a small nondeterminism-plus-writer monad for semantic
generation strategies. A strategy composes:

- `choose`: finite symbolic choices
- `require`: symbolic constraints
- `cover`: intended coverage items
- `annotate`: strategy metadata

For example, an `n`-wise strategy:

```text
choose n-wise aspect assignment
  -> cover that assignment
  -> require feasibility
  -> return symbolic case
```

The strategy can be interpreted cheaply before materialization. This gives a
planning and debugging layer over the abstract test space.

## Coverage Goals

A common criterion is feasible `n`-wise aspect coverage:

```text
covered feasible n-wise combinations
/
feasible n-wise combinations
```

Feasibility is model-specific:

```text
exists witness .
  model(domain, witness) and combo(witness)
```

The same shape works for:

- input coverage: generated vectors instantiate input aspect combinations
- output coverage: executed tests observe output, outcome, or trace aspects
- hybrid coverage: tests connect input aspects with output or trace aspects

The denominator should be the filtered feasible space, not every syntactic
combination.

## Goal Ledgers and Funnels

`StrategyGoal` gives each semantic goal a stable `goal_id`. A domain-specific
generator can emit an expected-goals ledger during dry-run, then carry the same
goal IDs through materialization and execution metadata.

This enables funnel reports:

```text
symbolic goals
  -> feasible or completable goals
  -> selected goals
  -> materialized vectors
  -> executed vectors
  -> observed coverage
```

The funnel is useful for planning suite size and for debugging losses. For
example, if many goals are completable but few become vectors, the missing goal
IDs identify materializer, selection, or deduplication gaps.

## Domain Responsibilities

`semantic_gen` provides only the generic language and identifiers. Each domain
should provide:

- aspect vocabulary
- approximate feasibility or completion model
- strategy adapters from domain concepts to aspect dimensions
- materializers from completed witnesses to concrete tests
- oracle runner
- observed coverage extraction

The pyspec or domain reference implementation remains the final oracle. Models
are allowed to be approximate as long as generated vectors are validated by the
oracle.

## Backends

The strategy layer is backend-neutral. Different interpreters can eventually
target:

- MiniZinc or CP for finite-domain modelling and enumeration
- SAT for propositional aspect enumeration and approximate uniform sampling
- SMT or CDCL(T) for richer arithmetic and branch feasibility checks
- greedy or optimization-based suite minimization
- random or mutation-based stress generation

The goal is to keep high-level generation strategies stable while allowing the
solver, sampler, materializer, and minimizer implementations to evolve.
