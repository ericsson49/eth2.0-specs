# Python to MiniZinc Transpiler

## Goal

The transpiler lets Python developers describe constraint models using a small,
readable subset of Python. The Python source is treated as a constraint DSL and
is transpiled to MiniZinc.

The long-term goal is dual use:

- Generate satisfying assignments with MiniZinc.
- Re-run the same Python constraints against generated assignments to validate
  them with normal Python assertions.

This keeps MiniZinc available as the solver backend while letting contributors
author most rules in familiar Python syntax.

## Current Shape

The current transpiler supports a small top-level Python subset:

```python
from enum import Enum, auto
from dataclasses import dataclass


class ComparisonOp(Enum):
    LT = auto()
    EQ = auto()
    GT = auto()


@dataclass
class ValidatorStateProfile:
    slashed: bool
    exit_epoch_set: bool
    balance_to_effective_balance: ComparisonOp


p: ValidatorStateProfile = ...


if p.slashed:
    p.exit_epoch_set

p.balance_to_effective_balance in {ComparisonOp.LT, ComparisonOp.EQ}
```

This is converted to MiniZinc declarations and constraints:

```minizinc
enum ComparisonOp = { LT, EQ, GT };

type ValidatorStateProfile = record(
  bool: slashed,
  bool: exit_epoch_set,
  ComparisonOp: balance_to_effective_balance,
);

var ValidatorStateProfile: p;

constraint ((p).slashed) -> ((p).exit_epoch_set);
constraint ((p).balance_to_effective_balance) in { LT, EQ };
```

## Current Mapping

Python `Enum` classes become MiniZinc enums:

```python
class ComparisonOp(Enum):
    LT = auto()
    EQ = auto()
    GT = auto()
```

```minizinc
enum ComparisonOp = { LT, EQ, GT };
```

Python dataclasses become MiniZinc records:

```python
@dataclass
class ValidatorStateProfile:
    slashed: bool
    exit_epoch_set: bool
    op: ComparisonOp
```

```minizinc
type ValidatorStateProfile = record(
  bool: slashed,
  bool: exit_epoch_set,
  ComparisonOp: op,
);
```

Top-level annotated ellipsis assignments become MiniZinc decision variables:

```python
p: ValidatorStateProfile = ...
```

```minizinc
var ValidatorStateProfile: p;
```

Top-level expressions become MiniZinc constraints:

```python
p.exit_epoch_set == p.slashed
```

```minizinc
constraint ((p).exit_epoch_set) == ((p).slashed);
```

Top-level `if` statements become implications:

```python
if p.slashed:
    p.exit_epoch_set
```

```minizinc
constraint ((p).slashed) -> ((p).exit_epoch_set);
```

Multiple expressions in an `if` body become a conjunction:

```python
if p.slashed:
    p.exit_epoch_set
    p.balance_to_effective_balance == ComparisonOp.LT
```

```minizinc
constraint ((p).slashed) ->
  (((p).exit_epoch_set) /\ (((p).balance_to_effective_balance) == (LT)));
```

## Expression Mapping

Boolean operators map directly:

```python
a and b
a or b
not a
```

```minizinc
(a) /\ (b)
(a) \/ (b)
not (a)
```

Comparisons map directly:

```python
==  !=  <  <=  >  >=  in  not in
```

Set literals map to MiniZinc set literals:

```python
p.op in {ComparisonOp.LT, ComparisonOp.EQ}
```

```minizinc
((p).op) in { LT, EQ }
```

Enum member references drop the Python enum qualifier:

```python
ComparisonOp.LT
```

```minizinc
LT
```

Record field access maps to MiniZinc record field access:

```python
p.slashed
```

```minizinc
(p).slashed
```

## Proposed Extension: Constraint Functions

The current top-level expression style is compact, but it is not valid
validation code: a bare expression in Python is evaluated and discarded.

The proposed model is to introduce decorated constraint functions and use
`assert` for rule statements:

```python
@constraint
def lifecycle(p: ValidatorStateProfile):
    if p.slashed:
        assert p.exit_epoch_set

    assert p.exit_epoch_set == p.withdrawable_epoch_set
```

This is more verbose, but it is easier for Python developers to reason about.
It also lets the same function validate generated solutions:

```python
profile = ValidatorStateProfile(...)
lifecycle(profile)  # passes or raises AssertionError
```

The MiniZinc mapping would be:

```python
assert expr
```

```minizinc
constraint expr;
```

and:

```python
if condition:
    assert expr
```

```minizinc
constraint condition -> expr;
```

Multiple assertions under one condition would become a conjunction:

```python
if condition:
    assert expr1
    assert expr2
```

```minizinc
constraint condition -> (expr1 /\ expr2);
```

## Proposed Authoring Style

Constraint files should read like ordinary Python predicates over dataclasses
and enums:

```python
@constraint
def balance(p: ValidatorStateProfile):
    assert p.effective_balance_to_max_effective_balance in {
        ComparisonOp.LT,
        ComparisonOp.EQ,
    }

    if p.balance_is_zero:
        assert p.balance_to_effective_balance in {ComparisonOp.LT, ComparisonOp.EQ}

    if p.effective_balance_lte_ejection_balance:
        assert p.effective_balance_to_min_activation_balance == ComparisonOp.LT
```

This can be explained to contributors as:

> Write normal Python assertions over finite dataclass-shaped values. The
> transpiler turns those assertions into MiniZinc constraints, and Python can
> later execute the assertions to check generated solutions.

## Initial Constraint Function Subset

A conservative first version should support:

- `@constraint def name(args): ...`
- `assert expr`
- `if expr: assert ...`
- multiple `assert` statements under an `if`
- boolean operators: `and`, `or`, `not`
- comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`
- dataclass field access
- enum member access
- set literals

It should reject ambiguous or difficult Python until there is a clear need:

- mutation
- arbitrary function calls
- dynamic attribute access
- exception handling
- comprehensions
- `else` blocks
- chained comparisons

Rejecting unsupported Python explicitly is better than silently producing a
wrong model.

## Later Extension: Finite Loops

Finite loops are a natural extension because they map to MiniZinc `forall`.

Python:

```python
@constraint
def non_negative_balances(balances):
    for i in range(len(balances)):
        assert balances[i] >= 0
```

MiniZinc shape:

```minizinc
constraint forall(i in index_set(balances)) (
  balances[i] >= 0
);
```

Explicit finite domains are the safest starting point:

```python
@constraint
def allowed_ops():
    for op in {ComparisonOp.LT, ComparisonOp.EQ}:
        assert op != ComparisonOp.GT
```

```minizinc
constraint forall(op in { LT, EQ }) (
  op != GT
);
```

Nested loops can become nested `forall`s or a single `forall` with multiple
generators:

```python
@constraint
def unique_indices(indices):
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            assert indices[i] != indices[j]
```

```minizinc
constraint forall(i, j where j > i) (
  indices[i] != indices[j]
);
```

The key rule is that loop domains must be finite and transpilable. Good initial
domains include:

- set literals
- enum values
- `range(...)` with transpilable finite bounds
- named finite MiniZinc sets
- array index sets

General Python iteration should stay unsupported until the model has a clear
way to represent the iterable in MiniZinc.

## Solver Round Trip

The intended workflow is:

1. Author Python dataclasses, enums, and `@constraint` functions.
2. Transpile the model to MiniZinc.
3. Ask MiniZinc for satisfying assignments.
4. Convert MiniZinc records back into Python dataclass instances.
5. Execute the same `@constraint` functions against each generated assignment.

This gives two forms of confidence:

- MiniZinc proves that generated assignments satisfy the transpiled constraints.
- Python assertions verify that generated assignments satisfy the source-level
  constraints developers wrote.

## CLI

The transpiler implementation currently lives with the compliance-test tooling:

```text
tests/generators/compliance_runners/py_to_mzn.py
```

The current CLI lives at:

```text
tests/generators/compliance_runners/py_to_mzn_cli.py
```

Usage:

```bash
uv run --extra test python -m tests.generators.compliance_runners.py_to_mzn_cli vs.py
uv run --extra test python -m tests.generators.compliance_runners.py_to_mzn_cli vs.py -o vs.mzn
```

It prints MiniZinc to stdout by default and writes to a file when `--output` is
provided.
