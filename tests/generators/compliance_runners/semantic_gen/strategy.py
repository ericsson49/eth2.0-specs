from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class AspectDimension:
    """A finite semantic dimension that a strategy may choose from."""

    name: str
    values: tuple[Any, ...]
    model: str | None = None

    @property
    def qualified_name(self) -> str:
        if self.model is None:
            return self.name
        return f"{self.model}.{self.name}"


@dataclass(frozen=True)
class AspectAssignment:
    """A chosen value for an aspect dimension."""

    dimension: AspectDimension
    value: Any

    @property
    def qualified_name(self) -> str:
        return self.dimension.qualified_name

    def label(self) -> str:
        return f"{self.qualified_name}:{self.value}"


@dataclass(frozen=True)
class Constraint:
    """A symbolic requirement accumulated by a strategy."""

    kind: str
    data: Any


@dataclass(frozen=True)
class CoverageItem:
    """A semantic coverage obligation or observation target."""

    kind: str
    labels: tuple[str, ...]

    def label(self) -> str:
        return f"{self.kind}:{'|'.join(self.labels)}"


@dataclass(frozen=True)
class StrategyState:
    """Accumulated symbolic effects for one strategy path."""

    constraints: tuple[Constraint, ...] = ()
    coverage: tuple[CoverageItem, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_constraint(self, constraint: Constraint) -> StrategyState:
        return StrategyState(
            constraints=(*self.constraints, constraint),
            coverage=self.coverage,
            metadata=dict(self.metadata),
        )

    def with_coverage(self, item: CoverageItem) -> StrategyState:
        return StrategyState(
            constraints=self.constraints,
            coverage=(*self.coverage, item),
            metadata=dict(self.metadata),
        )

    def with_metadata(self, key: str, value: Any) -> StrategyState:
        metadata = dict(self.metadata)
        metadata[key] = value
        return StrategyState(
            constraints=self.constraints,
            coverage=self.coverage,
            metadata=metadata,
        )


@dataclass(frozen=True)
class StrategyCase(Generic[T]):
    """One interpreted path through a strategy program."""

    value: T
    constraints: tuple[Constraint, ...]
    coverage: tuple[CoverageItem, ...]
    metadata: dict[str, Any]


class Gen(Generic[T]):
    """A small nondeterminism-plus-writer monad for generation strategies."""

    def __init__(
        self,
        run: Callable[[StrategyState], Iterable[tuple[T, StrategyState]]],
    ) -> None:
        self._run = run

    def run(self, state: StrategyState | None = None) -> Iterable[tuple[T, StrategyState]]:
        yield from self._run(state or StrategyState())

    def bind(self, fn: Callable[[T], Gen[U]]) -> Gen[U]:
        def run(state: StrategyState) -> Iterable[tuple[U, StrategyState]]:
            for value, next_state in self.run(state):
                yield from fn(value).run(next_state)

        return Gen(run)

    def map(self, fn: Callable[[T], U]) -> Gen[U]:
        return self.bind(lambda value: pure(fn(value)))

    def then(self, other: Gen[U]) -> Gen[U]:
        return self.bind(lambda _value: other)


def pure(value: T) -> Gen[T]:
    return Gen(lambda state: ((value, state),))


def empty() -> Gen[Any]:
    return Gen(lambda _state: ())


def choose(values: Iterable[T]) -> Gen[T]:
    choices = tuple(values)
    return Gen(lambda state: ((value, state) for value in choices))


def guard(predicate: bool) -> Gen[None]:
    if predicate:
        return pure(None)
    return empty()


def require(constraint: Constraint) -> Gen[None]:
    return Gen(lambda state: ((None, state.with_constraint(constraint)),))


def cover(item: CoverageItem) -> Gen[None]:
    return Gen(lambda state: ((None, state.with_coverage(item)),))


def annotate(key: str, value: Any) -> Gen[None]:
    return Gen(lambda state: ((None, state.with_metadata(key, value)),))


def choose_aspect_value(dimension: AspectDimension) -> Gen[AspectAssignment]:
    return choose(
        AspectAssignment(dimension=dimension, value=value)
        for value in dimension.values
    )


def choose_n_wise_aspect_assignments(
    dimensions: Sequence[AspectDimension],
    order: int,
    *,
    include_lower_orders: bool = False,
) -> Gen[tuple[AspectAssignment, ...]]:
    """Choose assignments for `order`-wise finite aspect combinations."""
    if order < 1:
        raise ValueError(f"n-wise order must be at least 1: {order}")

    orders = range(1, order + 1) if include_lower_orders else (order,)
    dimension_groups = (
        group
        for current_order in orders
        for group in combinations(dimensions, current_order)
    )
    assignment_groups = (
        tuple(
            AspectAssignment(dimension=dimension, value=value)
            for dimension, value in zip(dimension_group, values, strict=True)
        )
        for dimension_group in dimension_groups
        for values in product(*(dimension.values for dimension in dimension_group))
    )
    return choose(assignment_groups)


def coverage_for_assignments(
    kind: str,
    assignments: Sequence[AspectAssignment],
) -> CoverageItem:
    return CoverageItem(kind=kind, labels=tuple(assignment.label() for assignment in assignments))


def feasibility_constraint(assignments: Sequence[AspectAssignment]) -> Constraint:
    return Constraint(
        kind="feasible",
        data=tuple(assignment.label() for assignment in assignments),
    )


def enumerate_strategy(
    program: Gen[T],
    *,
    accepts: Callable[[StrategyCase[T]], bool] | None = None,
    limit: int | None = None,
) -> Iterable[StrategyCase[T]]:
    """Interpret a strategy by enumerating its finite choices."""
    count = 0
    for value, state in program.run():
        case = StrategyCase(
            value=value,
            constraints=state.constraints,
            coverage=state.coverage,
            metadata=state.metadata,
        )
        if accepts is not None and not accepts(case):
            continue
        yield case
        count += 1
        if limit is not None and count >= limit:
            return


def n_wise_strategy(
    dimensions: Sequence[AspectDimension],
    order: int,
    *,
    coverage_kind: str,
    include_lower_orders: bool = False,
) -> Gen[tuple[AspectAssignment, ...]]:
    """Build a basic strategy for n-wise aspect coverage."""
    return choose_n_wise_aspect_assignments(
        dimensions,
        order,
        include_lower_orders=include_lower_orders,
    ).bind(
        lambda assignments: cover(coverage_for_assignments(coverage_kind, assignments))
        .then(require(feasibility_constraint(assignments)))
        .then(pure(assignments))
    )
