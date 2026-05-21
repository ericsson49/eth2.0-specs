from tests.generators.compliance_runners.semantic_gen import (
    AspectDimension,
    Constraint,
    cover,
    CoverageItem,
    enumerate_strategy,
    make_strategy_goal,
    n_wise_strategy,
    pure,
    require,
)


def test_strategy_accumulates_constraints_and_coverage():
    program = (
        cover(CoverageItem(kind="intent", labels=("handler:demo",)))
        .then(require(Constraint(kind="feasible", data=("x:1",))))
        .then(pure("case"))
    )

    cases = list(enumerate_strategy(program))

    assert len(cases) == 1
    assert cases[0].value == "case"
    assert cases[0].coverage == (CoverageItem(kind="intent", labels=("handler:demo",)),)
    assert cases[0].constraints == (Constraint(kind="feasible", data=("x:1",)),)


def test_n_wise_strategy_enumerates_aspect_assignments():
    dimensions = (
        AspectDimension(model="state", name="a", values=("A0", "A1")),
        AspectDimension(model="state", name="b", values=("B0",)),
    )

    cases = list(enumerate_strategy(n_wise_strategy(dimensions, 2, coverage_kind="pair")))

    assert len(cases) == 2
    assert [tuple(assignment.label() for assignment in case.value) for case in cases] == [
        ("state.a:A0", "state.b:B0"),
        ("state.a:A1", "state.b:B0"),
    ]
    assert [case.coverage[0].label() for case in cases] == [
        "pair:state.a:A0|state.b:B0",
        "pair:state.a:A1|state.b:B0",
    ]


def test_n_wise_strategy_excludes_non_coverage_dimensions_from_coverage_label():
    dimensions = (
        AspectDimension(model="state", name="coverage", values=("A",)),
        AspectDimension(
            model="state",
            name="materialization",
            values=("B",),
            include_in_coverage=False,
        ),
        AspectDimension(
            model="state",
            name="control",
            values=("C",),
            include_in_coverage=False,
        ),
    )

    cases = list(enumerate_strategy(n_wise_strategy(dimensions, 3, coverage_kind="goal")))

    assert len(cases) == 1
    assert tuple(assignment.label() for assignment in cases[0].value) == (
        "state.coverage:A",
        "state.materialization:B",
        "state.control:C",
    )
    assert cases[0].coverage[0].label() == "goal:state.coverage:A"


def test_strategy_goal_ids_are_stable():
    labels = (
        "queue.pending_partial_withdrawals:FULL",
        "withdrawal_request_input.request_kind:FULL_EXIT_REQUEST",
    )

    goal = make_strategy_goal(
        handler="withdrawal_request",
        kind="input_profile",
        labels=labels,
        completable=True,
    )
    same_goal = make_strategy_goal(
        handler="withdrawal_request",
        kind="input_profile",
        labels=labels,
        completable=False,
    )

    assert goal.goal_id == same_goal.goal_id
    assert goal.to_json_data()["labels"] == list(labels)
