from tests.generators.compliance_runners.gen_base import (
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
