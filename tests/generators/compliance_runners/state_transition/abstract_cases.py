from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from typing import Any

from tests.generators.compliance_runners.py_to_mzn import Convertor, get_solutions

MODEL_PACKAGE = "tests.generators.compliance_runners.state_transition.models"
VALIDATOR_STATE_MODEL = "validator_state.py"


@dataclass(frozen=True)
class AbstractStateTransitionCase:
    """A solved abstract profile plus runner-facing labels."""

    handler_name: str
    case_name: str
    profile: dict[str, Any]


HANDLER_NAMES = (
    "withdrawal_request",
    "consolidation_request",
    "registry_updates",
)


def load_validator_state_model() -> str:
    model = resources.files(MODEL_PACKAGE).joinpath(VALIDATOR_STATE_MODEL)
    return model.read_text()


def transpile_validator_state_model() -> str:
    return Convertor().convert(load_validator_state_model())


def solve_validator_state_profiles(limit: int | None = None) -> Iterable[dict[str, Any]]:
    for index, profile in enumerate(get_solutions(transpile_validator_state_model())):
        if limit is not None and index >= limit:
            return
        yield profile


def enumerate_abstract_cases(limit: int | None = None) -> Iterable[AbstractStateTransitionCase]:
    for index, profile in enumerate(solve_validator_state_profiles(limit=limit)):
        handler_name = classify_handler(profile)
        yield make_abstract_case(handler_name, index, profile)


def select_abstract_cases(
    per_handler_limit: int,
    handlers: Iterable[str] = HANDLER_NAMES,
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")

    counts = {handler: 0 for handler in requested_handlers}
    for index, profile in enumerate(solve_validator_state_profiles()):
        handler_name = classify_handler(profile)
        if handler_name not in counts:
            continue
        if counts[handler_name] >= per_handler_limit:
            continue
        counts[handler_name] += 1
        yield make_abstract_case(handler_name, index, profile)
        if all(count >= per_handler_limit for count in counts.values()):
            return


def enumerate_matching_abstract_cases(
    handlers: Iterable[str],
) -> Iterable[AbstractStateTransitionCase]:
    requested_handlers = tuple(handlers)
    unknown_handlers = set(requested_handlers) - set(HANDLER_NAMES)
    if unknown_handlers:
        raise ValueError(f"Unknown handlers: {sorted(unknown_handlers)}")

    for index, profile in enumerate(solve_validator_state_profiles()):
        handler_name = classify_handler(profile)
        if handler_name in requested_handlers:
            yield make_abstract_case(handler_name, index, profile)


def make_abstract_case(
    handler_name: str,
    solution_index: int,
    profile: dict[str, Any],
) -> AbstractStateTransitionCase:
    return AbstractStateTransitionCase(
        handler_name=handler_name,
        case_name=f"validator_state_profile_{solution_index:04d}",
        profile=profile,
    )


def classify_handler(profile: dict[str, Any]) -> str:
    if profile["has_pending_consolidation_request"]:
        return "consolidation_request"
    if profile["has_pending_withdrawal_request"]:
        return "withdrawal_request"
    if profile["withdrawal_credential_type"] in ("ETH1", "COMP"):
        return "withdrawal_request"
    return "registry_updates"
