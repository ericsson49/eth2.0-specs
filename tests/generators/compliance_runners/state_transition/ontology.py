from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

ONTOLOGY_PACKAGE = "tests.generators.compliance_runners.state_transition"
ONTOLOGY_FILE = "test_ontology.yaml"


def load_test_ontology(path: Path | None = None) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    if path is not None:
        return yaml.load(path.read_text())
    ontology = resources.files(ONTOLOGY_PACKAGE).joinpath(ONTOLOGY_FILE)
    return yaml.load(ontology.read_text())


def stage_handlers(ontology: dict[str, Any] | None = None) -> dict[str, tuple[str, ...]]:
    ontology = ontology or load_test_ontology()
    return {
        stage_name: tuple(stage_data["handlers"])
        for stage_name, stage_data in ontology.get("stages", {}).items()
    }


def target_functions_by_runner(
    ontology: dict[str, Any] | None = None,
) -> dict[str, dict[str, tuple[str, ...]]]:
    ontology = ontology or load_test_ontology()
    return {
        runner: {
            handler: tuple(handler_data["functions"])
            for handler, handler_data in handlers.items()
        }
        for runner, handlers in ontology["targets"].items()
    }


def intent_outcomes_by_runner(
    ontology: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    ontology = ontology or load_test_ontology()
    return {
        runner: {
            handler: {
                intent_name: intent_data["outcome"]
                for intent_name, intent_data in intents.items()
            }
            for handler, intents in handlers.items()
        }
        for runner, handlers in ontology["intents"].items()
    }
