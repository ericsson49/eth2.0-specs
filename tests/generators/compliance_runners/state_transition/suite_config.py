from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

DEFAULT_SUITE_CONFIG_DIR = Path(
    "tests/generators/compliance_runners/state_transition/suite_configs"
)


def resolve_suite_config_path(suite: str) -> Path:
    suite_path = Path(suite)
    if suite_path.exists():
        return suite_path
    candidate = DEFAULT_SUITE_CONFIG_DIR / f"{suite}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unknown suite config: {suite}")


def read_yaml(path: Path):
    yaml = YAML(typ="safe")
    return yaml.load(path.read_text())
