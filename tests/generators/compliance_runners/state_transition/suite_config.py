from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

DEFAULT_SUITE_CONFIG_DIR = Path(
    "tests/generators/compliance_runners/state_transition/suite_configs"
)
DEFAULT_CAMPAIGN_CONFIG_DIR = Path(
    "tests/generators/compliance_runners/state_transition/campaign_configs"
)
DEFAULT_SUITE_OUTPUT_ROOT = Path("state_transition_tests")
DEFAULT_SUITE_COVERAGE_ROOT = Path("state_transition_coverage")
DEFAULT_CAMPAIGN_OUTPUT_ROOT = Path("state_transition_tests")


def resolve_suite_config_path(suite: str) -> Path:
    suite_path = Path(suite)
    if suite_path.exists():
        return suite_path
    candidate = DEFAULT_SUITE_CONFIG_DIR / f"{suite}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unknown suite config: {suite}")


def resolve_campaign_config_path(campaign: str) -> Path:
    campaign_path = Path(campaign)
    if campaign_path.exists():
        return campaign_path
    candidate = DEFAULT_CAMPAIGN_CONFIG_DIR / f"{campaign}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Unknown campaign config: {campaign}")


def read_yaml(path: Path):
    yaml = YAML(typ="safe")
    return yaml.load(path.read_text())


def suite_name_from_config(suite_config: dict, suite_config_path: Path) -> str:
    return suite_config.get("name") or suite_config_path.stem


def default_suite_output_dir(suite_config: dict, suite_config_path: Path) -> Path:
    suite_name = suite_name_from_config(suite_config, suite_config_path)
    configured_output = suite_config.get("generation", {}).get("output")
    if configured_output:
        return Path(configured_output)
    return DEFAULT_SUITE_OUTPUT_ROOT / suite_name


def default_suite_coverage_dir(suite_config: dict, suite_config_path: Path) -> Path:
    suite_name = suite_name_from_config(suite_config, suite_config_path)
    configured_output = suite_config.get("coverage", {}).get("output")
    if configured_output:
        return Path(configured_output)
    return DEFAULT_SUITE_COVERAGE_ROOT / suite_name


def default_campaign_output_dir(campaign_config: dict, campaign_config_path: Path) -> Path:
    campaign_name = campaign_config.get("name") or campaign_config_path.stem
    configured_output = campaign_config.get("output")
    if configured_output:
        return Path(configured_output)
    return DEFAULT_CAMPAIGN_OUTPUT_ROOT / campaign_name
