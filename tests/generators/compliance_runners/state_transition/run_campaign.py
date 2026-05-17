from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .goal_ledger import goal_ledger_path, load_expected_goals
from .lean_report import format_lean_report, load_or_create_expected_goals
from .run_suite import generate_from_config, measure_from_config, validate_suites
from .suite_config import (
    default_campaign_output_dir,
    read_yaml,
    resolve_campaign_config_path,
    resolve_suite_config_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a campaign of state-transition test-suite profiles"
    )
    parser.add_argument(
        "--campaign",
        default="electra_state_transition_evolution",
        help="Campaign config name or path. Defaults to electra_state_transition_evolution.",
    )
    parser.add_argument(
        "--generate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate vectors from campaign suites.",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate generated vectors with the local runner.",
    )
    parser.add_argument(
        "--coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure aggregate coverage for generated vectors.",
    )
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print an aggregate campaign health summary.",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        help="Override campaign coverage output directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override campaign vector output root directory.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional file to write the campaign health summary to.",
    )
    parser.add_argument(
        "--goals-output",
        type=Path,
        help=(
            "Optional compiled strategy-goal ledger path. "
            "Defaults to <output>/strategy_goals.json."
        ),
    )
    parser.add_argument(
        "--expected-goals",
        type=Path,
        help="Existing frozen strategy-goal ledger to use for summary.",
    )
    parser.add_argument(
        "--refresh-goals",
        action="store_true",
        help="Recompute the compiled strategy-goal ledger from the campaign suites.",
    )
    args = parser.parse_args()

    campaign_config_path = resolve_campaign_config_path(args.campaign)
    campaign_config = read_yaml(campaign_config_path)
    output_root = args.output or default_campaign_output_dir(campaign_config, campaign_config_path)
    suite_runs = resolve_campaign_suites(campaign_config, output_root=output_root)
    output_dirs = [output_root]
    generation_configs = [suite_run.generation_config for suite_run in suite_runs]
    goals_output = args.goals_output or goal_ledger_path(output_root)
    expected_goals = None

    if args.generate:
        if output_root.exists():
            shutil.rmtree(output_root)
        for suite_run in suite_runs:
            generation_config = dict(suite_run.generation_config)
            generation_config["keep_existing"] = True
            generate_from_config(generation_config, suite_run.output_dir)
        if args.expected_goals is None:
            expected_goals = load_or_create_expected_goals(
                generation_configs=generation_configs,
                ledger_path=goals_output,
                refresh=True,
            )

    if args.validate and not args.coverage:
        validate_suites(output_dirs)

    coverage_config = campaign_config.get("coverage", {})
    coverage_output = args.coverage_output or Path(coverage_config["output"])
    if args.coverage:
        measure_from_config(coverage_config, test_dir=output_dirs, output_dir=coverage_output)

    if args.summary:
        if args.expected_goals is not None:
            expected_goals = load_expected_goals(args.expected_goals)
        elif expected_goals is None:
            expected_goals = load_or_create_expected_goals(
                generation_configs=generation_configs,
                ledger_path=goals_output,
                refresh=args.refresh_goals,
            )
        summary = format_lean_report(
            test_dirs=output_dirs,
            coverage_dir=coverage_output if args.coverage or coverage_output.exists() else None,
            expected_goals=expected_goals,
            title="State Transition Campaign Summary",
        )
        print(summary)
        if args.summary_output is not None:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(summary)


class CampaignSuiteRun:
    def __init__(self, generation_config: dict, output_dir: Path) -> None:
        self.generation_config = generation_config
        self.output_dir = output_dir


def resolve_campaign_suites(campaign_config: dict, *, output_root: Path) -> list[CampaignSuiteRun]:
    suite_runs = []
    for suite_entry in campaign_config["suites"]:
        suite_name = normalize_suite_entry(suite_entry)
        suite_config_path = resolve_suite_config_path(suite_name)
        suite_config = read_yaml(suite_config_path)
        generation_config = dict(suite_config["generation"])
        suite_runs.append(
            CampaignSuiteRun(
                generation_config=generation_config,
                output_dir=output_root,
            )
        )
    return suite_runs


def normalize_suite_entry(suite_entry) -> str:
    if isinstance(suite_entry, str):
        return suite_entry
    if isinstance(suite_entry, dict):
        if "output" in suite_entry:
            raise ValueError(
                "Campaign suite entries cannot define per-suite output paths. "
                "Use run_campaign --output or campaign-level output instead."
            )
        return suite_entry["suite"]
    raise TypeError(f"Unsupported suite entry: {suite_entry!r}")


if __name__ == "__main__":
    main()
