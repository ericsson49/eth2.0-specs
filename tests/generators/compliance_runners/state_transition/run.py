"""Generate and validate state-transition compliance cases.

Usage:
    uv run python -m tests.generators.compliance_runners.state_transition.run \
        --comptests-output /path/to/output
    uv run python -m tests.generators.compliance_runners.state_transition.run \
        --handler withdrawals --comptests-output /path/to/output
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .catalog import ALL_HANDLERS, FORKS
from .provider import PROFILES, run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork", choices=FORKS, default="gloas")
    parser.add_argument("--handler", choices=(*ALL_HANDLERS, "all"), default="all")
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--preset", choices=("minimal", "mainnet"), default="minimal")
    parser.add_argument("--seed", type=int, help="Seed for deterministic materialization variation")
    parser.add_argument("--comptests-output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.handler, args.comptests_output, args.profile, args.preset, args.seed, args.fork)


if __name__ == "__main__":
    raise SystemExit(main())
