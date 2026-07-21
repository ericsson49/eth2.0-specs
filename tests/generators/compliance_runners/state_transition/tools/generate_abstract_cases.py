from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ruamel.yaml import YAML

from ..abstract_cases import (
    enumerate_abstract_cases,
    select_abstract_cases,
    transpile_validator_state_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate abstract state-transition cases from Python MiniZinc models"
    )
    parser.add_argument(
        "--emit-mzn",
        type=Path,
        help="Write the transpiled MiniZinc model instead of solving cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of raw solver-order abstract cases to emit.",
    )
    parser.add_argument(
        "--per-handler-limit",
        type=int,
        default=5,
        help="Number of abstract cases to select for each requested handler.",
    )
    parser.add_argument(
        "--handler",
        action="append",
        help="Handler to select. Can be repeated. Defaults to all known handlers.",
    )
    args = parser.parse_args()

    if args.emit_mzn is not None:
        args.emit_mzn.write_text(transpile_validator_state_model())
        return

    yaml = YAML()
    if args.limit is None:
        if args.handler is None:
            abstract_cases = select_abstract_cases(per_handler_limit=args.per_handler_limit)
        else:
            abstract_cases = select_abstract_cases(
                per_handler_limit=args.per_handler_limit,
                handlers=args.handler,
            )
    else:
        abstract_cases = enumerate_abstract_cases(limit=args.limit)

    cases = [
        {
            "handler": case.handler_name,
            "case": case.case_name,
            "profile": case.profile,
        }
        for case in abstract_cases
    ]
    yaml.dump({"cases": cases}, sys.stdout)


if __name__ == "__main__":
    main()
