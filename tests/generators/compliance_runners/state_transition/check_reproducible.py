from __future__ import annotations

import argparse
import filecmp
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .generate_vectors import generate_vectors, normalize_handlers
from .suite_config import read_yaml, resolve_suite_config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that a state-transition suite config generates reproducible vectors"
    )
    parser.add_argument(
        "--suite",
        default="electra_operations_guided",
        help="Suite config name or path. Defaults to electra_operations_guided.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep comparison directories for inspection.",
    )
    args = parser.parse_args()

    result = check_suite_reproducible(args.suite, keep_temp=args.keep_temp)
    print(result.format())
    raise SystemExit(0 if result.reproducible else 1)


def check_suite_reproducible(suite: str, *, keep_temp: bool = False) -> ReproducibilityResult:
    suite_config_path = resolve_suite_config_path(suite)
    suite_config = read_yaml(suite_config_path)
    generation_config = dict(suite_config["generation"])
    generation_config["keep_existing"] = False

    if keep_temp:
        temp_path = Path(tempfile.mkdtemp(prefix="state-transition-repro-"))
        left = temp_path / "first"
        right = temp_path / "second"
        return run_reproducibility_check(generation_config, left=left, right=right)

    with tempfile.TemporaryDirectory(prefix="state-transition-repro-") as temp_dir:
        temp_path = Path(temp_dir)
        left = temp_path / "first"
        right = temp_path / "second"
        return run_reproducibility_check(generation_config, left=left, right=right)


def run_reproducibility_check(
    generation_config: dict,
    *,
    left: Path,
    right: Path,
) -> ReproducibilityResult:
    generate_from_config(generation_config, left)
    generate_from_config(generation_config, right)
    return compare_trees(left, right)


def generate_from_config(generation_config: dict, output_dir: Path) -> None:
    generate_vectors(
        output_dir=output_dir,
        fork_name=generation_config["fork"],
        preset_name=generation_config["preset"],
        handlers=normalize_handlers(
            generation_config.get("handlers"),
            stages=generation_config.get("stages"),
        ),
        per_handler_limit=generation_config["per_handler_limit"],
        changed_only=generation_config.get("changed_only", False),
        unchanged_only=generation_config.get("unchanged_only", False),
        invalid_only=generation_config.get("invalid_only", False),
        guided=generation_config.get("guided", False),
        mode=generation_config.get("mode"),
        profile_dimensions=generation_config.get("profile_dimensions"),
        profile_interaction_order=generation_config.get("profile_interaction_order", 2),
        input_profile_order=generation_config.get("input_profile_order", 1),
        keep_existing=False,
        distribution=generation_config.get("distribution"),
    )


def compare_trees(left: Path, right: Path) -> ReproducibilityResult:
    left_files = collect_files(left)
    right_files = collect_files(right)
    missing = sorted(str(path) for path in right_files - left_files)
    extra = sorted(str(path) for path in left_files - right_files)
    common = sorted(left_files & right_files)
    different = [
        str(path)
        for path in common
        if not filecmp.cmp(left / path, right / path, shallow=False)
    ]
    return ReproducibilityResult(
        left=left,
        right=right,
        compared=len(common),
        missing=missing,
        extra=extra,
        different=different,
    )


def collect_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


@dataclass(frozen=True)
class ReproducibilityResult:
    left: Path
    right: Path
    compared: int
    missing: list[str]
    extra: list[str]
    different: list[str]

    @property
    def reproducible(self) -> bool:
        return not self.missing and not self.extra and not self.different

    def format(self) -> str:
        lines = ["State Transition Reproducibility", "================================", ""]
        lines.append(f"first output:  {self.left}")
        lines.append(f"second output: {self.right}")
        lines.append(f"compared files: {self.compared}")
        if self.reproducible:
            lines.append("result: reproducible")
            return "\n".join(lines)

        lines.append("result: different")
        lines.extend(format_paths("missing from first", self.missing))
        lines.extend(format_paths("extra in first", self.extra))
        lines.extend(format_paths("different content", self.different))
        return "\n".join(lines)


def format_paths(label: str, paths: list[str]) -> list[str]:
    if not paths:
        return [f"{label}: none"]
    lines = [f"{label}: {len(paths)}"]
    lines.extend(f"  {path}" for path in paths[:20])
    if len(paths) > 20:
        lines.append(f"  ... {len(paths) - 20} more")
    return lines


if __name__ == "__main__":
    main()
