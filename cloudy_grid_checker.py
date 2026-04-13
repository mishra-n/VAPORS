"""List missing Cloudy ``.col`` files for a configured parameter grid.

This helper reads the same YAML/JSON config consumed by ``cloudy_grid_runner``
and checks whether each expected ``.col`` output exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from cloudy_grid_runner import build_definition_from_config, build_file_prefix, load_config, resolve_output_paths


def _is_empty_col_file(path: Path) -> bool:
    """Return True when a ``.col`` file has no meaningful content."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return False
    except OSError:
        return True
    return True


def find_missing_and_empty_col_files(
    config_path: Path | str,
    *,
    output_dir: Optional[Path | str] = None,
) -> Tuple[list[Path], list[Path]]:
    """Return expected ``.col`` paths that are missing and files that are empty."""

    config = load_config(config_path)
    definition = build_definition_from_config(config)
    if output_dir is None:
        _, resolved_output_dir, _ = resolve_output_paths(config)
    else:
        resolved_output_dir = Path(output_dir)

    resolved_output_dir = Path(resolved_output_dir)
    missing: list[Path] = []
    empty: list[Path] = []
    for params in definition.iter_points():
        prefix = build_file_prefix(params, definition.stop_label)
        col_path = resolved_output_dir / f"{prefix}.col"
        if not col_path.exists():
            missing.append(col_path)
            continue
        if _is_empty_col_file(col_path):
            empty.append(col_path)
    return missing, empty


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="List missing Cloudy .col files for a grid config")
    parser.add_argument("config", help="Path to Cloudy grid YAML/JSON config")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override (otherwise uses config output settings)",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print only the number of missing files",
    )
    args = parser.parse_args(argv)

    missing, empty = find_missing_and_empty_col_files(args.config, output_dir=args.output_dir)
    if args.count_only:
        print(len(missing) + len(empty))
        return 0 if not missing and not empty else 1

    if not missing and not empty:
        print("No missing or empty .col files found.")
        return 0

    print(f"Missing .col files: {len(missing)}")
    for path in missing:
        print(path)
    print(f"Empty .col files: {len(empty)}")
    for path in empty:
        print(path)
    return 1


if __name__ == "__main__":
    sys.exit(main())
