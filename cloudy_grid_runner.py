"""Utilities to generate Cloudy input grids and execute Cloudy over parameter ranges.

This module consolidates earlier ad-hoc scripts into a reusable API that can:

* build regular grids in redshift, metallicity, hydrogen density, and column
  density (plus an optional temperature axis) using helper functions such as
  :func:`custom_arange`;
* write Cloudy ``.in`` files with consistent prefixes so downstream tools can
  locate the resulting ``.col`` files; and
* launch Cloudy on the command line either serially or with a simple
  multi-process pool.

Examples
--------
>>> from cloudy_grid_runner import (
...     CloudyGridDefinition,
...     CloudyRunConfig,
...     custom_arange,
...     generate_input_files,
...     run_cloudy_grid,
... )
>>> definition = CloudyGridDefinition(
...     redshifts=[0.57],
...     metallicities=custom_arange(-1.5, 0.3, 0.3),
...     hydrogen_densities=custom_arange(-4.5, -2.5, 0.3),
...     stopping_columns=custom_arange(16.0, 19.0, 0.3),
...     temperatures=None,
...     stop_label="NHI",
... )
>>> config = CloudyRunConfig(
...     input_dir="/data/cloudy/input",
...     output_dir="/data/cloudy/output",
...     uvb_table="KS19",
...     uvb_reference_redshift=None,
...     stop_mode="neutral",
...     max_workers=4,
... )
>>> jobs = generate_input_files(definition, config)
>>> run_cloudy_grid(jobs, config, skip_completed=True)
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
import math
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:  # optional YAML support
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is optional
    yaml = None


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


DEFAULT_BASE_OUTPUT = Path("/data/mishran/cloudy_outputs")


def custom_arange(start: float, stop: float, step: float, *, decimals: int = 6) -> np.ndarray:
    """Reproduce the legacy ``custom_arange`` behaviour with clean rounding."""

    if step <= 0:
        raise ValueError("step must be positive in custom_arange")
    values = [start]
    next_val = start
    while True:
        next_val = round(next_val + step, decimals)
        if next_val >= stop - 10 ** (-decimals - 2):
            break
        if next_val == -0.0:
            next_val = 0.0
        values.append(next_val)
    if not math.isclose(values[-1], stop, rel_tol=0, abs_tol=10 ** (-decimals)) and stop > values[-1]:
        values.append(round(stop, decimals))
    return np.asarray(values, dtype=float)


def _format_parameter(value: float, digits: int = 4) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def load_config(path: Path | str) -> MutableMapping[str, Any]:
    """Load a JSON or YAML configuration file."""

    cfg_path = Path(path)
    data: MutableMapping[str, Any]
    text = cfg_path.read_text()
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:  # pragma: no cover - optional dependency missing
            raise ImportError("Install PyYAML to load YAML configuration files")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, MutableMapping):
        raise TypeError("Configuration root must be a mapping")
    return data


def _coerce_sequence(entry: Any, name: str) -> Sequence[float]:
    if isinstance(entry, Mapping):
        if {"start", "stop", "step"} <= entry.keys():
            return custom_arange(float(entry["start"]), float(entry["stop"]), float(entry["step"]))
        if "values" in entry:
            entry = entry["values"]
        elif "list" in entry:
            entry = entry["list"]
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        return [float(v) for v in entry]
    raise TypeError(f"Expected a sequence or range definition for '{name}'")


def build_definition_from_config(config: Mapping[str, Any]) -> CloudyGridDefinition:
    grid_cfg = config.get("grid", config)

    def _get(key: str, aliases: Tuple[str, ...]) -> Any:
        if key in grid_cfg:
            return grid_cfg[key]
        for alias in aliases:
            if alias in grid_cfg:
                return grid_cfg[alias]
        raise KeyError(f"Missing grid configuration for '{key}'")

    redshift_vals = _coerce_sequence(_get("z", ("redshift", "redshifts", "z_range")), "redshift")
    metallicity_vals = _coerce_sequence(_get("Z", ("metallicity", "Z_range")), "metallicity")
    density_vals = _coerce_sequence(_get("n_H", ("density", "n_H_range")), "n_H")
    column_vals = _coerce_sequence(_get("N_H", ("NHI", "column", "N_H_range", "stop")), "N_H")

    temperature_vals: Optional[Sequence[float]] = None
    for key in ("T", "temperature", "T_range", "temperatures"):
        if key in grid_cfg:
            temperature_vals = _coerce_sequence(grid_cfg[key], "T")
            break

    stop_label = str(config.get("stop_label", grid_cfg.get("stop_label", "NHI")))

    return CloudyGridDefinition(
        redshifts=redshift_vals,
        metallicities=metallicity_vals,
        hydrogen_densities=density_vals,
        stopping_columns=column_vals,
        temperatures=temperature_vals,
        stop_label=stop_label,
    )


def resolve_output_paths(config: Mapping[str, Any], *, create_dirs: bool = False) -> Tuple[Path, Path, Path]:
    output_cfg = config.get("output", {}) or {}
    if not isinstance(output_cfg, Mapping):
        raise TypeError("'output' section must be a mapping")

    base_dir = Path(output_cfg.get("base_dir", DEFAULT_BASE_OUTPUT))
    uvb_table = str(config.get("uvb_table", "KS19"))
    grid_name = output_cfg.get("grid_name", "grid")
    root = base_dir / uvb_table / str(grid_name)

    input_dir = output_cfg.get("input_dir")
    if input_dir is None:
        input_dir = root / output_cfg.get("input_subdir", "input_files")
    else:
        input_dir = Path(input_dir)

    output_dir = output_cfg.get("output_dir")
    if output_dir is None:
        output_dir = root / output_cfg.get("output_subdir", "output_files")
    else:
        output_dir = Path(output_dir)

    dat_name = output_cfg.get("dat_name", "full_grid.dat")
    dat_dir = output_cfg.get("dat_dir")
    if dat_dir is None:
        dat_path = Path(dat_name)
        if not dat_path.is_absolute():
            dat_path = root / dat_path
    else:
        dat_path = Path(dat_dir) / dat_name

    if create_dirs:
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    return Path(input_dir), Path(output_dir), Path(dat_path)


def build_run_config_from_config(config: Mapping[str, Any]) -> CloudyRunConfig:
    input_dir, output_dir, _ = resolve_output_paths(config)
    run_kwargs = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "uvb_table": config.get("uvb_table", "KS19"),
        "uvb_reference_redshift": config.get("uvb_reference_redshift"),
        "temperature_floor_logk": config.get("temperature_floor_logk", 3.7),
        "radius_kpc": config.get("radius_kpc", 30.0),
        "stop_mode": config.get("stop_mode", "neutral"),
        "include_cmb": config.get("include_cmb", True),
        "include_average_file": config.get("include_average_file", False),
        "overwrite_inputs": config.get("overwrite_inputs", True),
        "max_workers": config.get("max_workers", 1),
        "worker_startup_delay": config.get("worker_startup_delay", 0.0),
        "include_heating_cooling": config.get("include_heating_cooling", False),
    }
    if "cloudy_command" in config:
        run_kwargs["cloudy_command"] = config["cloudy_command"]
    return CloudyRunConfig(**run_kwargs)


@dataclasses.dataclass(frozen=True)
class CloudyGridDefinition:
    """Describe the Cloudy parameter grid to evaluate."""

    redshifts: Sequence[float]
    metallicities: Sequence[float]
    hydrogen_densities: Sequence[float]
    stopping_columns: Sequence[float]
    temperatures: Optional[Sequence[float]] = None
    stop_label: str = "NHI"

    def iter_points(self) -> Iterator[Mapping[str, float]]:
        for combo in itertools.product(
            self.redshifts,
            self.metallicities,
            self.hydrogen_densities,
            self.stopping_columns,
            self.temperatures if self.temperatures is not None else [None],
        ):
            z, Z, n_H, stop, temp = combo
            params = {"z": float(z), "Z": float(Z), "n_H": float(n_H), self.stop_label: float(stop)}
            if temp is not None:
                params["T"] = float(temp)
            yield params


@dataclasses.dataclass
class CloudyRunConfig:
    """Options controlling how Cloudy input files are written and executed."""

    input_dir: Path | str
    output_dir: Path | str
    cloudy_command: str = "cloudy"
    uvb_table: str = "KS19"
    uvb_reference_redshift: Optional[float] = None
    temperature_floor_logk: float = 3.7
    radius_kpc: float = 30.0
    stop_mode: str = "neutral"  # accepted: neutral, total
    include_cmb: bool = True
    include_average_file: bool = False
    overwrite_inputs: bool = True
    max_workers: int = 1
    worker_startup_delay: float = 0.0
    include_heating_cooling: bool = False

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.stop_mode not in {"neutral", "total"}:
            raise ValueError("stop_mode must be 'neutral' or 'total'")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")


@dataclasses.dataclass
class CloudyJob:
    """Container describing one Cloudy run (input/output paths)."""

    prefix: str
    input_file: Path
    output_file: Path
    expected_columns_file: Path


# -----------------------------------------------------------------------------
# Input generation
# -----------------------------------------------------------------------------


def build_file_prefix(params: Mapping[str, float], stop_label: str) -> str:
    parts = [
        f"z={_format_parameter(params['z'])}",
        f"Z={_format_parameter(params['Z'])}",
        f"nh={_format_parameter(params['n_H'])}",
        f"{stop_label}={_format_parameter(params[stop_label])}",
    ]
    if "T" in params:
        parts.append(f"T={_format_parameter(params['T'])}")
    return "".join(parts)


def _render_cloudy_input(params: Mapping[str, float], definition: CloudyGridDefinition, config: CloudyRunConfig) -> str:
    prefix = build_file_prefix(params, definition.stop_label)
    uvb_redshift = config.uvb_reference_redshift if config.uvb_reference_redshift is not None else params["z"]
    lines: List[str] = ["title CGM"]
    if config.include_cmb:
        lines.append(f"cmb redshift {params['z']}")
    lines.append(f"table {config.uvb_table} redshift {uvb_redshift}")
    lines.append(f"metals {params['Z']} log")
    lines.append(f"hden {params['n_H']} log")
    lines.append(f"radius {config.radius_kpc}")
    if "T" in params:
        lines.append(f"constant temperature {params['T']} log")
    else:
        lines.append(f"set temperature floor {config.temperature_floor_logk}")
    if config.stop_mode == "neutral":
        lines.append(f"stop neutral column {params[definition.stop_label]} log")
    else:
        lines.append(f"stop column {params[definition.stop_label]} log")
    lines.append("double optical depths")
    lines.append("set trim -20")
    lines.append(f"save species column density last \"{prefix}.col\" all")
    if config.include_average_file:
        lines.extend(
            [
                f"save averages, \"{prefix}.avr\" last no clobber",
                "temperature, hydrogen 1 over volume",
                "end of averages",
            ]
        )
    if config.include_heating_cooling:
        lines.extend(
            [
                f"save heating \"{prefix}.heat\" last",
                f"save cooling \"{prefix}.cool\" last",
            ]
        )
    lines.append("iterate to convergence")
    lines.append("")
    return "\n".join(lines)


def generate_input_files(definition: CloudyGridDefinition, config: CloudyRunConfig) -> List[CloudyJob]:
    config.input_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[CloudyJob] = []
    for params in definition.iter_points():
        prefix = build_file_prefix(params, definition.stop_label)
        input_path = config.input_dir / f"{prefix}.in"
        output_path = config.output_dir / f"{prefix}.out"
        column_path = config.output_dir / f"{prefix}.col"
        if config.overwrite_inputs or not input_path.exists():
            text = _render_cloudy_input(params, definition, config)
            input_path.write_text(text)
        jobs.append(CloudyJob(prefix=prefix, input_file=input_path, output_file=output_path, expected_columns_file=column_path))
    logger.info("Prepared %s Cloudy input files in %s", len(jobs), config.input_dir)
    return jobs


# -----------------------------------------------------------------------------
# Execution helpers
# -----------------------------------------------------------------------------


def _run_job(job: CloudyJob, config: CloudyRunConfig, skip_completed: bool) -> str:
    if skip_completed and job.expected_columns_file.exists():
        logger.info("Skipping Cloudy job %s (columns file already present)", job.prefix)
        return "skipped"

    with job.input_file.open("r", encoding="utf-8") as src, job.output_file.open("w", encoding="utf-8") as dst:
        try:
            subprocess.run(
                [config.cloudy_command],
                stdin=src,
                stdout=dst,
                stderr=subprocess.STDOUT,
                cwd=config.output_dir,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Cloudy run failed for %s with code %s", job.prefix, exc.returncode)
            raise
    logger.info("Completed Cloudy job %s", job.prefix)
    return "completed"


def run_cloudy_grid(jobs: Sequence[CloudyJob], config: CloudyRunConfig, *, skip_completed: bool = True) -> None:
    """Execute Cloudy for the provided job list."""

    total_jobs = len(jobs)
    if total_jobs == 0:
        logger.info("No Cloudy jobs to run")
        return

    if config.max_workers == 1:
        for idx, job in enumerate(jobs, start=1):
            logger.info("Starting Cloudy job %d/%d: %s", idx, total_jobs, job.prefix)
            outcome = _run_job(job, config, skip_completed=skip_completed)
            logger.info("Finished Cloudy job %d/%d: %s (%s)", idx, total_jobs, job.prefix, outcome)
            if config.worker_startup_delay > 0:
                time.sleep(config.worker_startup_delay)
        return

    with multiprocessing.Pool(processes=config.max_workers) as pool:
        pool.starmap(
            _pool_run_job,
            [
                (idx, total_jobs, job, config, skip_completed, config.worker_startup_delay)
                for idx, job in enumerate(jobs, start=1)
            ],
        )


def _pool_run_job(index: int, total: int, job: CloudyJob, config: CloudyRunConfig, skip_completed: bool, delay: float) -> str:
    """Wrapper so ``multiprocessing.Pool.starmap`` can run :func:`_run_job`."""
    logger.info("Starting Cloudy job %d/%d: %s", index, total, job.prefix)
    if delay > 0:
        time.sleep(delay)
    outcome = _run_job(job, config, skip_completed=skip_completed)
    logger.info("Finished Cloudy job %d/%d: %s (%s)", index, total, job.prefix, outcome)
    return outcome


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Cloudy input grids and execute Cloudy runs")
    parser.add_argument("config", help="Path to JSON or YAML configuration file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help="Ignore existing .col files and rerun every grid point",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only prepare input files; skip executing Cloudy",
    )
    args = parser.parse_args(argv)

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level)
    else:
        logging.getLogger().setLevel(level)

    config_path = Path(args.config)
    config_data = load_config(config_path)
    definition = build_definition_from_config(config_data)
    run_config = build_run_config_from_config(config_data)
    input_dir, output_dir, dat_path = resolve_output_paths(config_data, create_dirs=True)

    logger.info("Using input directory %s", input_dir)
    logger.info("Using output directory %s", output_dir)

    jobs = generate_input_files(definition, run_config)
    logger.info("Prepared %d Cloudy jobs", len(jobs))

    if args.dry_run:
        logger.info("Dry run requested; skipping Cloudy execution")
    else:
        run_cloudy_grid(jobs, run_config, skip_completed=not args.rerun_all)

    logger.info("Configured grid definition: z=%s, Z=%s, n_H=%s, %s=%s",
                definition.redshifts,
                definition.metallicities,
                definition.hydrogen_densities,
                definition.stop_label,
                definition.stopping_columns)
    logger.info("Once Cloudy runs finish, assemble the grid via: python cloudy_grid_assembler.py %s", config_path)
    logger.info("Planned .dat output: %s", dat_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())


__all__ = [
    "CloudyGridDefinition",
    "CloudyRunConfig",
    "CloudyJob",
    "custom_arange",
    "load_config",
    "build_definition_from_config",
    "build_run_config_from_config",
    "resolve_output_paths",
    "build_file_prefix",
    "generate_input_files",
    "run_cloudy_grid",
    "main",
]
