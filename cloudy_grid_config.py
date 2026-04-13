"""Common configuration loader for Cloudy grid generation/assembly scripts."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from cloudy_grid_runner import CloudyGridDefinition, custom_arange
from cloudy_grid_assembler import GridAssemblyConfig


@dataclasses.dataclass
class AxisConfig:
    name: str
    values: Sequence[float]

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> "AxisConfig":
        if "values" in data:
            values = [float(v) for v in data["values"]]
        else:
            start = float(data["start"])
            stop = float(data["stop"])
            step = float(data["step"])
            decimals = int(data.get("decimals", 6))
            values = custom_arange(start, stop, step, decimals=decimals)
        return cls(name=name, values=values)


@dataclasses.dataclass
class RunnerPathConfig:
    output_root: Path
    input_subdir: str
    output_subdir: str
    dat_filename: str

    @property
    def input_dir(self) -> Path:
        return self.output_root / self.input_subdir

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.output_subdir

    @property
    def dat_path(self) -> Path:
        return self.output_root / self.dat_filename


@dataclasses.dataclass
class RunnerOptions:
    cloudy_command: str = "cloudy"
    uvb_table: str = "KS19"
    uvb_reference_redshift: Optional[float] = None
    stop_mode: str = "neutral"
    include_cmb: bool = True
    include_average_file: bool = False
    temperature_floor_logk: float = 3.7
    radius_kpc: float = 30.0
    worker_startup_delay: float = 0.0
    max_workers: int = 1
    include_heating_cooling: bool = False
    mode: str = "pie"
    td_n_timesteps: int = 300
    td_t_floor_kelvin: float = 1e4
    td_stop_age_years: float = 4.4e17


@dataclasses.dataclass
class AssemblerOptions:
    ions: Optional[Sequence[str]] = None
    include_temperature: bool = False
    log_columns: bool = False
    ion_aliases: Optional[Mapping[str, str]] = None
    stop_alias_overrides: Optional[Mapping[str, str]] = None
    include_heating_cooling: bool = False


@dataclasses.dataclass
class CloudyPipelineConfig:
    grid_definition: CloudyGridDefinition
    paths: RunnerPathConfig
    runner: RunnerOptions
    assembler: AssemblerOptions


def _parse_axis_block(name: str, data: Mapping[str, Any]) -> AxisConfig:
    return AxisConfig.from_mapping(name, data)


def _load_paths(data: Mapping[str, Any]) -> RunnerPathConfig:
    output_root = Path(data.get("output_root", "/data/mishran/cloudy_outputs"))
    input_subdir = data.get("input_subdir", "input_files")
    output_subdir = data.get("output_subdir", "output_files")
    dat_filename = data.get("dat_filename", "full_grid.dat")
    return RunnerPathConfig(output_root=output_root, input_subdir=input_subdir, output_subdir=output_subdir, dat_filename=dat_filename)


def _load_runner_options(data: Mapping[str, Any]) -> RunnerOptions:
    return RunnerOptions(
        cloudy_command=data.get("cloudy_command", "cloudy"),
        uvb_table=data.get("uvb_table", "KS19"),
        uvb_reference_redshift=data.get("uvb_reference_redshift"),
        stop_mode=data.get("stop_mode", "neutral"),
        include_cmb=bool(data.get("include_cmb", True)),
        include_average_file=bool(data.get("include_average_file", False)),
        temperature_floor_logk=float(data.get("temperature_floor_logk", 3.7)),
        radius_kpc=float(data.get("radius_kpc", 30.0)),
        worker_startup_delay=float(data.get("worker_startup_delay", 0.0)),
        max_workers=int(data.get("max_workers", 1)),
        include_heating_cooling=bool(data.get("include_heating_cooling", False)),
        mode=str(data.get("mode", "pie")),
        td_n_timesteps=int(data.get("td_n_timesteps", 300)),
        td_t_floor_kelvin=float(data.get("td_t_floor_kelvin", 1e4)),
        td_stop_age_years=float(data.get("td_stop_age_years", 4.4e17)),
    )


def _load_assembler_options(data: Mapping[str, Any]) -> AssemblerOptions:
    return AssemblerOptions(
        ions=data.get("ions"),
        include_temperature=bool(data.get("include_temperature", False)),
        log_columns=bool(data.get("log_columns", False)),
        ion_aliases=data.get("ion_aliases"),
        stop_alias_overrides=data.get("stop_alias_overrides"),
        include_heating_cooling=bool(data.get("include_heating_cooling", False)),
    )


def load_cloudy_config(path: Path | str) -> CloudyPipelineConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)

    grid_data = data.get("grid", {})
    stop_label = grid_data.get("stop_label", "N_H")
    mode = str(data.get("mode", data.get("runner", {}).get("mode", "pie")))

    redshift_axis = _parse_axis_block("z", grid_data["z"])
    metallicity_axis = _parse_axis_block("Z", grid_data["Z"])
    density_axis = _parse_axis_block("n_H", grid_data["n_H"])

    if mode == "td":
        stop_axis = AxisConfig(name=stop_label, values=[0.0])
    else:
        stop_axis = _parse_axis_block(stop_label, grid_data[stop_label])

    temp_axis = None
    if "T" in grid_data:
        temp_block = grid_data["T"]
        enabled = bool(temp_block.get("enabled", True))
        if enabled:
            temp_axis = _parse_axis_block("T", temp_block)

    t_init_axis = None
    if "T_init" in grid_data:
        t_init_axis = _parse_axis_block("T_init", grid_data["T_init"])

    definition = CloudyGridDefinition(
        redshifts=np.asarray(redshift_axis.values, dtype=float),
        metallicities=np.asarray(metallicity_axis.values, dtype=float),
        hydrogen_densities=np.asarray(density_axis.values, dtype=float),
        stopping_columns=np.asarray(stop_axis.values, dtype=float),
        temperatures=None if temp_axis is None else np.asarray(temp_axis.values, dtype=float),
        stop_label=stop_label,
        t_init_values=None if t_init_axis is None else np.asarray(t_init_axis.values, dtype=float),
    )

    paths = _load_paths(data.get("paths", {}))
    runner = _load_runner_options(data.get("runner", {}))
    assembler = _load_assembler_options(data.get("assembler", {}))

    return CloudyPipelineConfig(
        grid_definition=definition,
        paths=paths,
        runner=runner,
        assembler=assembler,
    )


def build_assembly_config(config: CloudyPipelineConfig) -> GridAssemblyConfig:
    options = config.assembler
    return GridAssemblyConfig(
        output_path=config.paths.dat_path,
        ion_aliases=options.ion_aliases,
        include_temperature=options.include_temperature,
        log_columns=options.log_columns,
        stop_alias_overrides=options.stop_alias_overrides,
        include_heating_cooling=options.include_heating_cooling,
        mode=config.runner.mode,
    )


__all__ = [
    "AxisConfig",
    "RunnerPathConfig",
    "RunnerOptions",
    "AssemblerOptions",
    "CloudyPipelineConfig",
    "load_cloudy_config",
    "build_assembly_config",
]
