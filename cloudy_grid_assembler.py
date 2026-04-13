"""Assemble Cloudy ``.col`` outputs into a consolidated data table.

This module works hand-in-hand with :mod:`cloudy_grid_runner` by consuming the
files produced for each grid point and writing a single ``.dat`` table in the
format previously used for ``full_grid.dat``. The API allows users to restrict
which ions are included and to customise the mapping between Cloudy column names
(e.g. ``C+2``) and astrophysical notation (e.g. ``N_CIII``).
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from astropy.io import ascii
from astropy.table import QTable


if not hasattr(np, "asscalar"):
    def _asscalar(array: np.ndarray) -> Any:
        return array.item()

    np.asscalar = _asscalar  # type: ignore[attr-defined]


def _read_cloudy_columns(col_path: Path) -> Mapping[str, float]:
    """Parse a Cloudy ``.col`` file into a mapping of column name -> value."""

    header_line: Optional[str] = None
    data_line: Optional[str] = None

    with col_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_line = stripped.lstrip("#").strip()
                continue
            data_line = stripped
            break

    if header_line is None or data_line is None:
        raise ValueError(f"File {col_path} does not contain a recognised header/data pair")

    header_tokens = header_line.replace("\t", " ").split()
    data_tokens = data_line.replace("\t", " ").split()

    if len(header_tokens) >= 3 and header_tokens[0].lower() == "column" and header_tokens[1].lower() == "density":
        merged = " ".join(header_tokens[:3])
        header_tokens = [merged] + header_tokens[3:]

    if len(header_tokens) != len(data_tokens):
        raise ValueError(
            f"Header/data length mismatch in {col_path}: {len(header_tokens)} vs {len(data_tokens)}"
        )

    values = {}
    for name, token in zip(header_tokens, data_tokens):
        try:
            values[name] = float(token)
        except ValueError as exc:
            raise ValueError(f"Unable to parse value {token!r} for column {name!r} in {col_path}") from exc
    return values


def _read_cloudy_average_temperature(avr_path: Path) -> Optional[float]:
    """Extract the hydrogen-weighted temperature saved by Cloudy.

    The ``save averages`` output stores a short header line followed by the
    numeric values. We return the first finite number encountered which
    corresponds to the hydrogen-weighted temperature in Kelvin.
    """

    if not avr_path.exists():
        return None

    try:
        with avr_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    value = float(stripped.split()[0])
                except (ValueError, IndexError):
                    continue
                if np.isfinite(value) and value > 0:
                    return value
    except OSError:
        return None
    return None


from typing import List as _List

from cloudy_grid_runner import (
    CloudyGridDefinition,
    build_definition_from_config,
    build_file_prefix,
    custom_arange,
    load_config,
    resolve_output_paths,
)


# ---------------------------------------------------------------------------
# Time-dependent output file parsers
# ---------------------------------------------------------------------------

TIM_COLUMNS = (
    "time_elapsed", "timestep", "continuum_scale", "density",
    "T_mean", "H+", "H0", "H2", "He+", "CO_H", "z_current", "ne_nH",
)


def _read_td_tim_file(tim_path: Path) -> _List[Dict[str, float]]:
    """Parse a Cloudy ``save time dependent`` ``.tim`` file.

    Returns one dict per time-step with keys from :data:`TIM_COLUMNS`.
    """
    rows: _List[Dict[str, float]] = []
    with tim_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.replace("\t", " ").split()
            if len(tokens) < len(TIM_COLUMNS):
                continue
            try:
                vals = [float(t) for t in tokens[: len(TIM_COLUMNS)]]
            except ValueError:
                continue
            rows.append(dict(zip(TIM_COLUMNS, vals)))
    return rows


def _read_td_col_file(
    col_path: Path,
) -> Tuple[_List[str], _List[_List[float]]]:
    """Parse a multi-row Cloudy ``.col`` file (written without ``last``).

    Returns ``(header_names, data_rows)`` where each element of
    *data_rows* is a list of floats aligned with *header_names*.
    """
    header_names: Optional[_List[str]] = None
    data_rows: _List[_List[float]] = []

    with col_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                candidate = line.lstrip("#").strip()
                if candidate:
                    header_names = candidate.replace("\t", " ").split()
                continue
            tokens = line.replace("\t", " ").split()
            try:
                vals = [float(t) for t in tokens]
            except ValueError:
                continue
            data_rows.append(vals)

    if header_names is None:
        raise ValueError(f"No header line found in {col_path}")

    # Merge "column density X" prefix if present
    if (
        len(header_names) >= 3
        and header_names[0].lower() == "column"
        and header_names[1].lower() == "density"
    ):
        merged = " ".join(header_names[:3])
        header_names = [merged] + header_names[3:]

    return header_names, data_rows

logger = logging.getLogger(__name__)

DEFAULT_ION_ALIASES: Mapping[str, str] = {
    "N_HI": "column density H",
    "N_H_total": "hydrogen column density",
    "N_CII": "C+",
    "N_CIII": "C+2",
    "N_CIV": "C+3",
    "N_OII": "O+",
    "N_OIII": "O+2",
    'N_OIV': 'O+3',
    "N_OVI": "O+5",
    "N_SII": "S+",
    "N_SIII": "S+2",
    "N_SIV": "S+3",
    "N_SV": "S+4",
    "N_FeII": "Fe+",
    "N_FeIII": "Fe+2",
    "N_FeIV": "Fe+3",
    "N_NeVIII": "Ne+7",
    "N_NII": "N+",
    "N_NIII": "N+2",
    "N_NIV": "N+3",
    "N_NV": "N+4",
    "N_SiII": "Si+",
    "N_SiIII": "Si+2",
    "N_SiIV": "Si+3",
}

STOP_ALIAS_NEUTRAL = "column density H"
STOP_ALIAS_TOTAL = "hydrogen column density"
K_BOLTZMANN = 1.380649e-16  # erg K^-1


def _safe_log10(value: Optional[float]) -> float:
    if value is None or not np.isfinite(value) or value <= 0:
        return float("nan")
    return float(np.log10(value))


def _parse_save_heat(path: Path) -> Tuple[float, float, float, float, float]:
    depth = temperature = heat_total = cool_total = dyn_heat = float("nan")
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            depth, temperature, heat_total, cool_total, dyn_heat = (float(part) for part in parts[:5])
            break
    if not np.isfinite(heat_total) or not np.isfinite(cool_total):
        raise ValueError(f"Failed to parse heating summary from {path}")
    return depth, temperature, heat_total, cool_total, dyn_heat


def _compute_cooling_time(
    *,
    temperature: Optional[float],
    log_density: Optional[float],
    heat_total: Optional[float],
    cool_total: Optional[float],
) -> Optional[float]:
    if temperature is None or log_density is None:
        return None
    if not np.isfinite(temperature) or not np.isfinite(log_density):
        return None
    if heat_total is None or cool_total is None:
        return None
    net_cooling = float(cool_total) - float(heat_total)
    if not np.isfinite(net_cooling) or net_cooling <= 0:
        return None
    density = 10.0 ** float(log_density)
    if density <= 0 or not np.isfinite(density):
        return None
    energy_density = 1.5 * K_BOLTZMANN * float(temperature) * density
    if energy_density <= 0 or not np.isfinite(energy_density):
        return None
    cooling_time = energy_density / net_cooling
    if cooling_time <= 0 or not np.isfinite(cooling_time):
        return None
    return float(cooling_time)

def _format_parameter(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def _determine_stop_alias(stop_label: str, overrides: Optional[Mapping[str, str]]) -> str:
    if overrides and stop_label in overrides:
        return overrides[stop_label]
    if stop_label.upper().startswith("NHI"):
        return STOP_ALIAS_NEUTRAL
    return STOP_ALIAS_TOTAL


@dataclasses.dataclass
class GridAssemblyConfig:
    output_path: Path | str
    ion_aliases: Mapping[str, str] | None = None
    include_temperature: bool = False
    log_columns: bool = False
    stop_alias_overrides: Mapping[str, str] | None = None
    overwrite: bool = True
    include_heating_cooling: bool = False
    mode: str = "pie"  # "pie" or "td"

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)


def assemble_cloudy_table(
    definition: CloudyGridDefinition,
    output_dir: Path | str,
    *,
    ions: Optional[Sequence[str]] = None,
    config: Optional[GridAssemblyConfig] = None,
) -> QTable:
    """Read Cloudy ``.col`` outputs for each grid point and build a ``QTable``."""

    output_dir = Path(output_dir)
    if config is None:
        config = GridAssemblyConfig(output_path=output_dir / "full_grid.dat")

    alias_map: Dict[str, str] = dict(DEFAULT_ION_ALIASES)
    if config.ion_aliases:
        alias_map.update(config.ion_aliases)

    stop_alias = _determine_stop_alias(definition.stop_label, config.stop_alias_overrides)
    alias_map.setdefault("N_HI", STOP_ALIAS_NEUTRAL)
    alias_map.setdefault("N_H_total", STOP_ALIAS_TOTAL)
    alias_map.setdefault(definition.stop_label, stop_alias)

    if ions is None:
        ions = [key for key in alias_map.keys() if key != definition.stop_label]
    else:
        ions = list(ions)

    for required in ("N_HI", "N_H_total"):
        if required not in ions:
            ions.append(required)

    include_temperature_outputs = bool(config.include_temperature)
    include_heating_outputs = bool(config.include_heating_cooling)

    columns: list[str] = ["z", "Z", "n_H"]
    if include_temperature_outputs and definition.temperatures is not None:
        columns.append("T")
    columns.append(definition.stop_label)
    if include_temperature_outputs:
        columns.append("T_cloudy")
    columns.extend(ions)
    if include_heating_outputs:
        columns.extend(["log_heat_rate", "log_cool_rate", "log_tcool"])

    table = QTable(names=columns)

    for params in definition.iter_points():
        prefix = build_file_prefix(params, definition.stop_label)
        col_path = output_dir / f"{prefix}.col"
        if not col_path.exists():
            raise FileNotFoundError(f"Missing Cloudy output file: {col_path}")
        try:
            data = ascii.read(col_path, format="commented_header", guess=True)
            data_map = {name: float(data[name][0]) for name in data.colnames}
        except Exception:
            data_map = _read_cloudy_columns(col_path)

        row = [params["z"], params["Z"], params["n_H"]]
        cloudy_temperature_value = float("nan")
        if include_temperature_outputs and definition.temperatures is not None:
            row.append(params.get("T"))
        requested_stop = params.get(definition.stop_label)
        if requested_stop is None:
            raise KeyError(f"Grid parameters missing stop value '{definition.stop_label}' for {col_path}")
        row.append(float(requested_stop))
        if include_temperature_outputs:
            avr_path = output_dir / f"{prefix}.avr"
            cloudy_temperature = _read_cloudy_average_temperature(avr_path)
            if cloudy_temperature is None:
                logger.debug("Cloudy average file missing or unreadable for %s", avr_path)
                row.append(float("nan"))
                cloudy_temperature_value = float("nan")
            else:
                cloudy_temperature_value = float(cloudy_temperature)
                row.append(cloudy_temperature_value)
        if stop_alias not in data_map:
            logger.debug(
                "Stop column alias '%s' not present in %s; using configured value only",
                stop_alias,
                col_path,
            )

        def _lookup(alias: Optional[str]) -> Optional[float]:
            if alias and alias in data_map:
                return float(data_map[alias])
            return None

        neutral_output = _lookup(alias_map.get("N_HI", STOP_ALIAS_NEUTRAL))
        total_output = _lookup(alias_map.get("N_H_total", STOP_ALIAS_TOTAL))
        if total_output is None:
            # Derive total hydrogen column from neutral and ionised components when Cloudy omits it.
            h_plus = _lookup("H+")
            neutral_for_total = neutral_output
            if neutral_for_total is None:
                neutral_for_total = _lookup(STOP_ALIAS_NEUTRAL)
            if neutral_for_total is None and h_plus is None:
                total_output = None
            else:
                base_neutral = 0.0 if neutral_for_total is None else neutral_for_total
                base_ionised = 0.0 if h_plus is None else h_plus
                total_output = base_neutral + base_ionised

        for ion in ions:
            if ion == "N_H_total":
                if total_output is None:
                    raise KeyError(f"Could not determine total hydrogen column in {col_path}")
                value = float(total_output)
            else:
                alias = alias_map.get(ion)
                if alias is None:
                    raise KeyError(f"No alias defined for ion '{ion}'")
                if alias not in data_map:
                    raise KeyError(f"Alias '{alias}' for ion '{ion}' missing in {col_path}")
                value = float(data_map[alias])
            if config.log_columns and ion != definition.stop_label:
                value = float(np.log10(value))
            row.append(value)

        if include_heating_outputs:
            heat_path = output_dir / f"{prefix}.heat"
            cool_path = output_dir / f"{prefix}.cool"
            if not heat_path.exists():
                raise FileNotFoundError(f"Missing Cloudy heating output file: {heat_path}")
            if not cool_path.exists():
                raise FileNotFoundError(f"Missing Cloudy cooling output file: {cool_path}")
            _, heating_temperature, heat_total, cool_total, _ = _parse_save_heat(heat_path)

            log_heat_rate = _safe_log10(heat_total)
            log_cool_rate = _safe_log10(cool_total)

            temperature_for_cooling: Optional[float] = heating_temperature if np.isfinite(heating_temperature) else None
            if temperature_for_cooling is None or temperature_for_cooling <= 0:
                if include_temperature_outputs and np.isfinite(cloudy_temperature_value):
                    temperature_for_cooling = cloudy_temperature_value
                else:
                    grid_temperature_log = params.get("T") if definition.temperatures is not None else None
                    if grid_temperature_log is not None and np.isfinite(grid_temperature_log):
                        temperature_for_cooling = 10.0 ** float(grid_temperature_log)

            cooling_time = _compute_cooling_time(
                temperature=temperature_for_cooling,
                log_density=params.get("n_H"),
                heat_total=heat_total,
                cool_total=cool_total,
            )

            row.extend([log_heat_rate, log_cool_rate, _safe_log10(cooling_time)])
        table.add_row(row)

    return table


def assemble_cloudy_td_table(
    definition: CloudyGridDefinition,
    output_dir: Path | str,
    *,
    ions: Optional[Sequence[str]] = None,
    config: Optional[GridAssemblyConfig] = None,
) -> QTable:
    """Read Cloudy TD ``.col`` / ``.tim`` outputs and build a time-series ``QTable``.

    Each grid point produces *multiple* rows (one per time-step).  The table
    columns are ``z, Z, n_H, T_init, time_elapsed, temperature, [ion cols]``.
    Ion columns store **raw single-zone column densities** (linear, cm^-2) as
    output by Cloudy.  Downstream inference (``CloudyGridInterpolator.from_td_table``)
    converts these to ion fractions (``N_ion / N_H_total``) and resamples onto a
    regular temperature grid.
    """

    output_dir = Path(output_dir)
    if config is None:
        config = GridAssemblyConfig(output_path=output_dir / "full_td_grid.dat", mode="td")

    alias_map: Dict[str, str] = dict(DEFAULT_ION_ALIASES)
    if config.ion_aliases:
        alias_map.update(config.ion_aliases)

    if ions is None:
        ions = [k for k in alias_map.keys()]
    else:
        ions = list(ions)

    for required in ("N_HI", "N_H_total"):
        if required not in ions:
            ions.append(required)

    col_names: list[str] = ["z", "Z", "n_H", "T_init", "time_elapsed", "temperature"]
    col_names.extend(ions)
    table = QTable(names=col_names)

    n_grid = 0
    n_steps_total = 0
    for params in definition.iter_points(mode="td"):
        prefix = build_file_prefix(params, definition.stop_label, mode="td")
        col_path = output_dir / f"{prefix}.col"
        tim_path = output_dir / f"{prefix}.tim"
        if not col_path.exists():
            raise FileNotFoundError(f"Missing TD .col file: {col_path}")
        if not tim_path.exists():
            raise FileNotFoundError(f"Missing TD .tim file: {tim_path}")

        header, data_rows = _read_td_col_file(col_path)
        tim_rows = _read_td_tim_file(tim_path)

        n_steps = min(len(data_rows), len(tim_rows))
        if n_steps == 0:
            logger.warning("No data rows in %s / %s – skipping", col_path, tim_path)
            continue

        header_map = {name: idx for idx, name in enumerate(header)}

        z_val = params["z"]
        Z_val = params["Z"]
        n_H_val = params["n_H"]
        t_init_val = params.get("T_init", float("nan"))

        # Resolve total-H alias once
        total_h_alias = alias_map.get("N_H_total", STOP_ALIAS_TOTAL)
        total_h_idx = header_map.get(total_h_alias)
        # Fallback: try "H+" + neutral
        neutral_h_alias = alias_map.get("N_HI", STOP_ALIAS_NEUTRAL)
        neutral_h_idx = header_map.get(neutral_h_alias)
        h_plus_idx = header_map.get("H+")

        for step_i in range(n_steps):
            drow = data_rows[step_i]
            trow = tim_rows[step_i]

            time_elapsed = trow["time_elapsed"]
            temperature = trow["T_mean"]

            # Determine N_H_total for this time-step
            if total_h_idx is not None and total_h_idx < len(drow):
                n_h_total = drow[total_h_idx]
            else:
                n_neutral = drow[neutral_h_idx] if (neutral_h_idx is not None and neutral_h_idx < len(drow)) else 0.0
                n_ionised = drow[h_plus_idx] if (h_plus_idx is not None and h_plus_idx < len(drow)) else 0.0
                n_h_total = n_neutral + n_ionised

            row = [z_val, Z_val, n_H_val, t_init_val, time_elapsed, temperature]

            for ion in ions:
                alias = alias_map.get(ion)
                if ion == "N_H_total":
                    value = n_h_total
                elif alias is not None and alias in header_map:
                    col_idx = header_map[alias]
                    value = drow[col_idx] if col_idx < len(drow) else float("nan")
                else:
                    value = float("nan")

                if config.log_columns:
                    value = _safe_log10(value)
                row.append(value)

            table.add_row(row)
            n_steps_total += 1
        n_grid += 1

    logger.info(
        "Assembled TD grid: %d grid points, %d total time-step rows",
        n_grid, n_steps_total,
    )
    return table


def write_td_grid(table: QTable, definition: CloudyGridDefinition, config: GridAssemblyConfig) -> Path:
    """Write the assembled TD table to disk and return the output path."""

    output_path = config.output_path
    if output_path.suffix not in {".dat", ".txt"}:
        z_range = _range_label(definition.redshifts)
        Z_range = _range_label(definition.metallicities)
        nH_range = _range_label(definition.hydrogen_densities)
        t_init_range = _range_label(definition.t_init_values) if definition.t_init_values else "none"
        output_path = output_path / (
            f"full_td__z={z_range}__Z={Z_range}__nh={nH_range}__Tinit={t_init_range}.dat"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    ascii.write(table, output_path, overwrite=True)
    logger.info("Wrote TD grid table with %s rows to %s", len(table), output_path)
    return output_path


def _range_label(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=float)
    return f"{_format_parameter(arr.min())}_{_format_parameter(arr.max())}"


def write_full_grid(table: QTable, definition: CloudyGridDefinition, config: GridAssemblyConfig) -> Path:
    """Write the assembled table to disk and return the output path."""

    output_path = config.output_path
    if output_path.suffix not in {".dat", ".txt"}:
        z_range = _range_label(definition.redshifts)
        Z_range = _range_label(definition.metallicities)
        nH_range = _range_label(definition.hydrogen_densities)
        stop_range = _range_label(definition.stopping_columns)
        suffix = ""
        if definition.temperatures is not None and config.include_temperature:
            suffix += f"__T={_range_label(definition.temperatures)}"
        output_path = output_path / (
            f"full__z={z_range}__Z={Z_range}__nh={nH_range}__{definition.stop_label}={stop_range}{suffix}.dat"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    ascii.write(table, output_path, overwrite=True)
    logger.info("Wrote Cloudy grid table with %s rows to %s", len(table), output_path)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble Cloudy column-density outputs into a grid table")
    parser.add_argument("config", help="Path to JSON or YAML configuration file used for the grid run")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
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
    _, output_dir, dat_path = resolve_output_paths(config_data)
    output_dir = Path(output_dir)
    dat_path = Path(dat_path)

    if not output_dir.exists():
        raise FileNotFoundError(f"Cloudy output directory does not exist: {output_dir}")

    assembler_cfg: Mapping[str, Any] = config_data.get("assembler", {}) or {}
    mode = str(config_data.get("mode", "pie"))

    ions = assembler_cfg.get("ions")
    if ions is not None:
        ions = [str(ion) for ion in ions]

    log_columns = assembler_cfg.get("log_columns", False)
    ion_aliases = assembler_cfg.get("ion_aliases")
    overwrite = assembler_cfg.get("overwrite", True)

    if mode == "td":
        assembly_config = GridAssemblyConfig(
            output_path=dat_path,
            ion_aliases=ion_aliases,
            log_columns=log_columns,
            overwrite=overwrite,
            mode="td",
        )
        logger.info("Assembling TD Cloudy outputs from %s", output_dir)
        table = assemble_cloudy_td_table(definition, output_dir, ions=ions, config=assembly_config)
        final_path = write_td_grid(table, definition, assembly_config)
    else:
        include_temperature = assembler_cfg.get(
            "include_temperature",
            definition.temperatures is not None,
        )
        stop_alias_overrides = assembler_cfg.get("stop_alias_overrides")
        include_heating_cooling = assembler_cfg.get("include_heating_cooling", False)

        assembly_config = GridAssemblyConfig(
            output_path=dat_path,
            ion_aliases=ion_aliases,
            include_temperature=include_temperature,
            log_columns=log_columns,
            stop_alias_overrides=stop_alias_overrides,
            overwrite=overwrite,
            include_heating_cooling=include_heating_cooling,
        )
        logger.info("Assembling Cloudy outputs from %s", output_dir)
        table = assemble_cloudy_table(definition, output_dir, ions=ions, config=assembly_config)
        final_path = write_full_grid(table, definition, assembly_config)

    logger.info("Wrote assembled grid to %s", final_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())


__all__ = [
    "DEFAULT_ION_ALIASES",
    "GridAssemblyConfig",
    "assemble_cloudy_table",
    "assemble_cloudy_td_table",
    "write_full_grid",
    "write_td_grid",
    "custom_arange",
    "main",
]
