"""Utilities for mapping Voigt-profile MCMC posteriors onto Cloudy grids.

This module provides:

* helpers to load Voigt-profile MCMC chains and extract per-component column
  density samples.
* A CloudyGridInterpolator that wraps a pre-computed Cloudy grid stored as an
  Astropy table and exposes a RegularGridInterpolator in log-space.
* Likelihood helpers that combine Cloudy predictions with column-density
  measurements or limits.
* An emcee-based MCMC driver that infers physical parameters (Z, n_H, N_H, T,
  and optional relative abundances) for a single absorber component.

Example
-------
>>> from cloudy_voigt_inference import (
...     load_voigt_component,
...     CloudyGridInterpolator,
...     build_observations_from_chain,
...     CloudyComponentFitter,
... )
>>> component = load_voigt_component(
...     chain_path, param_names_path, component_id=2, ions=["N_HI", "N_CIII"])
>>> obs = build_observations_from_chain(component, percentile_bounds=(16, 84))
>>> grid = CloudyGridInterpolator.from_table_path("/path/to/cloudy_grid.ecsv")
>>> config = FitterConfig(redshift=0.1234, temperature_mode="fixed_voigt")
>>> fitter = CloudyComponentFitter(grid, obs, config=config)
>>> sampler = fitter.run_mcmc(nwalkers=200, nsteps=2000)
"""
from __future__ import annotations

import dataclasses
import logging
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# Global variable to hold the fitter instance for multiprocessing
# This avoids pickling the entire instance (with large grids/KDEs) to workers
_GLOBAL_FITTER_INSTANCE = None

def _global_log_prob(theta):
    """Wrapper for multiprocessing that uses the global fitter instance."""
    # This function is pickled by name, and workers access _GLOBAL_FITTER_INSTANCE
    # which is inherited via fork (COW) or set via initializer (if spawn)
    return _GLOBAL_FITTER_INSTANCE._log_probability(theta)

import numpy as np
from astropy.table import QTable, Table
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from scipy.special import log_ndtr, ndtr

try:
    import emcee  # type: ignore
except ImportError:  # pragma: no cover - emcee is an optional runtime dep
    emcee = None  # type: ignore

try:
    from KDEpy import TreeKDE
    HAS_KDEPY = True
except ImportError:
    HAS_KDEPY = False

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from matplotlib.axes import Axes

# -----------------------------------------------------------------------------
# Constants and small helpers
# -----------------------------------------------------------------------------

CLOUDY_AXIS_COLUMNS: Tuple[str, ...] = (
    "z",
    "Z",
    "n_H",
    "N_H",
    "NHI",
    "T",
    "T_init",
    "C_O",
    "N_O",
    "Fe_O",
    "Si_O",
    "S_O",
    "Al_O",
    "Ne_O",
)

ELEMENT_TO_IONS: Mapping[str, Tuple[str, ...]] = {
    "C": ("N_CII", "N_CIII", "N_CIV"),
    "N": ("N_NI", "N_NII", "N_NIII", "N_NIV", "N_NV"),
    "O": ("N_OI", "N_OII", "N_OIII", "N_OIV", "N_OV", "N_OVI"),
    "Ne": ("N_NeVIII",),
    "Fe": ("N_FeII",),
    "Si": ("N_SiII", "N_SiIII", "N_SiIV"),
    "S": ("N_SII", "N_SIII", "N_SIV", "N_SV"),
    "Al": ("N_AlII",),
}

KMS_CM_RATIO = (1e12 / (1 * (1e12))).real  # placeholder to match previous code style
MULTIPLE_CONVERSION = 1.0  # dimensionless legacy placeholder
DEFAULT_PERCENTILES = (16.0, 50.0, 84.0)
LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
DEFAULT_REL_ABUNDANCE_BOUNDS = (-6.0, 6.0)
DEFAULT_SOFT_LIMIT_SIGMA = 0.01
LOG10_CM_PER_KPC = np.log10(3.085677581491367e21)
LOG10_PROTON_MASS_G = np.log10(1.6726219e-24)
LOG10_SOLAR_MASS_G = np.log10(1.98847e33)

# Solar photospheric abundances log10(X/H) by number (Asplund et al. 2009)
SOLAR_LOG_ABUNDANCE: Mapping[str, float] = {
    "H": 0.0, "C": -3.57, "N": -4.17, "O": -3.31,
    "Ne": -4.07,
    "Fe": -4.50, "Si": -4.49, "S": -4.88, "Al": -5.55,
}

# Atomic masses in grams and their log10 values
ATOMIC_MASS_G: Mapping[str, float] = {
    "H": 1.674e-24, "C": 1.994e-23, "N": 2.326e-23, "O": 2.657e-23,
    "Ne": 3.351e-23,
    "Fe": 9.274e-23, "Si": 4.664e-23, "S": 5.312e-23, "Al": 4.480e-23,
}
LOG10_ATOMIC_MASS_G: Mapping[str, float] = {k: np.log10(v) for k, v in ATOMIC_MASS_G.items()}

# Reverse lookup: ion name -> element symbol
ION_TO_ELEMENT: Mapping[str, str] = {}
for _el, _ions in ELEMENT_TO_IONS.items():
    for _ion in _ions:
        ION_TO_ELEMENT[_ion] = _el

# Jeans instability constants (CGS)
# lambda_J = c_s * sqrt(pi / (G * rho))
#          = sqrt(k_B T / (mu m_p)) * sqrt(pi / (G * m_p * n_H))
#          = sqrt(pi * k_B * T / (G * mu * m_p^2 * n_H))
# A_const  = log10( pi * k_B / (G * mu * m_p^2) )
_JEANS_MU = 0.6
_JEANS_A_CONST = (
    np.log10(np.pi)
    + np.log10(1.380649e-16)       # k_B  [erg K^-1]
    - np.log10(6.67430e-8)         # G    [cm^3 g^-1 s^-2]
    - 2.0 * np.log10(_JEANS_MU)          # mu
    - 2.0 * np.log10(1.6726219e-24)  # m_p^2  [g^2]
)

CORNER_LABEL_OVERRIDES = {
    "Z": "log Z",
    "n_H": "log n_H",
    "N_H": "log N_H (total)",
    "NHI": "log N_HI (neutral)",
    "T": "log T",
    "T_init": "log T_init (K)",
}


def _normalise_element_symbol(symbol: str) -> str:
    cleaned = symbol.strip()
    if not cleaned:
        raise ValueError("Element symbol cannot be empty for relative abundance configuration")
    canonical = cleaned[0].upper() + cleaned[1:].lower()
    if canonical not in ELEMENT_TO_IONS:
        allowed = ", ".join(sorted(ELEMENT_TO_IONS.keys()))
        raise ValueError(f"Unsupported element '{symbol}' for relative abundance configuration. Choose from: {allowed}")
    return canonical


def _build_ratio_key(numerator: str, denominator: str) -> str:
    return f"{numerator}_{denominator}"

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class VoigtComponent:
    """Container for chain samples of a single Voigt component."""

    ions: Dict[str, np.ndarray]
    temperature: Optional[np.ndarray] = None
    name: Optional[str] = None

    def available_ions(self) -> List[str]:
        return list(self.ions.keys())


@dataclasses.dataclass
class Observation:
    value: float
    sigma: Optional[float]
    is_upper_limit: bool = False
    is_lower_limit: bool = False


@dataclasses.dataclass
class ZScoreGridConfig:
    """Configuration for building monotonic z-score interpolation grids."""

    z_min: float = -3.5
    z_max: float = 3.5
    dz: float = 0.1
    min_slope: float = 1e-6
    min_spacing: float = 1e-6

    def make_grid(self) -> np.ndarray:
        if self.dz <= 0:
            raise ValueError("dz must be positive for ZScoreGridConfig")
        if self.z_max <= self.z_min:
            raise ValueError("z_max must be greater than z_min for ZScoreGridConfig")
        n_steps = int(np.floor((self.z_max - self.z_min) / self.dz + 0.5))
        grid = self.z_min + self.dz * np.arange(n_steps + 1)
        if grid[-1] < self.z_max - 0.5 * self.dz:
            grid = np.append(grid, self.z_max)
        else:
            grid[-1] = self.z_max
        return grid


@dataclasses.dataclass
class ZScoreObservation:
    """Observation represented through an empirical z-score interpolation."""

    value_grid: np.ndarray
    z_grid: np.ndarray
    dv_dz: np.ndarray
    is_upper_limit: bool = False
    upper_limit_value: Optional[float] = None
    is_lower_limit: bool = False
    lower_limit_value: Optional[float] = None

    @classmethod
    def from_samples(
        cls,
        samples: ArrayLike,
        config: ZScoreGridConfig,
        is_upper_limit: bool = False,
        upper_limit_value: Optional[float] = None,
        is_lower_limit: bool = False,
        lower_limit_value: Optional[float] = None,
    ) -> "ZScoreObservation":
        data = np.asarray(samples, dtype=float)
        data = data[np.isfinite(data)]
        if data.size < 8:
            raise ValueError("Need at least 8 finite samples to build z-score observation")

        z_grid = config.make_grid()
        cdf = ndtr(z_grid)
        # Clip extreme quantiles to stay inside (0,1)
        eps = 0.5 / data.size
        cdf = np.clip(cdf, eps, 1.0 - eps)
        value_grid = np.quantile(data, cdf)

        # Enforce strict monotonicity to avoid zero-width interpolation segments
        if config.min_spacing > 0:
            for i in range(1, value_grid.size):
                if value_grid[i] <= value_grid[i - 1]:
                    value_grid[i] = value_grid[i - 1] + config.min_spacing

        dv_dz = np.gradient(value_grid, z_grid)
        dv_dz = np.where(dv_dz <= 0, config.min_slope, dv_dz)
        dv_dz = np.clip(dv_dz, config.min_slope, None)

        if is_upper_limit and upper_limit_value is None:
            upper_limit_value = float(value_grid[-1])

        if is_lower_limit and lower_limit_value is None:
            lower_limit_value = float(value_grid[0])

        return cls(
            value_grid=np.asarray(value_grid, dtype=float),
            z_grid=np.asarray(z_grid, dtype=float),
            dv_dz=np.asarray(dv_dz, dtype=float),
            is_upper_limit=is_upper_limit,
            upper_limit_value=upper_limit_value,
            is_lower_limit=is_lower_limit,
            lower_limit_value=lower_limit_value,
        )

    def value_to_z(self, value: float) -> float:
        """Map a column-density value onto the z-grid with mild extrapolation."""

        if value <= self.value_grid[0]:
            slope = self.dv_dz[0]
            if slope <= 0:
                return float(self.z_grid[0])
            return float(self.z_grid[0] + (value - self.value_grid[0]) / slope)
        if value >= self.value_grid[-1]:
            slope = self.dv_dz[-1]
            if slope <= 0:
                return float(self.z_grid[-1])
            return float(self.z_grid[-1] + (value - self.value_grid[-1]) / slope)
        return float(np.interp(value, self.value_grid, self.z_grid))

    def _dv_dz(self, value: float) -> float:
        if value <= self.value_grid[0]:
            return float(self.dv_dz[0])
        if value >= self.value_grid[-1]:
            return float(self.dv_dz[-1])
        return float(np.interp(value, self.value_grid, self.dv_dz))

    def log_pdf(self, value: float) -> float:
        slope = self._dv_dz(value)
        if slope <= 0 or not np.isfinite(slope):
            return float(-np.inf)
        z = self.value_to_z(value)
        return float(-0.5 * z * z - LOG_SQRT_2PI - np.log(slope))

    def log_cdf(self, value: float) -> float:
        z = self.value_to_z(value)
        return float(log_ndtr(z))


ObservationType = Union[Observation, ZScoreObservation]


# -----------------------------------------------------------------------------
# Voigt chain helpers
# -----------------------------------------------------------------------------


def _to_log10(columns: np.ndarray) -> np.ndarray:
    columns = np.asarray(columns, dtype=float)
    with np.errstate(divide="ignore"):
        return np.log10(columns)


def _find_param_index(param_names: Sequence[str], target: str) -> Optional[int]:
    try:
        return list(param_names).index(target)
    except ValueError:
        return None


def load_voigt_component(
    chain_path: ArrayLike,
    param_names_path: ArrayLike,
    component_id: int,
    ions: Sequence[str],
    base_unit_scale: float = MULTIPLE_CONVERSION,
    convert_columns_to_log: bool = True,
    convert_temperature_to_log: bool = True,
) -> VoigtComponent:
    """Load Voigt MCMC samples and extract the subset for a single component.

    Parameters
    ----------
    chain_path
        Path to the ``*_chain.npy`` file produced by the Voigt-profile fit.
    param_names_path
        Path to the companion ``*_param_names.npy`` file.
    component_id
        Integer component index (matches the suffix used in the chain names).
    ions
        Iterable of ion column names to extract (e.g. "N_CIII").
    base_unit_scale
        Factor applied before taking the logarithm (legacy conversion).
    convert_columns_to_log
        If ``True`` convert column density samples to ``log10`` space.
    convert_temperature_to_log
        If ``True`` convert temperatures to ``log10``.
    """

    chain = np.load(chain_path)
    param_names = np.load(param_names_path)
    if chain.ndim != 2:
        raise ValueError("Expected chain to be 2-D (nsamples, nparams)")

    samples = {}
    for ion in ions:
        label = f"{ion}_{component_id}"
        idx = _find_param_index(param_names, label)
        if idx is None:
            raise KeyError(f"Ion parameter '{label}' not found in chain")
        data = chain[:, idx]
        if convert_columns_to_log:
            data = _to_log10(data * np.round(base_unit_scale))
        samples[ion] = data

    temp_idx = _find_param_index(param_names, f"T_{component_id}")
    temperature = None
    if temp_idx is not None:
        temperature = chain[:, temp_idx]
        if convert_temperature_to_log:
            temperature = _to_log10(temperature)

    return VoigtComponent(ions=samples, temperature=temperature, name=str(component_id))


def summarise_samples(
    samples: np.ndarray,
    percentiles: Tuple[float, float, float] = DEFAULT_PERCENTILES,
) -> Tuple[float, float]:
    lo, med, hi = np.percentile(samples, percentiles)
    sigma = 0.5 * (hi - lo)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = None
    return med, sigma


def build_observations_from_chain(
    component: VoigtComponent,
    percentile_bounds: Tuple[float, float] = (16.0, 84.0),
    upper_limits: Optional[Mapping[str, float]] = None,
    lower_limits: Optional[Mapping[str, float]] = None,
    override_sigmas: Optional[Mapping[str, float]] = None,
    temperature_bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, Observation]:
    """Convert chain samples into Gaussian, upper-limit, or lower-limit observations.

    Parameters
    ----------
    upper_limits : mapping, optional
        ``{ion: logN_value}`` — ions treated as upper limits.  The value is
        used directly when not ``None``; otherwise the chain median is used.
    lower_limits : mapping, optional
        ``{ion: logN_value}`` — ions treated as lower limits.  Same convention.
    """

    observations: Dict[str, Observation] = {}
    lower, upper = percentile_bounds
    percs = (lower, 50.0, upper)

    for ion, samples in component.ions.items():
        med, sigma = summarise_samples(samples, percs)
        upper_limit_value = None if upper_limits is None else upper_limits.get(ion)
        lower_limit_value = None if lower_limits is None else lower_limits.get(ion)
        if ion in (upper_limits or {}):
            value = upper_limit_value if upper_limit_value is not None else med
            observations[ion] = Observation(value=value, sigma=sigma, is_upper_limit=True)
        elif ion in (lower_limits or {}):
            value = lower_limit_value if lower_limit_value is not None else med
            observations[ion] = Observation(value=value, sigma=sigma, is_lower_limit=True)
        else:
            if override_sigmas and ion in override_sigmas:
                sigma = override_sigmas[ion]
            observations[ion] = Observation(value=med, sigma=sigma, is_upper_limit=False)

    if component.temperature is not None:
        med, sigma = summarise_samples(component.temperature, percs)
        if temperature_bounds is not None:
            lo, hi = temperature_bounds
            med = np.clip(med, lo, hi)
        observations["T"] = Observation(value=med, sigma=sigma, is_upper_limit=False)

    return observations


def build_zscore_observations_from_chain(
    component: VoigtComponent,
    config: Optional[ZScoreGridConfig] = None,
    upper_limits: Optional[Mapping[str, float]] = None,
    lower_limits: Optional[Mapping[str, float]] = None,
    temperature_bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, ObservationType]:
    """Build observations using empirical z-score interpolation grids.

    Parameters
    ----------
    upper_limits : mapping, optional
        ``{ion: logN_value}`` — ions treated as upper limits.
    lower_limits : mapping, optional
        ``{ion: logN_value}`` — ions treated as lower limits.
    """

    if config is None:
        config = ZScoreGridConfig()

    observations: Dict[str, ObservationType] = {}
    for ion, samples in component.ions.items():
        is_upper = upper_limits is not None and ion in upper_limits
        ul_value = None if upper_limits is None else upper_limits.get(ion)
        is_lower = lower_limits is not None and ion in lower_limits
        ll_value = None if lower_limits is None else lower_limits.get(ion)
        observations[ion] = ZScoreObservation.from_samples(
            samples,
            config,
            is_upper_limit=is_upper,
            upper_limit_value=ul_value,
            is_lower_limit=is_lower,
            lower_limit_value=ll_value,
        )

    if component.temperature is not None:
        med, sigma = summarise_samples(component.temperature)
        if temperature_bounds is not None:
            lo, hi = temperature_bounds
            med = np.clip(med, lo, hi)
        observations["T"] = Observation(value=med, sigma=sigma, is_upper_limit=False)

    return observations


# -----------------------------------------------------------------------------
# Cloudy grid interpolation
# -----------------------------------------------------------------------------


def _prepare_sort_key(table: Table, keys: Sequence[str]) -> np.ndarray:
    arrays = [np.asarray(table[key]) for key in keys]
    return np.lexsort(arrays[::-1])


class CloudyGridInterpolator:
    """Wrap a gridded Cloudy output table with an n-D interpolator."""

    def __init__(
        self,
        points: Tuple[np.ndarray, ...],
        values: np.ndarray,
        ion_order: Tuple[str, ...],
        axis_names: Tuple[str, ...] = CLOUDY_AXIS_COLUMNS,
        is_td_grid: bool = False,
    ) -> None:
        self.points = tuple(np.asarray(p, dtype=float) for p in points)
        self.values = np.asarray(values, dtype=float)
        self.ion_order = ion_order
        self.axis_names = axis_names
        self.is_td_grid = is_td_grid
        self.interpolator = RegularGridInterpolator(
            self.points,
            self.values,
            bounds_error=False,
            fill_value=None,
        )

    @classmethod
    def from_table(cls, table: Table, ion_columns: Optional[Sequence[str]] = None) -> "CloudyGridInterpolator":
        if "N_H" not in table.colnames:
            if "NH" in table.colnames:
                table = table.copy()
                table["N_H"] = table["NH"]
                logger.info("Copied 'NH' column to 'N_H' for grid axis consistency")
            # Note: We do NOT copy NHI to N_H anymore, to support NHI as a distinct axis.

        axis_names = tuple(col for col in CLOUDY_AXIS_COLUMNS if col in table.colnames)
        if ion_columns is None:
            inferred_columns = [col for col in table.colnames if col.startswith("N_")]
            for extra in ("T_cloudy", "log_heat_rate", "log_cool_rate", "log_tcool"):
                if extra in table.colnames:
                    inferred_columns.append(extra)
            ion_columns = inferred_columns
        sort_idx = _prepare_sort_key(table, axis_names)
        sorted_table = table[sort_idx]

        axes = [np.unique(sorted_table[col]) for col in axis_names]
        shape = tuple(len(ax) for ax in axes)

        ion_order = tuple(ion_columns)
        cube = np.empty(shape + (len(ion_order),))

        grid_size = int(np.prod(shape))
        if len(sorted_table) != grid_size:
            axis_summary = ", ".join(f"{name}={len(values)}" for name, values in zip(axis_names, axes))
            raise ValueError(
                "Cloudy grid table is not a complete rectangular grid: "
                f"expected {grid_size} rows from axes ({axis_summary}), but found {len(sorted_table)}. "
                "This usually means some parameter combinations are missing or duplicated in the assembled grid."
            )

        for i, ion in enumerate(ion_order):
            data = np.asarray(sorted_table[ion])
            with np.errstate(divide="ignore"):
                cube[..., i] = np.log10(data).reshape(shape)

        # Filter out degenerate axes (length 1) to avoid RegularGridInterpolator errors
        valid_dims = [i for i, ax in enumerate(axes) if len(ax) > 1]
        
        # New points and axis names
        new_points = tuple(axes[i] for i in valid_dims)
        new_axis_names = tuple(axis_names[i] for i in valid_dims)
        
        # New values shape: only valid dims + ion dim
        new_shape = tuple(len(ax) for ax in new_points) + (len(ion_order),)
        values = cube.reshape(new_shape)
        
        # Log dropped axes
        dropped_axes = [axis_names[i] for i, ax in enumerate(axes) if len(ax) == 1]
        if dropped_axes:
            print(f"CloudyGridInterpolator: Dropped degenerate axes {dropped_axes} (len=1)")

        points = new_points
        return cls(points, values, ion_order, axis_names=new_axis_names)

    @classmethod
    def from_table_path(
        cls, path: ArrayLike, ion_columns: Optional[Sequence[str]] = None
    ) -> "CloudyGridInterpolator":
        table = Table.read(path)
        return cls.from_table(table, ion_columns=ion_columns)

    @classmethod
    def from_td_table(
        cls,
        table: Table,
        ion_columns: Optional[Sequence[str]] = None,
        temperature_step: float = 0.05,
    ) -> "CloudyGridInterpolator":
        """Build an interpolator from a raw time-dependent assembled grid.

        The input *table* has irregular time-step rows per grid point with
        columns ``z, Z, n_H, T_init, time_elapsed, temperature, [ions], N_H_total``.
        Ion values are raw single-zone column densities (linear).

        This method:
        1. Groups rows by outer grid axes ``(z, Z, n_H, T_init)``
        2. Converts ``temperature`` (Kelvin) to ``log10(T)``
        3. Computes ion fractions: ``log10(N_ion / N_H_total)``
        4. Resamples each group onto a common regular ``T`` axis
        5. Returns a ``CloudyGridInterpolator`` with ``is_td_grid=True``

        Parameters
        ----------
        temperature_step : float
            Spacing of the regular log10(T) axis in dex (default 0.05).
        """
        table = table.copy()

        _TD_OUTER_AXES = ("z", "Z", "n_H", "T_init")
        outer_axes = tuple(col for col in _TD_OUTER_AXES if col in table.colnames)

        if "temperature" not in table.colnames:
            raise ValueError("TD table must contain a 'temperature' column (linear Kelvin)")
        if "N_H_total" not in table.colnames:
            raise ValueError("TD table must contain an 'N_H_total' column")

        raw_temp = np.asarray(table["temperature"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_temp = np.log10(raw_temp)

        if ion_columns is None:
            ion_columns = [
                col for col in table.colnames
                if col.startswith("N_") and col != "N_H_total"
            ]
        ion_order = tuple(ion_columns)

        # --- Build regular T axis ---
        finite_mask = np.isfinite(log_temp) & (raw_temp > 0)
        if not np.any(finite_mask):
            raise ValueError("No finite temperature values in TD table")
        global_min = np.floor(np.min(log_temp[finite_mask]) / temperature_step) * temperature_step
        global_max = np.ceil(np.max(log_temp[finite_mask]) / temperature_step) * temperature_step
        t_axis = np.round(np.arange(global_min, global_max + 0.5 * temperature_step, temperature_step), decimals=6)
        n_t = len(t_axis)

        # --- Identify unique outer-axis combos ---
        outer_unique_vals = [np.unique(np.asarray(table[col], dtype=float)) for col in outer_axes]
        outer_shape = tuple(len(u) for u in outer_unique_vals)
        full_shape = outer_shape + (n_t, len(ion_order))
        cube = np.full(full_shape, np.nan, dtype=float)

        # Build a mapping from outer-axis value tuple to multi-index
        from itertools import product as _product
        outer_val_to_idx = {}
        for multi_idx in _product(*(range(len(u)) for u in outer_unique_vals)):
            key = tuple(float(outer_unique_vals[d][i]) for d, i in enumerate(multi_idx))
            outer_val_to_idx[key] = multi_idx

        n_h_total_col = np.asarray(table["N_H_total"], dtype=float)

        # Pre-extract ion data columns
        ion_data = {ion: np.asarray(table[ion], dtype=float) for ion in ion_order}

        # Group rows by outer axes and resample
        outer_col_data = [np.asarray(table[col], dtype=float) for col in outer_axes]

        # Build group keys for every row
        row_keys = np.column_stack(outer_col_data) if outer_col_data else np.zeros((len(table), 0))

        # Use structured grouping
        unique_keys_set: Dict[tuple, List[int]] = {}
        for row_i in range(len(table)):
            key = tuple(float(row_keys[row_i, d]) for d in range(len(outer_axes)))
            unique_keys_set.setdefault(key, []).append(row_i)

        for key, row_indices in unique_keys_set.items():
            multi_idx = outer_val_to_idx.get(key)
            if multi_idx is None:
                continue

            idx_arr = np.array(row_indices)
            group_log_t = log_temp[idx_arr]
            group_n_h_total = n_h_total_col[idx_arr]

            # Sort by ascending log_T for np.interp
            sort_order = np.argsort(group_log_t)
            group_log_t = group_log_t[sort_order]
            group_n_h_total = group_n_h_total[sort_order]
            idx_arr_sorted = idx_arr[sort_order]

            # Remove duplicates / non-finite
            valid = np.isfinite(group_log_t) & (group_n_h_total > 0)
            if not np.any(valid):
                continue
            group_log_t = group_log_t[valid]
            group_n_h_total = group_n_h_total[valid]
            idx_arr_sorted = idx_arr_sorted[valid]

            # Remove duplicate temperatures (keep last = most evolved)
            _, unique_idx = np.unique(group_log_t, return_index=True)
            group_log_t = group_log_t[unique_idx]
            group_n_h_total = group_n_h_total[unique_idx]
            idx_arr_sorted = idx_arr_sorted[unique_idx]

            for ion_i, ion in enumerate(ion_order):
                raw_ion = ion_data[ion][idx_arr_sorted]
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_frac = np.log10(raw_ion) - np.log10(group_n_h_total)

                # Mask non-finite fractions
                frac_valid = np.isfinite(log_frac)
                if not np.any(frac_valid):
                    continue

                xp = group_log_t[frac_valid]
                fp = log_frac[frac_valid]

                if len(xp) < 2:
                    # Single point — fill only the nearest T bin
                    nearest = np.argmin(np.abs(t_axis - xp[0]))
                    cube[multi_idx + (nearest, ion_i)] = fp[0]
                else:
                    resampled = np.interp(t_axis, xp, fp, left=np.nan, right=np.nan)
                    cube[multi_idx + (slice(None), ion_i)] = resampled

        # Combine outer axes + T into the full axis list
        all_axes = list(outer_unique_vals) + [t_axis]
        all_axis_names = list(outer_axes) + ["T"]

        # Drop degenerate axes (length 1)
        valid_dims = [i for i, ax in enumerate(all_axes) if len(ax) > 1]
        new_points = tuple(all_axes[i] for i in valid_dims)
        new_axis_names = tuple(all_axis_names[i] for i in valid_dims)

        # Reshape values: collapse degenerate dims, keep ion dim
        keep_shape = tuple(len(all_axes[i]) for i in valid_dims) + (len(ion_order),)
        values = cube.reshape(keep_shape)

        dropped = [all_axis_names[i] for i, ax in enumerate(all_axes) if len(ax) == 1]
        if dropped:
            logger.info("CloudyGridInterpolator.from_td_table: Dropped degenerate axes %s", dropped)
        logger.info(
            "Built TD grid interpolator: axes=%s, shape=%s, ions=%d",
            new_axis_names, tuple(len(p) for p in new_points), len(ion_order),
        )
        return cls(new_points, values, ion_order, axis_names=new_axis_names, is_td_grid=True)

    @classmethod
    def from_td_table_path(
        cls,
        path: ArrayLike,
        ion_columns: Optional[Sequence[str]] = None,
        temperature_step: float = 0.05,
    ) -> "CloudyGridInterpolator":
        table = Table.read(path)
        return cls.from_td_table(table, ion_columns=ion_columns, temperature_step=temperature_step)

    def evaluate(self, params: Mapping[str, float]) -> Dict[str, float]:
        ordered = []
        for axis in self.axis_names:
            if axis not in params:
                raise KeyError(f"Missing grid parameter '{axis}'")
            ordered.append(params[axis])

        log_columns = np.asarray(self.interpolator(ordered))
        if log_columns.ndim == 0:
            log_columns = np.expand_dims(log_columns, axis=0)
        elif log_columns.ndim > 1:
            log_columns = np.reshape(log_columns, (-1,))
        if log_columns.size != len(self.ion_order):
            raise ValueError(
                "Interpolator returned unexpected shape: "
                f"got {log_columns.shape}, expected {len(self.ion_order)} entries"
            )
        return {ion: float(log_columns[i]) for i, ion in enumerate(self.ion_order)}

    def parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {axis: (float(points[0]), float(points[-1])) for axis, points in zip(self.axis_names, self.points)}


# -----------------------------------------------------------------------------
# Likelihood machinery
# -----------------------------------------------------------------------------


def _loglike_gaussian(model: float, obs: Observation) -> float:
    if obs.sigma is None or obs.sigma <= 0:
        return 0.0 if np.isclose(model, obs.value) else -0.5 * (model - obs.value) ** 2
    chi = (model - obs.value) / obs.sigma
    return -0.5 * (chi**2 + np.log(2 * np.pi * obs.sigma**2))


def _loglike_upper_limit(model: float, obs: Observation) -> float:
    if obs.sigma is None or obs.sigma <= 0:
        return 0.0 if model <= obs.value else -np.inf
    z = (obs.value - model) / obs.sigma
    return float(log_ndtr(z))


def _loglike_lower_limit(model: float, obs: Observation) -> float:
    """Lower-limit log-likelihood: penalise models predicting *less* than the limit."""
    if obs.sigma is None or obs.sigma <= 0:
        return 0.0 if model >= obs.value else -np.inf
    z = (model - obs.value) / obs.sigma
    return float(log_ndtr(z))


def _loglike_zscore(model: float, obs: ZScoreObservation) -> float:
    if obs.is_lower_limit:
        limit_value = obs.lower_limit_value if obs.lower_limit_value is not None else obs.value_grid[0]
        z_limit = obs.value_to_z(limit_value)
        z_model = obs.value_to_z(model)
        return float(log_ndtr(z_model - z_limit))
    if obs.is_upper_limit:
        limit_value = obs.upper_limit_value if obs.upper_limit_value is not None else obs.value_grid[-1]
        z_limit = obs.value_to_z(limit_value)
        z_model = obs.value_to_z(model)
        return float(log_ndtr(z_limit - z_model))
    return obs.log_pdf(model)


def log_likelihood(
    model: Mapping[str, float],
    observations: Mapping[str, ObservationType],
) -> float:
    total = 0.0
    for ion, obs in observations.items():
        if ion not in model:
            logger.debug("Model missing ion %s; skipping term", ion)
            continue
        pred = model[ion]
        if isinstance(obs, ZScoreObservation):
            total += _loglike_zscore(pred, obs)
        elif obs.is_lower_limit:
            total += _loglike_lower_limit(pred, obs)
        elif obs.is_upper_limit:
            total += _loglike_upper_limit(pred, obs)
        else:
            total += _loglike_gaussian(pred, obs)
    return total


# -----------------------------------------------------------------------------
# MCMC driver
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class FitterConfig:
    redshift: float
    temperature_mode: str = "free"  # choices: free, fixed_voigt, fixed_value
    fixed_temperature_value: Optional[float] = None
    relative_abundance_bounds: Mapping[str, Tuple[float, float]] = dataclasses.field(default_factory=dict)
    relative_abundance_fixed: Mapping[str, float] = dataclasses.field(default_factory=dict)
    abundance_reference_element: str = "O"
    free_abundances: Sequence[str] = dataclasses.field(default_factory=list)
    fixed_abundances: Mapping[str, float] = dataclasses.field(default_factory=dict)

    # Custom flat-prior bounds overriding grid extents (e.g. {"Z": (-2, 0.5), "n_H": (-4, -1)})
    parameter_bounds_override: Mapping[str, Tuple[float, float]] = dataclasses.field(default_factory=dict)

    # Derived-quantity priors (log-space bounds applied inside _log_probability)
    cloud_length_bounds: Optional[Tuple[float, float]] = None      # log L (kpc), per component
    cloud_mass_bounds: Optional[Tuple[float, float]] = None        # log M (M_sun), per component
    total_length_bounds: Optional[Tuple[float, float]] = None      # log sum(L) (kpc), across components
    total_mass_bounds: Optional[Tuple[float, float]] = None        # log sum(M) (M_sun), across components
    jeans_length_prior: bool = False                                # enforce L < L_Jeans per component

    # Total element mass bounds (log M_sun).  Keyed by element symbol.
    # The prior is applied to the total mass summed across all components.
    # Example: {"C": (2.0, 10.0), "O": (3.0, 11.0)}
    element_mass_bounds: Optional[Dict[str, Tuple[float, float]]] = None

    # Total ion column-density bounds (log cm^-2), keyed by ion name.
    # In the joint fitter this is applied to the sum across all Cloudy
    # components in linear space (then converted to log10). In the single
    # fitter this is applied directly to the model column.
    # Example: {"N_OVI": (13.5, 14.8), "N_HI": (15.8, 16.6)}
    total_column_density_bounds: Optional[Dict[str, Tuple[float, float]]] = None

    # Voigt temperature prior: compare PIE T_cloudy to Voigt profile temperature
    voigt_temperature_samples: Optional[List[Optional[np.ndarray]]] = None  # per-component T chains
    use_voigt_temperature_prior: bool = False

    # Absorber grouping: maps each Cloudy component index to a list of
    # original Voigt absorber IDs.  Used by ``posterior_absorption_plot``
    # to split predicted columns back to individual absorbers.
    # Example: {0: [0, 1], 1: [2]} means Cloudy comp 0 groups Voigt
    # absorbers 0 & 1.  ``None`` means no grouping.
    component_groups: Optional[Dict[int, List[int]]] = None

    # The original Voigt chain and param_names are needed when
    # component_groups is set so that ``split_group_columns`` can
    # compute proportional fractions.  Set these in the notebook.
    voigt_chain_for_splitting: Optional[np.ndarray] = None
    voigt_chain_param_names: Optional[List[str]] = None

    # Per-component abundance configuration for JointCloudyComponentFitter.
    # When set, overrides global abundance_reference_element / free_abundances /
    # fixed_abundances on a per-component basis.  Each entry is a dict:
    #   {"abundance_reference_element": "C", "free_abundances": ["O_C"], "fixed_abundances": {"Si_O": 0.0}}
    # Length must match the number of components.  ``None`` falls back to the
    # global settings for all components (backwards-compatible).
    per_component_abundance_config: Optional[List[Dict[str, Any]]] = None

    # Per-component temperature mode overrides for JointCloudyComponentFitter.
    # When set, each element specifies the temperature_mode for that component.
    # Valid values: "free", "fixed_voigt", "fixed_value", "auto".
    # "auto" (default when None) means: use "free" if the grid has a T axis,
    # otherwise fall back to the global temperature_mode.
    # Length must match the number of components.  ``None`` → auto-detect for all.
    per_component_temperature_mode: Optional[List[str]] = None

    # Joint fitter likelihood toggle.
    # True  -> include KDE/GMM likelihood term (default, current behavior).
    # False -> run with priors + observation limits only.
    use_kde_likelihood: bool = True

    # Cross-component parameter constraints for JointCloudyComponentFitter.
    # Each entry is a string expression.  Inequality examples:
    #   "n_H_0 > n_H_1"  – reject samples where comp-0 density <= comp-1
    #   "Z_0 < Z_1"      – reject samples where comp-0 metallicity >= comp-1
    # Equality examples:
    #   "Z_0 == Z_1"      – tie Z across two components (reduces MCMC dim)
    #   "n_H_0 == n_H_1"  – tie density across two components
    # Supported parameter prefixes include Z, n_H, N_H / NHI, and any
    # relative-abundance key.  The ``_0``, ``_1``, etc. suffixes refer to
    # the component index as it appears in ``component_ids``.
    cross_component_constraints: Optional[List[str]] = None

    # TD grid inference: log10 N_H bounds when N_H is a free MCMC parameter.
    # Used when the grid is a time-dependent grid (is_td_grid=True) that does
    # not have N_H as an axis.  When None, defaults to (17.0, 23.0).
    td_n_h_bounds: Optional[Tuple[float, float]] = None

    def normalised_relative_abundances(self) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
        bounds = {key: tuple(value) for key, value in self.relative_abundance_bounds.items()}
        fixed = {key: float(val) for key, val in self.relative_abundance_fixed.items()}

        for key in self.free_abundances:
            bounds.setdefault(key, DEFAULT_REL_ABUNDANCE_BOUNDS)
        for key, value in self.fixed_abundances.items():
            fixed.setdefault(key, float(value))

        return bounds, fixed

    def normalised_relative_abundances_for(
        self, comp_cfg: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Tuple[float, float]], Dict[str, float]]:
        """Build normalised abundance reference, bounds, and fixed values from
        a per-component config dict (used by ``per_component_abundance_config``).

        Falls back to the global ``self`` fields for any key not present in
        *comp_cfg*.
        """
        ref = comp_cfg.get("abundance_reference_element", self.abundance_reference_element)
        free = comp_cfg.get("free_abundances", self.free_abundances)
        fixed_in = comp_cfg.get("fixed_abundances", self.fixed_abundances)
        bounds_in = comp_cfg.get("relative_abundance_bounds", self.relative_abundance_bounds)
        fixed_base = comp_cfg.get("relative_abundance_fixed", self.relative_abundance_fixed)

        bounds: Dict[str, Tuple[float, float]] = {k: tuple(v) for k, v in bounds_in.items()}
        fixed: Dict[str, float] = {k: float(v) for k, v in fixed_base.items()}

        for key in free:
            bounds.setdefault(key, DEFAULT_REL_ABUNDANCE_BOUNDS)
        for key, value in fixed_in.items():
            fixed.setdefault(key, float(value))

        return ref, bounds, fixed


class CloudyComponentFitter:
    """Run an emcee-based fit of Cloudy parameters to Voigt observations."""

    def __init__(
        self,
        grid: CloudyGridInterpolator,
        observations: Mapping[str, ObservationType],
        config: FitterConfig,
    ) -> None:
        if emcee is None:
            raise RuntimeError("emcee is required for CloudyComponentFitter")
        self.grid = grid
        self.observations = dict(observations)
        self.config = config

        self.temperature_observation = self.observations.pop("T", None)
        self.fixed_temperature: Optional[float] = None

        bounds = grid.parameter_bounds()
        self.z_bounds = bounds.get("z", (config.redshift, config.redshift))

        self.param_order: List[str] = ["Z", "n_H"]
        
        # Determine which column density parameter is available
        if "N_H" in bounds:
            self.param_order.append("N_H")
        elif "NHI" in bounds:
            self.param_order.append("NHI")
        else:
            raise KeyError("Grid missing axis for column density (expected 'N_H' or 'NHI')")

        try:
            self.param_bounds: List[Tuple[float, float]] = [
                config.parameter_bounds_override.get(p, bounds[p]) for p in self.param_order
            ]
        except KeyError as exc:
            raise KeyError(f"Grid missing axis for parameter {exc.args[0]!r}") from exc

        if config.temperature_mode == "free":
            if "T" not in bounds:
                raise KeyError("Grid does not contain a temperature axis")
            self.param_order.append("T")
            self.param_bounds.append(bounds["T"])
        elif config.temperature_mode == "fixed_voigt":
            if self.temperature_observation is None:
                raise ValueError("Temperature observation required for fixed_voigt mode")
            self.fixed_temperature = self.temperature_observation.value
        elif config.temperature_mode == "fixed_value":
            if config.fixed_temperature_value is None:
                raise ValueError("Specify fixed_temperature_value for mode='fixed_value'")
            self.fixed_temperature = config.fixed_temperature_value
        else:
            raise ValueError("temperature_mode must be one of {'free','fixed_voigt','fixed_value'}")


        reference = config.abundance_reference_element or "O"
        self.abundance_reference = _normalise_element_symbol(reference)
        self.relative_ratio_to_ions: Dict[str, Tuple[str, ...]] = {
            _build_ratio_key(element, self.abundance_reference): ions
            for element, ions in ELEMENT_TO_IONS.items()
            if element != self.abundance_reference
        }
        self.supported_relative_keys: Tuple[str, ...] = tuple(self.relative_ratio_to_ions.keys())

        raw_bounds, raw_fixed = config.normalised_relative_abundances()

        def _canonicalise_bounds(data: Mapping[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
            canonical: Dict[str, Tuple[float, float]] = {}
            for raw_key, bound in data.items():
                canonical_key = self._canonicalise_ratio_key(raw_key)
                lo, hi = bound
                canonical[canonical_key] = (float(lo), float(hi))
            return canonical

        def _canonicalise_fixed(
            data: Mapping[str, float],
        ) -> Tuple[Dict[str, float], List[Tuple[str, str, float]]]:
            canonical: Dict[str, float] = {}
            constraints: List[Tuple[str, str, float]] = []
            for raw_key, value in data.items():
                numerator, denominator = self._split_ratio_key(raw_key)
                if denominator == self.abundance_reference:
                    ratio_key = _build_ratio_key(numerator, denominator)
                    if ratio_key not in self.relative_ratio_to_ions:
                        allowed = ", ".join(sorted(self.supported_relative_keys))
                        raise ValueError(
                            f"Unknown relative-abundance key '{raw_key}'. Allowed keys for reference "
                            f"'{self.abundance_reference}' are: {allowed}"
                        )
                    canonical[ratio_key] = float(value)
                else:
                    constraints.append((numerator, denominator, float(value)))
            return canonical, constraints

        abundance_bounds = _canonicalise_bounds(raw_bounds)
        abundance_fixed, nonref_constraints = _canonicalise_fixed(raw_fixed)

        self.relative_param_keys: List[str] = []
        for key, bound in abundance_bounds.items():
            lo, hi = bound
            if lo > hi:
                raise ValueError(f"Relative-abundance bounds invalid for {key}: ({lo}, {hi})")
            self.param_order.append(key)
            self.param_bounds.append((float(lo), float(hi)))
            self.relative_param_keys.append(key)

        self.relative_abundance_fixed = {key: float(val) for key, val in abundance_fixed.items()}
        for key in self.supported_relative_keys:
            if key not in abundance_bounds and key not in self.relative_abundance_fixed:
                self.relative_abundance_fixed[key] = 0.0

        self.relative_abundance_constraints: List[Tuple[str, str, float]] = []
        for numerator, denominator, value in nonref_constraints:
            if numerator == self.abundance_reference:
                num_key = None
                num_free = False
                num_known = True
                num_fixed_value = 0.0
            else:
                num_key = _build_ratio_key(numerator, self.abundance_reference)
                num_free = num_key in abundance_bounds
                num_known = num_key in abundance_bounds or num_key in self.relative_abundance_fixed
                num_fixed_value = self.relative_abundance_fixed.get(num_key)

            den_key = _build_ratio_key(denominator, self.abundance_reference)
            den_free = den_key in abundance_bounds
            den_known = den_key in abundance_bounds or den_key in self.relative_abundance_fixed
            den_fixed_value = self.relative_abundance_fixed.get(den_key)

            if num_free and den_free:
                raise ValueError(
                    f"Fixed relative-abundance ratio '{numerator}_{denominator}' cannot be enforced because "
                    f"both '{num_key}' and '{den_key}' are free. Fix one of them or remove the constraint."
                )

            if not (num_known or den_known):
                raise ValueError(
                    f"Fixed relative-abundance ratio '{numerator}_{denominator}' requires either '{num_key}' "
                    f"or '{den_key}' to be specified in free/fixed abundances."
                )

            if num_fixed_value is not None and den_fixed_value is not None:
                expected = num_fixed_value - den_fixed_value
                if not np.isclose(expected, value, rtol=0, atol=1e-6):
                    raise ValueError(
                        f"Fixed relative-abundance ratio '{numerator}_{denominator}' inconsistent with fixed "
                        f"values for '{num_key}' and '{den_key}'."
                    )
            elif numerator == self.abundance_reference and den_fixed_value is not None:
                expected = 0.0 - den_fixed_value
                if not np.isclose(expected, value, rtol=0, atol=1e-6):
                    raise ValueError(
                        f"Fixed relative-abundance ratio '{numerator}_{denominator}' inconsistent with fixed "
                        f"value for '{den_key}'."
                    )

            self.relative_abundance_constraints.append((numerator, denominator, float(value)))

        self.ndim = len(self.param_order)
        self.sampler = None
        self.samples = None

        # Pre-build Voigt temperature z-score observation (single component)
        self._voigt_temp_obs: Optional[ZScoreObservation] = None
        if config.use_voigt_temperature_prior and config.voigt_temperature_samples is not None:
            # For single-component fitter, use index 0 of the list
            t_samples_list = config.voigt_temperature_samples
            if len(t_samples_list) > 0 and t_samples_list[0] is not None:
                zscore_cfg = ZScoreGridConfig()
                self._voigt_temp_obs = ZScoreObservation.from_samples(
                    t_samples_list[0], zscore_cfg
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _canonicalise_ratio_key(self, key: str) -> str:
        numerator, denominator = self._split_ratio_key(key)
        if denominator != self.abundance_reference:
            raise ValueError(
                f"Relative-abundance key '{key}' must use reference element '{self.abundance_reference}'"
            )
        ratio_key = _build_ratio_key(numerator, denominator)
        if ratio_key not in self.relative_ratio_to_ions:
            allowed = ", ".join(sorted(self.supported_relative_keys))
            raise ValueError(
                f"Unknown relative-abundance key '{key}'. Allowed keys for reference "
                f"'{self.abundance_reference}' are: {allowed}"
            )
        return ratio_key

    @staticmethod
    def _split_ratio_key(key: str) -> Tuple[str, str]:
        if "_" not in key:
            raise ValueError(
                "Relative-abundance keys must be in 'Element_Reference' format, e.g., 'N_O'"
            )
        numerator_raw, denominator_raw = key.split("_", 1)
        numerator = _normalise_element_symbol(numerator_raw)
        denominator = _normalise_element_symbol(denominator_raw)
        if numerator == denominator:
            raise ValueError("Relative-abundance numerator and denominator must be different elements")
        return numerator, denominator

    def _compose_parameter_dict(self, theta: Sequence[float]) -> Dict[str, float]:
        params: Dict[str, float] = {}
        if "z" in self.grid.axis_names:
            params["z"] = self.config.redshift

        for value, key in zip(theta, self.param_order):
            params[key] = value

        if self.config.temperature_mode in {"fixed_voigt", "fixed_value"}:
            params["T"] = self.fixed_temperature

        for key, value in self.relative_abundance_fixed.items():
            params.setdefault(key, value)

        for numerator, denominator, value in self.relative_abundance_constraints:
            if numerator == self.abundance_reference:
                num_key = None
                num_val = 0.0
            else:
                num_key = _build_ratio_key(numerator, self.abundance_reference)
                num_val = params.get(num_key)
            den_key = _build_ratio_key(denominator, self.abundance_reference)
            den_val = params.get(den_key)
            if num_val is not None and den_val is not None:
                continue
            if num_val is not None:
                params[den_key] = float(num_val - value)
            elif den_val is not None:
                if num_key is not None:
                    params[num_key] = float(den_val + value)

        for key in self.supported_relative_keys:
            params.setdefault(key, 0.0)

        return params

    def _apply_abundance_offsets(self, model: Dict[str, float], params: Mapping[str, float]) -> None:
        for key, ions in self.relative_ratio_to_ions.items():
            offset = params.get(key, self.relative_abundance_fixed.get(key, 0.0))
            for ion in ions:
                if ion in model:
                    model[ion] += offset

    def _evaluate_column_for_theta(self, theta: Sequence[float], ion: str) -> Optional[float]:
        params = self._compose_parameter_dict(theta)
        model = self.grid.evaluate(params)
        self._apply_abundance_offsets(model, params)
        value = model.get(ion)
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    def _evaluate_column_for_samples(self, ion: str) -> np.ndarray:
        if self.samples is None:
            raise RuntimeError("Run MCMC before evaluating column densities for samples")
        values = np.full(self.samples.shape[0], np.nan, dtype=float)
        for idx, theta in enumerate(self.samples):
            value = self._evaluate_column_for_theta(theta, ion)
            if value is not None:
                values[idx] = value
        return values

    def _log_prior(self, theta: Sequence[float]) -> float:
        for value, bounds in zip(theta, self.param_bounds):
            lo, hi = bounds
            if value < lo or value > hi:
                return -np.inf
        return 0.0

    def _log_probability(self, theta: Sequence[float]) -> float:
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        params = self._compose_parameter_dict(theta)
        model = self.grid.evaluate(params)
        self._apply_abundance_offsets(model, params)
        temperature_term = 0.0
        if self.temperature_observation is not None and self.config.temperature_mode == "free":
            temp_obs = self.temperature_observation
            model_temp = params.get("T")
            if model_temp is not None:
                if temp_obs.is_upper_limit:
                    temperature_term = _loglike_upper_limit(model_temp, temp_obs)
                else:
                    temperature_term = _loglike_gaussian(model_temp, temp_obs)

        ll = lp + log_likelihood(model, self.observations) + temperature_term

        # Derived-quantity priors (cloud length, mass, Jeans, T)
        cfg = self.config
        if (cfg.cloud_length_bounds is not None or cfg.cloud_mass_bounds is not None
                or cfg.jeans_length_prior or cfg.use_voigt_temperature_prior):
            log_n_H = params.get("n_H")
            log_N_H = params.get("N_H") or params.get("NHI")

            if log_N_H is not None and log_n_H is not None:
                log_L_kpc = log_N_H - log_n_H - LOG10_CM_PER_KPC
                log_M_solar = (3.0 * log_N_H - 2.0 * log_n_H
                               + LOG10_PROTON_MASS_G - LOG10_SOLAR_MASS_G)

                if cfg.cloud_length_bounds is not None:
                    lo, hi = cfg.cloud_length_bounds
                    if log_L_kpc < lo or log_L_kpc > hi:
                        return -np.inf

                if cfg.cloud_mass_bounds is not None:
                    lo, hi = cfg.cloud_mass_bounds
                    if log_M_solar < lo or log_M_solar > hi:
                        return -np.inf

            # Jeans & temperature prior
            log_T = model.get("T_cloudy") or params.get("T")
            if log_T is not None and np.isfinite(log_T):
                if cfg.jeans_length_prior and log_N_H is not None and log_n_H is not None:
                    log_L_J_kpc = (0.5 * (_JEANS_A_CONST + log_T)
                                   - 0.5 * log_n_H - LOG10_CM_PER_KPC)
                    if log_L_kpc > log_L_J_kpc:
                        return -np.inf

                if cfg.use_voigt_temperature_prior and self._voigt_temp_obs is not None:
                    ll += _loglike_zscore(log_T, self._voigt_temp_obs)

        # Priors on model total ion columns (single-component case).
        if cfg.total_column_density_bounds is not None:
            for ion, (lo, hi) in cfg.total_column_density_bounds.items():
                pred = model.get(ion)
                if pred is None or not np.isfinite(pred):
                    return -np.inf
                # Use Observation-like soft limits for numerical stability,
                # matching the sigma=0.01 convention used for limit constraints.
                ll += _loglike_lower_limit(
                    pred,
                    Observation(value=float(lo), sigma=DEFAULT_SOFT_LIMIT_SIGMA, is_lower_limit=True),
                )
                ll += _loglike_upper_limit(
                    pred,
                    Observation(value=float(hi), sigma=DEFAULT_SOFT_LIMIT_SIGMA, is_upper_limit=True),
                )

        return ll

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initial_positions(self, nwalkers: int, scatter: float = 1e-2) -> np.ndarray:
        p0 = []
        rng = np.random.default_rng()
        for bounds in self.param_bounds:
            lo, hi = bounds
            if np.isclose(hi, lo):
                arr = np.full(nwalkers, lo)
            else:
                if not np.isfinite(lo) or not np.isfinite(hi):
                    arr = rng.normal(loc=0.0, scale=1.0, size=nwalkers)
                else:
                    arr = rng.uniform(lo, hi, size=nwalkers)
            p0.append(arr)
        p0 = np.stack(p0, axis=1)
        if scatter > 0:
            p0 += scatter * rng.standard_normal(size=p0.shape)
        for dim, (lo, hi) in enumerate(self.param_bounds):
            p0[:, dim] = np.clip(p0[:, dim], lo, hi)
        return p0

    def run_mcmc(
        self,
        nwalkers: int = 200,
        nsteps: int = 2000,
        burnin: Optional[int] = 500,
        progress: bool = True,
        initial_positions: Optional[np.ndarray] = None,
        **sampler_kwargs,
    ):
        if initial_positions is None:
            initial_positions = self.initial_positions(nwalkers)
        if initial_positions.shape != (nwalkers, self.ndim):
            raise ValueError("Initial positions shape mismatch")

        if burnin is not None:
            if burnin < 0:
                raise ValueError("burnin must be non-negative")
            if burnin >= nsteps:
                raise ValueError("burnin must be less than nsteps")
            discard = int(burnin)
        else:
            discard = 0

        sampler = emcee.EnsembleSampler(
            nwalkers,
            self.ndim,
            self._log_probability,
            **sampler_kwargs,
        )
        sampler.run_mcmc(initial_positions, nsteps, progress=progress)
        self.sampler = sampler
        self.samples = sampler.get_chain(discard=discard, flat=True)
        return sampler

    def posterior_column_densities(
        self,
        ions: Optional[Sequence[str]] = None,
        *,
        thin: int = 1,
        max_samples: Optional[int] = None,
        random_state: Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        """Evaluate Cloudy-predicted columns for posterior samples.

        Parameters
        ----------
        ions
            Limit the returned mapping to these ion column names. Defaults to
            every ion available from the underlying Cloudy grid.
        thin
            Keep one sample out of every ``thin`` entries (default: 1, i.e. no thinning).
        max_samples
            Optionally down-sample to at most this many posterior draws after
            thinning. Sampling is without replacement to preserve diversity.
        random_state
            Optional :class:`numpy.random.Generator` used when sub-selecting
            samples. When omitted, a fresh generator is created.

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping of ion name to ``log10`` column-density predictions for the
            retained posterior draws.
        """

        if self.samples is None:
            raise RuntimeError("Run MCMC before calling posterior_column_densities")
        if thin < 1:
            raise ValueError("thin must be >= 1")

        draw_indices = np.arange(0, self.samples.shape[0], thin, dtype=int)
        if draw_indices.size == 0:
            raise ValueError("Thinning removed all samples; check the 'thin' setting")

        if max_samples is not None and max_samples > 0 and draw_indices.size > max_samples:
            rng = random_state if random_state is not None else np.random.default_rng()
            draw_indices = np.sort(rng.choice(draw_indices, size=max_samples, replace=False))

        target_ions: Sequence[str]
        if ions is None:
            target_ions = self.grid.ion_order
        else:
            missing = [ion for ion in ions if ion not in self.grid.ion_order]
            if missing:
                logger.warning("Requested ions not present in Cloudy grid: %s", missing)
            target_ions = ions

        collected: Dict[str, List[float]] = {ion: [] for ion in target_ions}
        for idx in draw_indices:
            theta = self.samples[idx]
            params = self._compose_parameter_dict(theta)
            model = self.grid.evaluate(params)
            self._apply_abundance_offsets(model, params)
            for ion in target_ions:
                value = model.get(ion)
                if value is not None and np.isfinite(value):
                    collected[ion].append(float(value))

        return {ion: np.asarray(values, dtype=float) for ion, values in collected.items() if values}

    def plot_column_density_violin(
        self,
        ions: Optional[Sequence[str]] = None,
        *,
        thin: int = 1,
        max_samples: Optional[int] = None,
        random_state: Optional[np.random.Generator] = None,
        voigt_samples: Optional[Mapping[str, ArrayLike]] = None,
        credibility_percentiles: Sequence[Tuple[float, float]] = ((2.5, 97.5), (16.0, 84.0)),
        figsize: Tuple[float, float] = (8.0, 5.0),
        violin_kwargs: Optional[Mapping[str, Any]] = None,
        scatter_kwargs: Optional[Mapping[str, Any]] = None,
    ax: Optional["Axes"] = None,
    ):
        """Generate violin plots of Cloudy-predicted columns with Voigt overlays.

        The violins represent Cloudy model predictions evaluated across the
        posterior samples, while markers/lines denote percentile summaries from
        the Voigt-profile posterior for direct comparison.

        Parameters
        ----------
        ions
            Sequence of ion names to display. Defaults to all ions produced by
            the Cloudy grid.
        thin, max_samples, random_state
            Passed through to :meth:`posterior_column_densities` to control
            sub-sampling of posterior draws.
        voigt_samples
            Mapping from ion name to the corresponding Voigt posterior samples
            (in ``log10`` column space). When provided, percentile markers are
            over-plotted for each ion.
        credibility_percentiles
            Ordered sequence of percentile pairs defining the credible intervals
            to draw (e.g. ``((2.5, 97.5), (16, 84))``). Pairs are plotted from
            widest to narrowest using decreasing line opacity.
        figsize
            Figure size passed to :func:`matplotlib.pyplot.subplots` when
            creating a new axes.
        violin_kwargs
            Additional keyword arguments forwarded to
            :func:`matplotlib.axes.Axes.violinplot`.
        scatter_kwargs
            Base keyword arguments for the percentile markers.
        ax
            Optional axes to draw onto. When omitted, a new figure and axes are
            created and returned.

        Returns
        -------
        tuple
            ``(fig, ax)`` containing the Matplotlib figure and axes used for
            the plot.
        """

        posterior = self.posterior_column_densities(
            ions=ions,
            thin=thin,
            max_samples=max_samples,
            random_state=random_state,
        )
        if not posterior:
            raise RuntimeError("No posterior column densities available to plot")

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("matplotlib is required to plot column-density violins") from exc

        ordered_ions = [ion for ion in (ions or self.grid.ion_order) if ion in posterior]
        positions = np.arange(1, len(ordered_ions) + 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        vk = {"showmeans": False, "showmedians": False, "showextrema": False}
        if violin_kwargs:
            vk.update(dict(violin_kwargs))

        datasets = [posterior[ion] for ion in ordered_ions]
        parts = ax.violinplot(datasets, positions=positions, **vk)

        for body in parts.get("bodies", []):
            body.set_facecolor("#6baed6")
            body.set_edgecolor("#084594")
            body.set_alpha(0.6)

        line_color_cycle = ["#54278f", "#de2d26", "#ff7f00", "#238b45"]
        scatter_defaults = {"s": 36, "zorder": 5}
        if scatter_kwargs:
            scatter_defaults.update(dict(scatter_kwargs))

        upper_limit_values: Dict[str, float] = {}
        for ion in ordered_ions:
            obs = self.observations.get(ion)
            if isinstance(obs, Observation):
                if obs.is_upper_limit and obs.value is not None:
                    upper_limit_values[ion] = float(obs.value)
            elif isinstance(obs, ZScoreObservation):
                if obs.is_upper_limit:
                    limit_value = obs.upper_limit_value if obs.upper_limit_value is not None else obs.value_grid[-1]
                    upper_limit_values[ion] = float(limit_value)

        handles = []
        labels = []
        if voigt_samples is not None:
            color_iter = iter(line_color_cycle)
            for pair_index, (lo_pct, hi_pct) in enumerate(sorted(credibility_percentiles, reverse=True)):
                color = next(color_iter, "#252525")
                marker_size = scatter_defaults.get("s", 36) * (0.9 ** pair_index)
                label = f"{100 - 2 * lo_pct:.0f}% credible"
                for xpos, ion in zip(positions, ordered_ions):
                    samples = np.asarray(voigt_samples.get(ion, []), dtype=float)
                    samples = samples[np.isfinite(samples)]
                    if samples.size == 0:
                        continue
                    lo, hi = np.percentile(samples, [lo_pct, hi_pct])
                    ax.plot([xpos, xpos], [lo, hi], color=color, linewidth=2 - 0.3 * pair_index, alpha=0.8)
                    ax.scatter([xpos, xpos], [lo, hi], color=color, s=marker_size, alpha=0.9)
                handles.append(ax.plot([], [], color=color, linewidth=1.8, label=label)[0])
                labels.append(label)

            # Median markers after intervals so they are visible.
            for xpos, ion in zip(positions, ordered_ions):
                samples = np.asarray(voigt_samples.get(ion, []), dtype=float)
                samples = samples[np.isfinite(samples)]
                if samples.size == 0:
                    continue
                median = np.percentile(samples, 50)
                ax.scatter([xpos], [median], color="#4a1486", marker="D", s=scatter_defaults.get("s", 36) * 1.2, zorder=6)
            handles.append(ax.scatter([], [], color="#4a1486", marker="D", s=scatter_defaults.get("s", 36) * 1.2))
            labels.append("Voigt median")

        if upper_limit_values:
            upper_kwargs = {
                "marker": "v",
                "color": "#b30000",
                "s": scatter_defaults.get("s", 36) * 1.5,
                "zorder": 7,
            }
            for xpos, ion in zip(positions, ordered_ions):
                limit = upper_limit_values.get(ion)
                if limit is None or not np.isfinite(limit):
                    continue
                ax.scatter([xpos], [limit], **upper_kwargs)
            handles.append(ax.scatter([], [], **upper_kwargs))
            labels.append("Upper limit")

        ax.set_xticks(positions)
        ax.set_xticklabels(ordered_ions, rotation=30, ha="right")
        ax.set_ylabel("log$_{10}$ column density")
        ax.set_title("Cloudy vs. Voigt column-density posteriors")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if handles and labels:
            ax.legend(handles, labels, loc="best")

        return fig, ax

    def _overlay_jeans_boundary(
        self,
        fig,
        plot_labels: List[str],
        plot_samples: np.ndarray,
        temperature_samples: np.ndarray,
        *,
        jeans_color: str = "forestgreen",
        jeans_alpha: float = 0.15,
        jeans_label: str = "Jeans instability limit (PIE)",
    ) -> None:
        """Overlay Jeans instability boundary on corner-plot panels involving n_H, N_H, and L."""
        finite_mask = np.isfinite(temperature_samples)
        if not np.any(finite_mask):
            logger.warning("No finite T samples available for Jeans overlay")
            return

        t_finite = temperature_samples[finite_mask]
        log_T_16, log_T_50, log_T_84 = np.percentile(t_finite, [16, 50, 84])

        def _B(log_T: float) -> float:
            return _JEANS_A_CONST + log_T

        # Locate relevant axes in the corner-plot label list
        idx_nH: Optional[int] = None
        idx_NH: Optional[int] = None
        idx_L: Optional[int] = None
        nH_label = CORNER_LABEL_OVERRIDES.get("n_H", "log n_H")
        NH_label = CORNER_LABEL_OVERRIDES.get("N_H", "log N_H (total)")
        for i, label in enumerate(plot_labels):
            if label == nH_label:
                idx_nH = i
            elif label == NH_label or label == "log N_H_total":
                idx_NH = i
            elif "log L" in label and "N_H / n_H" in label:
                idx_L = i

        if idx_nH is None:
            logger.warning("Cannot overlay Jeans boundary: n_H axis not found in corner plot")
            return

        ndim = plot_samples.shape[1]
        axes = np.array(fig.axes).reshape(ndim, ndim)

        label_used = False

        def _add_jeans_line(ax, x_range, y_func, log_T_vals):
            nonlocal label_used
            x = np.linspace(x_range[0], x_range[1], 300)
            y_med = y_func(x, _B(log_T_vals[1]))
            y_lo = y_func(x, _B(log_T_vals[0]))
            y_hi = y_func(x, _B(log_T_vals[2]))
            line_label = jeans_label if not label_used else None
            ax.plot(x, y_med, ":", color=jeans_color, linewidth=1.5,
                    label=line_label, zorder=10)
            ax.fill_between(x, y_lo, y_hi, alpha=jeans_alpha, color=jeans_color, zorder=5)
            label_used = True

        T_vals = (log_T_16, log_T_50, log_T_84)

        # --- Panel (n_H, N_H): log N_H = 0.5*log_nH + 0.5*B(log_T) ---
        if idx_NH is not None:
            if idx_NH > idx_nH:
                ax = axes[idx_NH, idx_nH]
                xlim = ax.get_xlim()
                _add_jeans_line(ax, xlim, lambda x, b: 0.5 * x + 0.5 * b, T_vals)
                ax.set_xlim(xlim)
            elif idx_nH > idx_NH:
                ax = axes[idx_nH, idx_NH]
                xlim = ax.get_xlim()  # x = log N_H
                _add_jeans_line(ax, xlim, lambda x, b: 2.0 * x - b, T_vals)
                ax.set_xlim(xlim)

        # --- Panel (n_H, L): log L_kpc = -0.5*log_nH + 0.5*B(log_T) - LOG_CM_KPC ---
        if idx_L is not None and idx_L > idx_nH:
            ax = axes[idx_L, idx_nH]
            xlim = ax.get_xlim()
            _add_jeans_line(
                ax, xlim,
                lambda x, b: -0.5 * x + 0.5 * b - LOG10_CM_PER_KPC,
                T_vals,
            )
            ax.set_xlim(xlim)

        # --- Panel (N_H, L): log L_kpc = -log_NH + B(log_T) - LOG_CM_KPC ---
        if idx_NH is not None and idx_L is not None and idx_L > idx_NH:
            ax = axes[idx_L, idx_NH]
            xlim = ax.get_xlim()
            _add_jeans_line(
                ax, xlim,
                lambda x, b: -x + b - LOG10_CM_PER_KPC,
                T_vals,
            )
            ax.set_xlim(xlim)

    def corner_plot(
        self,
        truths: Optional[Sequence[float]] = None,
        labels: Optional[Sequence[str]] = None,
        jeans_overlay: bool = False,
        jeans_kwargs: Optional[Mapping[str, Any]] = None,
        **corner_kwargs,
    ):
        if self.samples is None:
            raise RuntimeError("Run MCMC before calling corner_plot")
        try:
            import corner  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install 'corner' to create corner plots") from exc

        base_samples = self.samples
        base_dim = base_samples.shape[1]

        derived_arrays: List[np.ndarray] = []
        derived_labels: List[str] = []

        has_total_axis = "N_H" in self.param_order
        has_neutral_axis = "NHI" in self.param_order
        idx_total_axis = self.param_order.index("N_H") if has_total_axis else None
        idx_neutral_axis = self.param_order.index("NHI") if has_neutral_axis else None

        total_columns: Optional[np.ndarray] = None
        add_total_dimension = False
        if has_total_axis and idx_total_axis is not None:
            total_columns = base_samples[:, idx_total_axis]
        elif "N_H_total" in self.grid.ion_order:
            evaluated_total = self._evaluate_column_for_samples("N_H_total")
            if np.all(~np.isfinite(evaluated_total)):
                logger.warning(
                    "Unable to evaluate N_H_total for posterior samples; skipping total column in corner plot"
                )
            else:
                total_columns = evaluated_total
                derived_arrays.append(evaluated_total[:, np.newaxis])
                derived_labels.append("log N_H_total")
                add_total_dimension = True

        neutral_columns: Optional[np.ndarray] = None
        add_neutral_dimension = False
        if has_neutral_axis and idx_neutral_axis is not None:
            neutral_columns = base_samples[:, idx_neutral_axis]
        elif "N_HI" in self.grid.ion_order:
            evaluated_neutral = self._evaluate_column_for_samples("N_HI")
            if np.all(~np.isfinite(evaluated_neutral)):
                logger.warning(
                    "Unable to evaluate N_HI for posterior samples; skipping neutral column in corner plot"
                )
            else:
                neutral_columns = evaluated_neutral
                neutral_label = CORNER_LABEL_OVERRIDES.get("NHI", "log N_HI")
                derived_arrays.append(evaluated_neutral[:, np.newaxis])
                derived_labels.append(neutral_label)
                add_neutral_dimension = True

        include_length = (
            total_columns is not None
            and np.any(np.isfinite(total_columns))
            and "n_H" in self.param_order
        )
        if include_length:
            idx_density = self.param_order.index("n_H")
            log_length_cm = total_columns - base_samples[:, idx_density]
            log_length_kpc = log_length_cm - LOG10_CM_PER_KPC
            derived_arrays.append(log_length_kpc[:, np.newaxis])
            derived_labels.append("log L (kpc)\n[N_H / n_H]")

            # Mass calculation: M = n_H * L^3 * m_p
            # log M = log n_H + 3 * log L + log m_p
            #       = log n_H + 3 * (log N_H - log n_H) + log m_p
            #       = 3 log N_H - 2 log n_H + log m_p
            log_mass_g = (3.0 * total_columns) - (2.0 * base_samples[:, idx_density]) + LOG10_PROTON_MASS_G
            log_mass_solar = log_mass_g - LOG10_SOLAR_MASS_G
            derived_arrays.append(log_mass_solar[:, np.newaxis])
            derived_labels.append("log M (M_sun)\n[n_H * L^3 * m_p]")

        include_cloudy_temperature = "T_cloudy" in self.grid.ion_order and "T" not in self.param_order
        cloudy_temperature_samples: Optional[np.ndarray] = None
        if include_cloudy_temperature:
            cloudy_temperature_samples = self._evaluate_column_for_samples("T_cloudy")
            if np.all(~np.isfinite(cloudy_temperature_samples)):
                logger.warning(
                    "Unable to evaluate Cloudy equilibrium temperature for posterior samples; skipping temperature output"
                )
                include_cloudy_temperature = False
            else:
                derived_arrays.append(cloudy_temperature_samples[:, np.newaxis])
                derived_labels.append("log T_cloudy")

        include_cooling_time = "log_tcool" in self.grid.ion_order and "T" in self.param_order
        cooling_time_samples: Optional[np.ndarray] = None
        if include_cooling_time:
            cooling_time_samples = self._evaluate_column_for_samples("log_tcool")
            if np.all(~np.isfinite(cooling_time_samples)):
                logger.warning(
                    "Unable to evaluate cooling-time predictions for posterior samples; skipping t_cool output"
                )
                include_cooling_time = False
            else:
                derived_arrays.append(cooling_time_samples[:, np.newaxis])
                derived_labels.append("log t_cool (s)")

        plot_samples = base_samples
        if derived_arrays:
            plot_samples = np.hstack([base_samples] + derived_arrays)

        finite_mask = np.all(np.isfinite(plot_samples), axis=1)
        if not np.all(finite_mask):
            logger.warning(
                "Dropping %d posterior samples with non-finite derived parameters in corner plot",
                int(plot_samples.shape[0] - np.count_nonzero(finite_mask)),
            )
            plot_samples = plot_samples[finite_mask]
            if plot_samples.size == 0:
                raise RuntimeError(
                    "All posterior samples contained non-finite values once derived parameters were added; "
                    "try disabling derived outputs or investigating grid completeness."
                )

        derived_count = len(derived_labels)

        if labels is None:
            base_labels = [CORNER_LABEL_OVERRIDES.get(name, name) for name in self.param_order]
            plot_labels: List[str] = base_labels + derived_labels
        else:
            plot_labels = list(labels)
            if len(plot_labels) == base_dim:
                plot_labels = plot_labels + derived_labels
            elif len(plot_labels) != base_dim + derived_count:
                raise ValueError(
                    "Provided labels must match the base parameter count or the extended set including derived columns"
                )

        if len(plot_labels) != plot_samples.shape[1]:
            raise ValueError("Number of labels does not match corner plot dimensionality")

        if truths is None:
            plot_truths: Optional[Sequence[float]] = None
        else:
            truth_list = list(truths)
            if len(truth_list) == base_dim:
                derived_truths: List[float] = []
                require_theta = (
                    (add_total_dimension and not has_total_axis)
                    or add_neutral_dimension
                    or include_cloudy_temperature
                    or (include_length and not has_total_axis)
                )
                theta_values: List[float] = []
                valid_theta = True
                if require_theta:
                    for val in truth_list:
                        try:
                            theta_values.append(float(val))
                        except (TypeError, ValueError):
                            valid_theta = False
                            break
                    if not theta_values or not np.all(np.isfinite(theta_values)):
                        valid_theta = False

                total_truth_val = np.nan
                if has_total_axis and idx_total_axis is not None:
                    try:
                        total_truth_val = float(truth_list[idx_total_axis])
                    except (TypeError, ValueError):
                        total_truth_val = np.nan
                elif (add_total_dimension or include_length) and valid_theta:
                    total_eval = self._evaluate_column_for_theta(theta_values, "N_H_total")
                    if total_eval is not None:
                        total_truth_val = total_eval

                if add_total_dimension:
                    derived_truths.append(total_truth_val)

                if add_neutral_dimension:
                    neutral_truth_val = np.nan
                    if valid_theta:
                        neutral_eval = self._evaluate_column_for_theta(theta_values, "N_HI")
                        if neutral_eval is not None:
                            neutral_truth_val = neutral_eval
                    derived_truths.append(neutral_truth_val)

                if include_length:
                    density_idx = self.param_order.index("n_H")
                    length_truth_val = np.nan
                    try:
                        density_value = float(truth_list[density_idx])
                    except (TypeError, ValueError):
                        density_value = np.nan
                    if np.isfinite(total_truth_val) and np.isfinite(density_value):
                        length_truth_val = total_truth_val - density_value - LOG10_CM_PER_KPC
                    derived_truths.append(length_truth_val)

                if include_cloudy_temperature:
                    temperature_truth_val = np.nan
                    if valid_theta:
                        temp_eval = self._evaluate_column_for_theta(theta_values, "T_cloudy")
                        if temp_eval is not None:
                            temperature_truth_val = temp_eval
                    derived_truths.append(temperature_truth_val)

                if include_cooling_time:
                    cooling_truth_val = np.nan
                    if valid_theta:
                        tcool_eval = self._evaluate_column_for_theta(theta_values, "log_tcool")
                        if tcool_eval is not None:
                            cooling_truth_val = tcool_eval
                    derived_truths.append(cooling_truth_val)

                truth_list.extend(derived_truths)
            elif len(truth_list) != base_dim + derived_count:
                raise ValueError(
                    "Provided truths must match the base parameter count or the extended set including derived columns"
                )
            plot_truths = truth_list

        self._last_corner_labels = list(plot_labels)
        
        # Default to contours only, 1/2/3 sigma
        defaults = dict(
            plot_datapoints=False,
            plot_density=False,
            plot_contours=True,
            no_fill_contours=True,
            levels=(0.68, 0.95, 0.99),
            hist_kwargs={"density": True},
        )
        defaults.update(corner_kwargs)
        
        fig = corner.corner(
            plot_samples,
            labels=plot_labels,
            truths=plot_truths,
            range=[0.995] * plot_samples.shape[1],
            **defaults,
        )

        # Colour diagonal histograms to match labelled panel meanings
        try:
            import matplotlib.patches as mpatches
            import matplotlib as mpl
            for ax in fig.get_axes():
                # Identify histogram axes by presence of Rectangle patches
                rects = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)]
                if not rects:
                    continue
                # Determine the panel label (fall back to title if xlabel empty)
                lbl = (ax.get_xlabel() or ax.get_title() or "")
                # Map special suffixes to colours – keep defaults otherwise
                if "[pred]" in lbl:
                    c = "#de2d26"
                elif "[obs]" in lbl:
                    c = "#000000"
                else:
                    c = None
                if c is not None:
                    for r in rects:
                        r.set_facecolor(c)
                        r.set_edgecolor("none")
                        r.set_alpha(0.6)
        except Exception:
            # Non-fatal: plotting nicety, don't break user code on failures
            pass

        if jeans_overlay:
            # Find temperature samples in the already-filtered plot_samples
            t_idx: Optional[int] = None
            for i, lbl in enumerate(plot_labels):
                if lbl == "log T_cloudy":
                    t_idx = i
                    break
            if t_idx is None and "T" in self.param_order:
                t_idx = self.param_order.index("T")
            if t_idx is not None:
                t_samples = plot_samples[:, t_idx]
                overlay_kw = dict(jeans_kwargs) if jeans_kwargs else {}
                self._overlay_jeans_boundary(fig, plot_labels, plot_samples, t_samples, **overlay_kw)
            else:
                logger.warning(
                    "Cannot overlay Jeans boundary: no temperature data available in corner plot"
                )

        return fig


class KDEObservation:
    """Wraps a Kernel Density Estimator (scipy or KDEpy)."""

    def __init__(self, samples: ArrayLike, param_names: Sequence[str], max_samples: int = 100_000, bandwidth: Union[str, float] = 'silverman', bandwidth_scale: float = 1.0, method: str = 'scipy') -> None:
        """
        :param samples: (N_samples, N_params) array of posterior chain samples
        :param param_names: List of parameter names corresponding to columns
        :param max_samples: Maximum number of samples to use for KDE construction (randomly subsampled)
        :param bandwidth: Bandwidth selection method ('scott', 'silverman', or float). 
        :param bandwidth_scale: Multiplier for the selected bandwidth (default 1.0).
        :param method: 'scipy' or 'kdepy'.
                       Defaults to 'scipy' because 'kdepy' (scalar bw) can wash out constraints in correlated high-D data.
        """
        self.param_names = list(param_names)
        
        # Subsample if too large
        self.dataset = np.asarray(samples)
        n_total = self.dataset.shape[0]
        if n_total > max_samples:
            indices = np.random.choice(n_total, max_samples, replace=False)
            self.dataset = self.dataset[indices, :]
            
        self.n = self.dataset.shape[0]
        self.d = self.dataset.shape[1]
        
        self.use_kdepy = (method == 'kdepy') and HAS_KDEPY
        self.kde = None
        
        if method == 'kdepy' and not HAS_KDEPY:
            logger.warning("KDEpy requested but not installed. Falling back to Scipy.")
        
        if self.use_kdepy:
            try:
                # Calculate manual bandwidth for TreeKDE (auto selectors often fail for N-dim)
                bw_val = 1.0
                if isinstance(bandwidth, str):
                    if bandwidth == 'scott':
                        bw_val = self.n**(-1./(self.d+4))
                    elif bandwidth == 'silverman':
                        bw_val = (self.n * (self.d + 2) / 4.)**(-1. / (self.d + 4))
                    else:
                        # Fallback for other strings (e.g. 'ISJ') to silverman and warn
                         print(f"Warning: Bandwidth '{bandwidth}' not fully supported for TreeKDE N-dim. Using 'silverman' rule.")
                         bw_val = (self.n * (self.d + 2) / 4.)**(-1. / (self.d + 4))
                elif isinstance(bandwidth, (float, int)):
                    bw_val = float(bandwidth)
                
                # Apply scale
                bw_val *= bandwidth_scale
                
                # Fit TreeKDE
                # Note: We must ensure data is double precision
                self.kde = TreeKDE(bw=bw_val).fit(self.dataset.astype(np.float64))
            except Exception as e:
                print(f"KDEpy initialization failed ({e}). Falling back to Scipy.")
                self.use_kdepy = False

        if not self.use_kdepy:
            # Scipy fallback (supports 'scott' and 'silverman' strings natively)
            # If bandwidth was a float, scipy accepts it as scalar factor? 
            # Scipy bw_method can be scalar or callable or string.
            self.kde = gaussian_kde(self.dataset.T, bw_method=bandwidth)
            
            # Apply scale: scipy stores covariance_factor. We can manipulate it or use set_bandwidth
            # set_bandwidth(bw_method) re-computes.
            # We want current_factor * scale.
            # self.kde.factor usually returns the scalar multiplier.
            # So:
            current_factor = self.kde.factor
            self.kde.set_bandwidth(bw_method=current_factor * bandwidth_scale)

    def log_prob(self, values: ArrayLike) -> float:
        """Evaluate log PDF at a single point or array of points."""
        # gaussian_kde.logpdf expects (dims, n_points)
        # If values is 1D (dims,), treat as single point
        values = np.asarray(values)
        if values.ndim == 1:
            values_2d = values.reshape(1, -1)
            if self.use_kdepy:
                try:
                    probs = self.kde.evaluate(values_2d.astype(np.float64))
                    return float(np.log(probs + 1e-300)[0])
                except Exception:
                     # Fallback
                     return -np.inf
            else:
                return float(self.kde.logpdf(values_2d.T)[0])
        else:
            if self.use_kdepy:
                try:
                    probs = self.kde.evaluate(values.astype(np.float64))
                    return np.log(probs + 1e-300)
                except Exception:
                     return np.full(values.shape[0], -np.inf)
            else:
                return self.kde.logpdf(values.T)

    def plot_corner(self, figsize: Tuple[float, float] = (10, 10), **corner_kwargs):
        """Plot the KDE samples (resampled) or original dataset to verify."""
        try:
            import corner
            import matplotlib.pyplot as plt
        except ImportError:
            raise RuntimeError("corner and matplotlib required for plotting")
            
        fig = plt.figure(figsize=figsize)
        
        # User requested ONLY contours at 68, 95, 99
        # No datapoints, no density.
        levels = (0.68, 0.95, 0.99)
        
        # Plot original samples (True Posterior) in Black
        corner.corner(
            self.dataset, 
            labels=self.param_names, 
            color="black", 
            fig=fig, 
            plot_datapoints=False,
            plot_density=False,
            plot_contours=True, # Explicitly true
            no_fill_contours=True,
            range=[0.995] * self.dataset.shape[1],
            levels=levels,
            contour_kwargs={"linestyles": "-"},
            hist_kwargs={"density": True},
            **corner_kwargs
        )
        
        # Resample from KDE (Red)
        if self.use_kdepy:
             # Just plot the original data for now to avoid crashes
             pass
        else:
            kde_samples = self.kde.resample(size=self.dataset.shape[0]).T
            corner.corner(
                kde_samples, 
                fig=fig, 
                color="red", 
                # weights? No, samples are representative
                plot_datapoints=False,
                plot_density=False,
                plot_contours=True,
                no_fill_contours=True,
                range=[0.995] * kde_samples.shape[1],
                levels=levels,
                contour_kwargs={"linestyles": "--"},
                hist_kwargs={"density": True},
            )
        
        return fig


class JointCloudyComponentFitter:
    """Infers Cloudy parameters for multiple components simultaneously.

    By default a joint KDE/GMM likelihood is used; this can be disabled via
    ``FitterConfig.use_kde_likelihood=False`` to run with priors + limits only.
    """

    def __init__(
        self,
        grids: Sequence[CloudyGridInterpolator],
        kde_obs: Optional[KDEObservation],
        component_ids: Sequence[int],
        ions_per_component: Sequence[str], # e.g. ["N_HI", "N_CIII"] for each component, or list of lists
        config: FitterConfig, 
        observations: Optional[Mapping[str, ObservationType]] = None, # For upper limits
    ):
        self.grids = list(grids)
        self.kde_obs = kde_obs
        self.component_ids = list(component_ids)
        self.config = config
        self.observations = dict(observations) if observations is not None else {}
        self.use_kde_likelihood = bool(getattr(config, "use_kde_likelihood", True))

        if self.use_kde_likelihood and self.kde_obs is None:
            raise ValueError("kde_obs must be provided when use_kde_likelihood=True")
        
        if len(self.grids) != len(self.component_ids):
            raise ValueError("Must provide one grid per component ID")
        
        # Handle ions input: can be single list (applied to all) or list of lists
        if isinstance(ions_per_component[0], str):
             self.ions_per_component = [list(ions_per_component) for _ in self.component_ids]
        else:
             self.ions_per_component = [list(x) for x in ions_per_component] # type: ignore

        # Validate/attach KDE dimensions only when KDE likelihood is enabled.
        self.kde_indices = []
        self._temp_kde_idx_for_component: Dict[int, int] = {}
        self._kinematic_kde_idx_for_component: Dict[int, Dict[str, int]] = {}
        _KINEMATIC_PARAMS = ("v", "b_other")

        if self.use_kde_likelihood:
            assert self.kde_obs is not None
            for i, comp_id in enumerate(self.component_ids):
                indices = []
                for ion in self.ions_per_component[i]:
                    expected_name = f"{ion}_{comp_id}"
                    try:
                        idx = self.kde_obs.param_names.index(expected_name)
                        indices.append(idx)
                    except ValueError:
                        raise ValueError(
                            f"Expected parameter '{expected_name}' not found in KDE observation param_names: {self.kde_obs.param_names}"
                        )
                self.kde_indices.append(indices)

            for i, comp_id in enumerate(self.component_ids):
                t_name = f"T_{comp_id}"
                if t_name in self.kde_obs.param_names:
                    self._temp_kde_idx_for_component[i] = self.kde_obs.param_names.index(t_name)
            if self._temp_kde_idx_for_component:
                logger.info(
                    "Temperature parameters detected in observation for components: %s",
                    [self.component_ids[i] for i in self._temp_kde_idx_for_component],
                )

            for i, comp_id in enumerate(self.component_ids):
                kin_map: Dict[str, int] = {}
                for kin_param in _KINEMATIC_PARAMS:
                    kin_name = f"{kin_param}_{comp_id}"
                    if kin_name in self.kde_obs.param_names:
                        kin_map[kin_param] = self.kde_obs.param_names.index(kin_name)
                if kin_map:
                    self._kinematic_kde_idx_for_component[i] = kin_map
            if self._kinematic_kde_idx_for_component:
                logger.info(
                    "Kinematic parameters detected in observation for components: %s",
                    {self.component_ids[i]: list(m.keys()) for i, m in self._kinematic_kde_idx_for_component.items()},
                )
        else:
            # Keep structure lengths aligned; no KDE dimensions are used.
            self.kde_indices = [[] for _ in self.component_ids]

        self.has_temperature_in_likelihood = len(self._temp_kde_idx_for_component) > 0
        self.has_kinematics_in_likelihood = len(self._kinematic_kde_idx_for_component) > 0

        # --- Relative Abundance Setup ---
        # Build per-component abundance configuration.  When
        # config.per_component_abundance_config is provided each component
        # gets its own reference element, free keys, and fixed map.
        # Otherwise the global config fields are shared by all components
        # (backwards-compatible).

        def _canonicalise_ratio_simple(key):
            if "_" not in key:
                raise ValueError(f"Invalid ratio key {key}")
            num, den = key.split("_", 1)
            num = _normalise_element_symbol(num)
            den = _normalise_element_symbol(den)
            return _build_ratio_key(num, den)

        def _build_abundance_maps_for_ref(ref_str, raw_bounds, raw_fixed):
            """Return (reference, ratio_to_ions, direct_bounds, direct_fixed,
            cross_ratio_constraints) for one reference element.

            Direct bounds/fixed have their denominator matching the reference
            (e.g. O_C when ref=C) and become MCMC free params or fixed values.

            Cross-ratio constraints have a different denominator (e.g. N_O, Si_O
            when ref=C).  These are stored as constraint specs and enforced as
            derived-quantity priors during sampling.

            Each cross-ratio constraint is a tuple:
                (ratio_key, numerator, denominator, is_fixed, value_or_None,
                 bounds_or_None)
            where ratio_key is the canonical "Num_Den" string.
            """
            reference = _normalise_element_symbol(ref_str)
            ratio_to_ions: Dict[str, Tuple[str, ...]] = {
                _build_ratio_key(el, reference): ions
                for el, ions in ELEMENT_TO_IONS.items()
                if el != reference
            }
            supported = tuple(ratio_to_ions.keys())

            ab_bounds: Dict[str, Tuple[float, float]] = {}
            cross_constraints: List[Tuple[str, str, str, bool, Optional[float], Optional[Tuple[float, float]]]] = []

            for k, v in raw_bounds.items():
                canon = _canonicalise_ratio_simple(k)
                num_str, den_str = canon.split("_", 1)
                num_el = _normalise_element_symbol(num_str)
                den_el = _normalise_element_symbol(den_str)
                if den_el == reference:
                    # Direct bound — becomes an MCMC free parameter
                    ab_bounds[canon] = v
                else:
                    # Cross-ratio bound — becomes a derived constraint with bounds
                    cross_constraints.append(
                        (canon, num_el, den_el, False, None, tuple(v))
                    )

            ab_fixed: Dict[str, float] = {}
            for k, v in raw_fixed.items():
                if "_" not in k:
                    raise ValueError(f"Invalid ratio key {k}")
                num, den = k.split("_", 1)
                num = _normalise_element_symbol(num)
                den = _normalise_element_symbol(den)
                if den == reference:
                    ab_fixed[_build_ratio_key(num, den)] = float(v)
                else:
                    # Cross-ratio fixed — becomes a derived constraint (hard)
                    canon = _build_ratio_key(num, den)
                    cross_constraints.append(
                        (canon, num, den, True, float(v), None)
                    )

            for key in supported:
                if key not in ab_bounds and key not in ab_fixed:
                    ab_fixed[key] = 0.0

            return reference, ratio_to_ions, supported, ab_bounds, ab_fixed, cross_constraints

        n_comp = len(self.component_ids)
        use_per_comp = (config.per_component_abundance_config is not None)

        if use_per_comp:
            if len(config.per_component_abundance_config) != n_comp:
                raise ValueError(
                    f"per_component_abundance_config has {len(config.per_component_abundance_config)} "
                    f"entries but {n_comp} components were given"
                )

        # Lists indexed by component
        self._per_comp_abundance_reference: List[str] = []
        self._per_comp_ratio_to_ions: List[Dict[str, Tuple[str, ...]]] = []
        self._per_comp_abundance_fixed: List[Dict[str, float]] = []
        _per_comp_abundance_bounds: List[Dict[str, Tuple[float, float]]] = []
        # Cross-ratio constraints per component.  Each entry is a list of
        # tuples: (ratio_key, numerator, denominator, is_fixed, value, bounds)
        self._per_comp_abundance_constraints: List[
            List[Tuple[str, str, str, bool, Optional[float], Optional[Tuple[float, float]]]]
        ] = []

        if use_per_comp:
            for ci in range(n_comp):
                comp_cfg = config.per_component_abundance_config[ci]
                ref_str, raw_b, raw_f = config.normalised_relative_abundances_for(comp_cfg)
                ref, r2i, _, ab_b, ab_f, cross_c = _build_abundance_maps_for_ref(ref_str, raw_b, raw_f)
                self._per_comp_abundance_reference.append(ref)
                self._per_comp_ratio_to_ions.append(r2i)
                self._per_comp_abundance_fixed.append(ab_f)
                _per_comp_abundance_bounds.append(ab_b)
                self._per_comp_abundance_constraints.append(cross_c)
        else:
            global_ref = config.abundance_reference_element or "O"
            raw_bounds_g, raw_fixed_g = config.normalised_relative_abundances()
            ref, r2i, _, ab_b, ab_f, cross_c = _build_abundance_maps_for_ref(
                global_ref, raw_bounds_g, raw_fixed_g,
            )
            for _ in range(n_comp):
                self._per_comp_abundance_reference.append(ref)
                self._per_comp_ratio_to_ions.append(r2i)
                self._per_comp_abundance_fixed.append(ab_f)
                _per_comp_abundance_bounds.append(ab_b)
                self._per_comp_abundance_constraints.append(list(cross_c))

        # Backwards-compatible aliases (used by some plotting/debug helpers)
        self.abundance_reference = self._per_comp_abundance_reference[0]
        self.relative_ratio_to_ions = self._per_comp_ratio_to_ions[0]
        self.relative_abundance_fixed = self._per_comp_abundance_fixed[0]

        # TD grid detection (per component)
        self._td_grid_flags: List[bool] = [
            getattr(grid, "is_td_grid", False) for grid in self.grids
        ]
        if any(self._td_grid_flags):
            logger.info(
                "TD grids detected for components: %s",
                [self.component_ids[i] for i, f in enumerate(self._td_grid_flags) if f],
            )

        _DEFAULT_TD_NH_BOUNDS = (17.0, 23.0)

        # Build per-component temperature mode (auto-detect from grid axes)
        self._per_comp_temperature_mode: List[str] = []
        for i, grid in enumerate(self.grids):
            if config.per_component_temperature_mode is not None:
                mode = config.per_component_temperature_mode[i]
            else:
                mode = "auto"

            if mode == "auto":
                if self._td_grid_flags[i]:
                    mode = "free" if "T" in grid.parameter_bounds() else config.temperature_mode
                elif "T" in grid.parameter_bounds():
                    mode = "free"
                else:
                    mode = config.temperature_mode

            self._per_comp_temperature_mode.append(mode)
        logger.info(
            "Per-component temperature modes: %s",
            {self.component_ids[i]: m for i, m in enumerate(self._per_comp_temperature_mode)},
        )

        # Build Parameter Space
        self.param_slices = []
        self.param_names = []
        self.param_bounds = []
        self.component_param_templates = []
        
        start = 0
        for i, grid in enumerate(self.grids):
            bounds = grid.parameter_bounds()
            is_td = self._td_grid_flags[i]
            p_order = ["Z", "n_H"]

            if "N_H" in bounds:
                p_order.append("N_H")
            elif "NHI" in bounds:
                p_order.append("NHI")
            elif is_td:
                p_order.append("N_H")

            if is_td:
                if "T_init" in bounds:
                    p_order.append("T_init")
                if "T" in bounds:
                    p_order.append("T")
            elif self._per_comp_temperature_mode[i] == "free":
                p_order.append("T")
            
            # Add Relative Abundances (Free Params) -- per-component keys
            ab_bounds_i = _per_comp_abundance_bounds[i]
            sorted_abund_keys = sorted(ab_bounds_i.keys())
            for key in sorted_abund_keys:
                 p_order.append(key)

            # Add kinematic pass-through parameters when present in the GMM
            if i in self._kinematic_kde_idx_for_component:
                for kin_param in sorted(self._kinematic_kde_idx_for_component[i].keys()):
                    p_order.append(kin_param)
            
            self.component_param_templates.append(p_order)

            for p in p_order:
                if p in config.parameter_bounds_override:
                    self.param_bounds.append(config.parameter_bounds_override[p])
                elif p in ab_bounds_i:
                    self.param_bounds.append(ab_bounds_i[p])
                elif p in _KINEMATIC_PARAMS:
                    kin_name = f"{p}_{self.component_ids[i]}"
                    kin_kde_idx = self.kde_obs.param_names.index(kin_name)
                    kin_data = self.kde_obs.dataset[:, kin_kde_idx]
                    kin_lo, kin_hi = float(np.min(kin_data)), float(np.max(kin_data))
                    margin = max(0.1 * (kin_hi - kin_lo), 1.0)
                    self.param_bounds.append((kin_lo - margin, kin_hi + margin))
                elif is_td and p == "N_H" and "N_H" not in bounds:
                    self.param_bounds.append(
                        config.td_n_h_bounds if config.td_n_h_bounds is not None
                        else _DEFAULT_TD_NH_BOUNDS
                    )
                else:
                    self.param_bounds.append(bounds[p])
                self.param_names.append(f"{p}_{self.component_ids[i]}")
            
            n_params = len(p_order)
            self.param_slices.append(slice(start, start + n_params))
            start += n_params

        self.ndim = len(self.param_names)

        # ---- Cross-component constraints (inequality & equality) --------
        self._parsed_inequality_constraints: List[Tuple[int, int, Any]] = []
        self._equality_mappings: List[Tuple[int, int]] = []  # (source_full, target_full)
        self._full_param_names = list(self.param_names)
        self._full_param_bounds = list(self.param_bounds)
        self._full_ndim = self.ndim

        if config.cross_component_constraints:
            _ops = {'>': lambda a, b: a > b, '<': lambda a, b: a < b}
            name_to_idx = {n: i for i, n in enumerate(self.param_names)}

            ineq_list: List[Tuple[int, int, Any]] = []
            eq_list: List[Tuple[int, int]] = []

            for expr in config.cross_component_constraints:
                expr_clean = expr.replace(' ', '')
                if '==' in expr_clean:
                    a_name, b_name = expr_clean.split('==')
                    ia = name_to_idx.get(a_name)
                    ib = name_to_idx.get(b_name)
                    if ia is None or ib is None:
                        raise ValueError(
                            f"Equality constraint '{expr}' references unknown "
                            f"parameter(s).  Available: {list(name_to_idx)}"
                        )
                    eq_list.append((ia, ib))
                else:
                    matched = False
                    for op_str, op_fn in _ops.items():
                        if op_str in expr_clean:
                            a_name, b_name = expr_clean.split(op_str)
                            ia = name_to_idx.get(a_name)
                            ib = name_to_idx.get(b_name)
                            if ia is None or ib is None:
                                raise ValueError(
                                    f"Inequality constraint '{expr}' references "
                                    f"unknown parameter(s).  Available: {list(name_to_idx)}"
                                )
                            ineq_list.append((ia, ib, op_fn))
                            matched = True
                            break
                    if not matched:
                        raise ValueError(
                            f"Cannot parse cross-component constraint '{expr}'. "
                            f"Supported operators: ==, >, <"
                        )

            # --- Process equality constraints: dimension reduction ----------
            # For each (source, target) pair we keep source and drop target.
            # ``_equality_mappings`` is stored in *full* indices.
            # ``_dropped_indices`` tracks which full-space indices are removed.
            dropped: set = set()
            for src_full, tgt_full in eq_list:
                if tgt_full in dropped:
                    raise ValueError(
                        f"Parameter '{self._full_param_names[tgt_full]}' is "
                        f"already tied by another equality constraint."
                    )
                if src_full in dropped:
                    raise ValueError(
                        f"Parameter '{self._full_param_names[src_full]}' is "
                        f"already dropped by another equality constraint."
                    )
                self._equality_mappings.append((src_full, tgt_full))
                dropped.add(tgt_full)

            if dropped:
                kept = [i for i in range(self._full_ndim) if i not in dropped]
                full_to_reduced = {full_i: red_i for red_i, full_i in enumerate(kept)}

                # Reduced-space names/bounds/ndim used by MCMC
                self._mcmc_param_names = [self._full_param_names[i] for i in kept]
                self._mcmc_param_bounds = [self._full_param_bounds[i] for i in kept]
                self._mcmc_ndim = len(self._mcmc_param_names)

                # Store the expansion lookup: for each full-space index, either
                # a reduced-space index (int) or a source reduced-space index
                # via an equality mapping.
                self._expand_lookup: List[int] = []
                eq_src_map = {tgt: src for src, tgt in self._equality_mappings}
                for fi in range(self._full_ndim):
                    if fi in full_to_reduced:
                        self._expand_lookup.append(full_to_reduced[fi])
                    else:
                        src_full = eq_src_map[fi]
                        self._expand_lookup.append(full_to_reduced[src_full])

                logger.info(
                    "Equality constraints dropped %d parameter(s); "
                    "reduced ndim %d → %d.  Dropped: %s",
                    len(dropped), self._full_ndim, self._mcmc_ndim,
                    [self._full_param_names[i] for i in sorted(dropped)],
                )
            else:
                self._mcmc_param_names = list(self.param_names)
                self._mcmc_param_bounds = list(self.param_bounds)
                self._mcmc_ndim = self.ndim
                self._expand_lookup = list(range(self._full_ndim))

            # --- Remap inequality indices to reduced space -----------------
            full_to_reduced_ineq = {fi: ri for ri, fi in enumerate(
                [i for i in range(self._full_ndim) if i not in dropped]
            )} if dropped else {i: i for i in range(self._full_ndim)}
            # Equality-mapped targets share the source's reduced index
            eq_src_map_ineq = {tgt: src for src, tgt in self._equality_mappings}
            def _to_reduced(fi: int) -> int:
                if fi in full_to_reduced_ineq:
                    return full_to_reduced_ineq[fi]
                return full_to_reduced_ineq[eq_src_map_ineq[fi]]

            self._parsed_inequality_constraints = [
                (_to_reduced(ia), _to_reduced(ib), fn) for ia, ib, fn in ineq_list
            ]

            if self._parsed_inequality_constraints:
                logger.info(
                    "Inequality constraints: %s",
                    [
                        f"{self._full_param_names[ia]} op {self._full_param_names[ib]}"
                        for ia, ib, _ in ineq_list
                    ],
                )
        else:
            self._mcmc_param_names = list(self.param_names)
            self._mcmc_param_bounds = list(self.param_bounds)
            self._mcmc_ndim = self.ndim
            self._expand_lookup = list(range(self._full_ndim))

        self.sampler = None
        self.samples = None

        # Pre-build Voigt temperature z-score observations (one per component)
        self._voigt_temp_obs: List[Optional[ZScoreObservation]] = []
        if config.use_voigt_temperature_prior and config.voigt_temperature_samples is not None:
            zscore_cfg = ZScoreGridConfig()
            for ci, t_samples in enumerate(config.voigt_temperature_samples):
                if t_samples is not None and len(t_samples) > 0:
                    self._voigt_temp_obs.append(
                        ZScoreObservation.from_samples(t_samples, zscore_cfg)
                    )
                else:
                    self._voigt_temp_obs.append(None)
        else:
            self._voigt_temp_obs = [None] * len(self.component_ids)

    # -----------------------------------------------------------------
    # Theta expansion (equality constraints)
    # -----------------------------------------------------------------
    def _expand_theta(self, theta: np.ndarray) -> np.ndarray:
        """Expand a reduced-space theta vector back to the full parameter
        space by duplicating values for equality-tied parameters.

        When there are no equality constraints this is a zero-copy identity.
        """
        if self._mcmc_ndim == self._full_ndim:
            return theta
        return theta[self._expand_lookup]

    def _expand_samples(self, samples: np.ndarray) -> np.ndarray:
        """Row-wise expansion of an (N, ndim_reduced) array to (N, ndim_full)."""
        if self._mcmc_ndim == self._full_ndim:
            return samples
        return samples[:, self._expand_lookup]

    def _reduce_theta(self, theta_full: np.ndarray) -> np.ndarray:
        """Project a full-space vector to the reduced MCMC space by keeping
        only non-dropped indices.  Inverse of ``_expand_theta``."""
        if self._mcmc_ndim == self._full_ndim:
            return theta_full
        dropped = {tgt for _, tgt in self._equality_mappings}
        return np.array([theta_full[i] for i in range(self._full_ndim) if i not in dropped])

    def _apply_abundance_offsets_for_component(
        self,
        comp_index: int,
        model: Dict[str, float],
        params: Dict[str, float],
    ) -> None:
        """Apply relative-abundance offsets to *model* using
        the per-component abundance maps."""
        ratio_to_ions = self._per_comp_ratio_to_ions[comp_index]
        ab_fixed = self._per_comp_abundance_fixed[comp_index]
        for key, ions_list in ratio_to_ions.items():
            offset = params.get(key, ab_fixed.get(key, 0.0))
            for ion in ions_list:
                if ion in model:
                    model[ion] += offset

    def _apply_abundance_offsets_batch(
        self,
        comp_index: int,
        ion: str,
        col_arr: np.ndarray,
        comp_samples: np.ndarray,
        p_order: List[str],
    ) -> np.ndarray:
        """Apply abundance offsets to a batch array of one ion's predictions.

        Works on a 1-D array of log10 column densities (one per posterior
        sample) instead of a single-point dict.  Returns the modified array.
        """
        ratio_to_ions = self._per_comp_ratio_to_ions[comp_index]
        ab_fixed = self._per_comp_abundance_fixed[comp_index]
        for key, ions_list in ratio_to_ions.items():
            if ion in ions_list:
                if key in p_order:
                    col_arr = col_arr + comp_samples[:, p_order.index(key)]
                else:
                    col_arr = col_arr + ab_fixed.get(key, 0.0)
        return col_arr

    def _z_label_for_component(self, comp_index: int) -> str:
        """Return a human-readable label for the metallicity parameter of a
        given component (e.g. ``'log C/H'`` when the reference element is C)."""
        ref = self._per_comp_abundance_reference[comp_index]
        return f"log {ref}/H"

    def _prettify_param_label(self, full_name: str) -> str:
        """Turn an internal parameter name like ``Z_0`` or ``O_C_1`` into a
        publication-quality label such as ``log O/H_0`` or ``log O/C_1``.

        Falls back to the raw name when no mapping applies.
        """
        parts = full_name.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            return full_name
        base_name, comp_id_str = parts
        comp_id = int(comp_id_str)

        try:
            ci = self.component_ids.index(comp_id)
        except ValueError:
            return full_name

        if base_name == "Z":
            ref = self._per_comp_abundance_reference[ci]
            return f"log {ref}/H$_{{{comp_id}}}$"
        # Abundance ratio key: e.g. "O_C" → "log O/C"
        if base_name in self._per_comp_ratio_to_ions[ci]:
            num, den = base_name.split("_", 1)
            return f"log {num}/{den}$_{{{comp_id}}}$"
        return full_name

    def _constrained_elements(self) -> List[str]:
        """Return a sorted list of element symbols that have at least one
        constrained ion (in the GMM or in the upper/lower-limit observations)."""
        constrained: set = set()
        for ion_list in self.ions_per_component:
            for ion in ion_list:
                el = ION_TO_ELEMENT.get(ion)
                if el is not None and el != "H":
                    constrained.add(el)
        for ion in self.observations:
            el = ION_TO_ELEMENT.get(ion)
            if el is not None and el != "H":
                constrained.add(el)
        return sorted(constrained)

    def _compute_element_mass_log(
        self,
        comp_index: int,
        params: Dict[str, float],
        element: str,
    ) -> Optional[float]:
        """Compute log10(M_element / M_sun) for one component.

        Uses Z, relative abundances, n_H, and N_H from *params* together
        with the per-component abundance reference element.

        Returns ``None`` when the required grid parameters are missing or
        the element has no solar abundance entry.
        """
        log_n_H = params.get("n_H")
        log_N_H = params.get("N_H")
        if log_N_H is None:
            log_N_H = params.get("NHI")
        if log_n_H is None or log_N_H is None:
            return None
        if element not in SOLAR_LOG_ABUNDANCE or element not in LOG10_ATOMIC_MASS_G:
            return None

        Z = params.get("Z", 0.0)
        ref = self._per_comp_abundance_reference[comp_index]

        if element == ref:
            log_X_over_H = SOLAR_LOG_ABUNDANCE[element] + Z
        else:
            ratio_key = _build_ratio_key(element, ref)
            ab_fixed = self._per_comp_abundance_fixed[comp_index]
            offset = params.get(ratio_key, ab_fixed.get(ratio_key, 0.0))
            log_X_over_H = SOLAR_LOG_ABUNDANCE[element] + Z + offset

        log_M_g = (
            log_X_over_H
            + LOG10_ATOMIC_MASS_G[element]
            + 3.0 * log_N_H
            - 2.0 * log_n_H
        )
        return log_M_g - LOG10_SOLAR_MASS_G

    def _compute_cross_ratio_value(
        self,
        comp_index: int,
        params: Dict[str, float],
        numerator: str,
        denominator: str,
    ) -> Optional[float]:
        """Compute a derived element ratio log10(X/Y) from the reference-basis.

        For example, if reference = C and we want N/O:
            N/O = (N/C) - (O/C)  (in log solar-relative space)
        where N/C and O/C are read from ``params`` or from fixed values.

        Returns ``None`` when a required parameter is unavailable.
        """
        ref = self._per_comp_abundance_reference[comp_index]
        ab_fixed = self._per_comp_abundance_fixed[comp_index]

        # Get log(numerator / ref) offset
        if numerator == ref:
            num_offset = 0.0
        else:
            num_key = _build_ratio_key(numerator, ref)
            num_offset = params.get(num_key, ab_fixed.get(num_key))
            if num_offset is None:
                return None

        # Get log(denominator / ref) offset
        if denominator == ref:
            den_offset = 0.0
        else:
            den_key = _build_ratio_key(denominator, ref)
            den_offset = params.get(den_key, ab_fixed.get(den_key))
            if den_offset is None:
                return None

        # log(X/Y)_solar_relative = log(X/ref) - log(Y/ref)
        #   but we also need the solar part:
        #   log(X/Y) = log(X/H) - log(Y/H)
        #            = [solar(X) + Z + num_offset] - [solar(Y) + Z + den_offset]
        #            = solar(X) - solar(Y) + num_offset - den_offset
        # However, the user's bounds are defined in the same solar-relative
        # convention as the free_abundances (offsets from solar ratios in the
        # Cloudy grid).  So the derived value is simply:
        return float(num_offset - den_offset)

    def _has_abundance_constraints(self) -> bool:
        """Return True if any component has cross-ratio abundance constraints."""
        return any(len(cc) > 0 for cc in self._per_comp_abundance_constraints)

    def _log_probability(self, theta: np.ndarray) -> float:
        # 1. Box priors (MCMC / reduced space)
        for val, (lo, hi) in zip(theta, self._mcmc_param_bounds):
            if val < lo or val > hi:
                return -np.inf

        # 1b. Cross-component inequality constraints (MCMC / reduced space)
        for idx_a, idx_b, op_fn in self._parsed_inequality_constraints:
            if not op_fn(theta[idx_a], theta[idx_b]):
                return -np.inf

        # 1c. Expand to full parameter space (no-op when no equalities)
        theta_full = self._expand_theta(theta)
        
        # 2. Likelihood
        predicted_vector = None
        if self.use_kde_likelihood:
            assert self.kde_obs is not None
            predicted_vector = np.zeros(len(self.kde_obs.param_names))
        
        # We iterate through components, predict their ions, and place them in the vector
        # using pre-computed indices.
        
        # Initialize total linear model for upper limits
        total_linear_model = {k: 0.0 for k in self.observations}
        
        for i, sl in enumerate(self.param_slices):
            comp_theta = theta_full[sl]
            # Unpack theta to dict
            # We know the order is Z, nH, [NH], [T]
            # This is brittle, ideally we reuse the single fitter logic or store metadata
            grid = self.grids[i]
            p_order = self.component_param_templates[i]
            
            params = {}
            if "z" in grid.axis_names:
                params["z"] = self.config.redshift
            
            for idx, name in enumerate(p_order):
                params[name] = comp_theta[idx]

            # DEBUG: Print params for first component occasionally
            # if i == 0 and np.random.rand() < 0.0001:
                # print(f"DEBUG: Params for comp 0 (log_prob): {list(params.keys())}")
                # if "O_C" in params:
                    # print(f"DEBUG: O_C value: {params['O_C']}")

            
            # Evaluate grid
            model = grid.evaluate(params)
            
            # Apply Abundance Offsets (per-component maps)
            self._apply_abundance_offsets_for_component(i, model, params)

            # TD grids: scale ion fractions by N_H and inject T_cloudy
            if self._td_grid_flags[i]:
                n_h_scale = params.get("N_H", 0.0)
                for _ion_key in list(model.keys()):
                    if _ion_key.startswith("N_"):
                        model[_ion_key] += n_h_scale
                t_current = params.get("T")
                if t_current is not None:
                    model["T_cloudy"] = t_current

            # Fill KDE vector
            # The KDE expects predicted log column densities for specific component-ions
            if self.use_kde_likelihood:
                assert predicted_vector is not None
                for ion_name, kde_idx in zip(self.ions_per_component[i], self.kde_indices[i]):
                    val = model.get(ion_name)
                    if val is None or not np.isfinite(val):
                        return -np.inf
                    predicted_vector[kde_idx] = val

            # Fill T_cloudy in observation vector when temperature is in the GMM
            if self.use_kde_likelihood and self.has_temperature_in_likelihood and i in self._temp_kde_idx_for_component:
                t_val = model.get("T_cloudy")
                if t_val is None:
                    t_val = model.get("T")  # fallback to free-T axis
                if t_val is None or not np.isfinite(t_val):
                    return -np.inf
                predicted_vector[self._temp_kde_idx_for_component[i]] = t_val

            # Fill kinematic pass-through parameters (v, b_other).
            # These are NOT predicted by the Cloudy grid — the walker value
            # is placed directly so the GMM scores the joint distribution.
            if self.use_kde_likelihood and self.has_kinematics_in_likelihood and i in self._kinematic_kde_idx_for_component:
                for kin_param, kde_idx in self._kinematic_kde_idx_for_component[i].items():
                    kin_val = params.get(kin_param)
                    if kin_val is None or not np.isfinite(kin_val):
                        return -np.inf
                    predicted_vector[kde_idx] = kin_val

            # Accumulate totals for external observations (upper limits)
            for ion in total_linear_model:
                if ion in model:
                    total_linear_model[ion] += 10**model[ion]
        
        # 3. KDE Likelihood (optional)
        if self.use_kde_likelihood:
            assert self.kde_obs is not None and predicted_vector is not None
            lp = self.kde_obs.log_prob(predicted_vector)
            if not np.isfinite(lp):
                return -np.inf
        else:
            lp = 0.0
             
        # 4. Observation Likelihood (Upper Limits on Total)
        if self.observations:
            # Convert linear sums back to log10
            total_log_model = {}
            for ion, linear_val in total_linear_model.items():
                if linear_val > 0:
                    total_log_model[ion] = np.log10(linear_val)
                else:
                    total_log_model[ion] = -np.inf 

            lp += log_likelihood(total_log_model, self.observations)

        # 5. Derived-quantity priors (cloud length, mass, Jeans, temperature)
        dq_lp = self._derived_quantity_log_prior(theta_full)
        if not np.isfinite(dq_lp):
            return -np.inf
        lp += dq_lp

        return lp

    def _derived_quantity_log_prior(self, theta_full: np.ndarray) -> float:
        """Evaluate priors on derived physical quantities (cloud size, mass, Jeans, T, element mass).

        ``theta_full`` must be in the **full** (unexpanded) parameter space.
        """
        cfg = self.config
        _has_cross_constraints = self._has_abundance_constraints()
        has_dq_priors = (
            cfg.cloud_length_bounds is not None
            or cfg.cloud_mass_bounds is not None
            or cfg.total_length_bounds is not None
            or cfg.total_mass_bounds is not None
            or cfg.jeans_length_prior
            or cfg.use_voigt_temperature_prior
            or cfg.element_mass_bounds is not None
            or cfg.total_column_density_bounds is not None
            or _has_cross_constraints
        )
        if not has_dq_priors:
            return 0.0

        lp = 0.0
        component_lengths_linear = []
        component_masses_linear = []

        # Accumulate per-element linear masses across components for total prior
        _elem_mass_linear: Dict[str, List[float]] = {}
        _need_elem_mass = cfg.element_mass_bounds is not None

        # Accumulate total ion columns across components for total-column prior.
        _need_total_columns = cfg.total_column_density_bounds is not None
        _total_column_linear: Dict[str, float] = {}
        if _need_total_columns:
            _total_column_linear = {
                ion: 0.0 for ion in cfg.total_column_density_bounds.keys()
            }

        for i, sl in enumerate(self.param_slices):
            comp_theta = theta_full[sl]
            grid = self.grids[i]
            p_order = self.component_param_templates[i]

            params = {}
            if "z" in grid.axis_names:
                params["z"] = cfg.redshift
            for idx, name in enumerate(p_order):
                params[name] = comp_theta[idx]

            log_n_H = params.get("n_H")
            log_N_H = params.get("N_H")
            if log_N_H is None:
                log_N_H = params.get("NHI")

            # --- Cloud length & mass ---
            if log_N_H is not None and log_n_H is not None:
                log_L_cm = log_N_H - log_n_H
                log_L_kpc = log_L_cm - LOG10_CM_PER_KPC
                log_M_g = 3.0 * log_N_H - 2.0 * log_n_H + LOG10_PROTON_MASS_G
                log_M_solar = log_M_g - LOG10_SOLAR_MASS_G

                if cfg.cloud_length_bounds is not None:
                    lo, hi = cfg.cloud_length_bounds
                    if log_L_kpc < lo or log_L_kpc > hi:
                        return -np.inf

                if cfg.cloud_mass_bounds is not None:
                    lo, hi = cfg.cloud_mass_bounds
                    if log_M_solar < lo or log_M_solar > hi:
                        return -np.inf

                component_lengths_linear.append(10.0 ** log_L_kpc)
                component_masses_linear.append(10.0 ** log_M_solar)

            # --- Element masses ---
            if _need_elem_mass:
                for elem in cfg.element_mass_bounds:
                    log_me = self._compute_element_mass_log(i, params, elem)
                    if log_me is not None:
                        _elem_mass_linear.setdefault(elem, []).append(10.0 ** log_me)

            # --- Jeans length enforcement & temperature prior ---
            need_zscore_T = (
                cfg.use_voigt_temperature_prior
                and not self.has_temperature_in_likelihood
            )
            if cfg.jeans_length_prior or need_zscore_T or _need_total_columns:
                model = grid.evaluate(params)
                self._apply_abundance_offsets_for_component(i, model, params)

                if _need_total_columns:
                    for ion in _total_column_linear:
                        val = model.get(ion)
                        if val is not None and np.isfinite(val):
                            _total_column_linear[ion] += 10.0 ** val

                log_T = model.get("T_cloudy")
                if log_T is None:
                    log_T = params.get("T")

                if log_T is not None and np.isfinite(log_T):
                    if cfg.jeans_length_prior and log_N_H is not None and log_n_H is not None:
                        log_L_J_kpc = 0.5 * (_JEANS_A_CONST + log_T) - 0.5 * log_n_H - LOG10_CM_PER_KPC
                        if log_L_kpc > log_L_J_kpc:
                            return -np.inf

                    if need_zscore_T and i < len(self._voigt_temp_obs):
                        t_obs = self._voigt_temp_obs[i]
                        if t_obs is not None:
                            lp += _loglike_zscore(log_T, t_obs)

        # --- Summed priors across components ---
        if cfg.total_length_bounds is not None and component_lengths_linear:
            total_L = sum(component_lengths_linear)
            if total_L > 0:
                log_total_L = np.log10(total_L)
                lo, hi = cfg.total_length_bounds
                if log_total_L < lo or log_total_L > hi:
                    return -np.inf

        if cfg.total_mass_bounds is not None and component_masses_linear:
            total_M = sum(component_masses_linear)
            if total_M > 0:
                log_total_M = np.log10(total_M)
                lo, hi = cfg.total_mass_bounds
                if log_total_M < lo or log_total_M > hi:
                    return -np.inf

        # --- Total element mass priors ---
        if _need_elem_mass:
            for elem, (lo, hi) in cfg.element_mass_bounds.items():
                parts = _elem_mass_linear.get(elem)
                if parts:
                    total = sum(parts)
                    if total > 0:
                        log_total = np.log10(total)
                        if log_total < lo or log_total > hi:
                            return -np.inf

        # --- Total ion column-density priors ---
        if _need_total_columns:
            for ion, (lo, hi) in cfg.total_column_density_bounds.items():
                total_linear = _total_column_linear.get(ion, 0.0)
                if total_linear <= 0.0:
                    return -np.inf
                log_total = np.log10(total_linear)
                # Apply bounds as soft Observation-style lower/upper limits
                # with sigma=0.01 for stable sampling near the boundaries.
                lp += _loglike_lower_limit(
                    log_total,
                    Observation(value=float(lo), sigma=DEFAULT_SOFT_LIMIT_SIGMA, is_lower_limit=True),
                )
                lp += _loglike_upper_limit(
                    log_total,
                    Observation(value=float(hi), sigma=DEFAULT_SOFT_LIMIT_SIGMA, is_upper_limit=True),
                )

        # --- Cross-ratio abundance constraints ---
        if _has_cross_constraints:
            for i, sl in enumerate(self.param_slices):
                constraints = self._per_comp_abundance_constraints[i]
                if not constraints:
                    continue
                comp_theta = theta_full[sl]
                p_order = self.component_param_templates[i]
                params = {}
                for idx, name in enumerate(p_order):
                    params[name] = comp_theta[idx]

                for ratio_key, num_el, den_el, is_fixed, fixed_val, bounds_val in constraints:
                    derived = self._compute_cross_ratio_value(i, params, num_el, den_el)
                    if derived is None:
                        continue
                    if is_fixed:
                        # Hard constraint: reject if derived value doesn't
                        # match the target (with a tiny tolerance).
                        if abs(derived - fixed_val) > 0.01:
                            return -np.inf
                    else:
                        # Bounded constraint: apply soft prior.
                        lo, hi = bounds_val
                        lp += _loglike_lower_limit(
                            derived,
                            Observation(value=float(lo), sigma=DEFAULT_SOFT_LIMIT_SIGMA, is_lower_limit=True),
                        )
                        lp += _loglike_upper_limit(
                            derived,
                            Observation(value=float(hi), sigma=DEFAULT_SOFT_LIMIT_SIGMA, is_upper_limit=True),
                        )

        return lp

    def _find_smart_initial_guess(self, n_trials: int = 10000):
        """
        Find best grid parameters by testing candidates from the input dataset.
        Instead of using the geometric mean (which might be low probability),
        we pick random samples from the 'True' posterior (dataset), map them 
        to the closest grid points, and select the one with the highest GMM likelihood.
        """
        if not self.use_kde_likelihood or self.kde_obs is None:
            raise RuntimeError("Smart KDE-based initialization requires use_kde_likelihood=True and kde_obs.")

        # 1. Pick a subset of samples from the input chain ("True" posterior)
        # These represent valid, high-probability column density configurations
        n_total = self.kde_obs.dataset.shape[0]
        n_trials = min(n_trials, n_total)
        indices = np.random.choice(n_total, n_trials, replace=False)
        candidate_vectors = self.kde_obs.dataset[indices, :] # (n_trials, n_dims)
        
        best_guess = None
        best_lp = -np.inf
        
        print(f"Scanning {n_trials} candidates from input chain for best initialization...")
        
        for i in range(n_trials):
            target_cols = candidate_vectors[i, :]
            
            # Construct parameter guess for this target vector
            current_guess = []
            
            valid_mapping = True
            for comp_idx, grid in enumerate(self.grids):
                # Identify which elements of target_cols belong to this component
                # and which grid axis they correspond to.
                
                # We need to map target_cols (values) to grid.values
                # grid.values shape: (d1, d2, ..., n_ions)
                # We need indices of grid ions in the joint vector
                
                # Get sub-vector for this component
                comp_ion_names = self.ions_per_component[comp_idx]
                comp_kde_indices = self.kde_indices[comp_idx]
                
                comp_target_vals = target_cols[comp_kde_indices]
                
                # Find grid point closest to these values
                # We only care about ions that exist in the grid
                grid_ion_indices = []
                target_sub = []
                
                for ion_name, val in zip(comp_ion_names, comp_target_vals):
                    if ion_name in grid.ion_order:
                        grid_ion_indices.append(grid.ion_order.index(ion_name))
                        target_sub.append(val)
                
                if not target_sub:
                    # Fallback if no ions match (unlikely)
                    current_guess.extend([0.0] * len(self.component_param_templates[comp_idx]))
                    continue

                target_sub = np.array(target_sub)
                
                # Extract relevant columns from grid
                # Flatten grid: (N_grid_points, N_all_ions)
                flat_grid = grid.values.reshape(-1, len(grid.ion_order))
                # Select only valid ions
                flat_grid_sub = flat_grid[:, grid_ion_indices]
                
                # Euclidean distance in log-N space
                diff = flat_grid_sub - target_sub
                dist = np.sum(diff**2, axis=1)
                best_grid_idx = np.argmin(dist)
                
                # Retrieve parameters for this grid point
                grid_pts_shape = tuple(len(p) for p in grid.points)
                unraveled = np.unravel_index(best_grid_idx, grid_pts_shape)
                
                # Build component parameters (Z, nH, etc.)
                # Grid params
                grid_params = {}
                for dim, array_idx in enumerate(unraveled):
                    axis = grid.axis_names[dim]
                    grid_params[axis] = grid.points[dim][array_idx]
                
                # Config params (T, abundances, kinematics)
                p_order = self.component_param_templates[comp_idx]
                for p_name in p_order:
                    if p_name in grid_params:
                        current_guess.append(grid_params[p_name])
                    elif p_name == "T" and self._per_comp_temperature_mode[comp_idx] == "free":
                        # If T is not in grid, use a default or try to infer?
                        # Using 5.0 (log K) as generic warm-hot guess
                        current_guess.append(5.0)
                    elif p_name in self._per_comp_abundance_fixed[comp_idx]:
                        current_guess.append(0.0) # Solar start
                    elif any(p_name in ab_b for ab_b in [self._per_comp_ratio_to_ions[comp_idx]]):
                        current_guess.append(0.0) # Solar start
                    elif p_name in ("v", "b_other"):
                        # Kinematic pass-through: initialise from the same
                        # GMM dataset sample we are matching.
                        kin_name = f"{p_name}_{self.component_ids[comp_idx]}"
                        if kin_name in self.kde_obs.param_names:
                            kin_kde_idx = self.kde_obs.param_names.index(kin_name)
                            current_guess.append(float(target_cols[kin_kde_idx]))
                        else:
                            current_guess.append(0.0)
                    else:
                        # Unknown param?
                        current_guess.append(0.0)

            # Evaluate this guess (reduce to MCMC space if equalities exist)
            guess_arr = self._reduce_theta(np.array(current_guess))
            lp = self._log_probability(guess_arr)
            
            if lp > best_lp and np.isfinite(lp):
                best_lp = lp
                best_guess = guess_arr
        
        if best_guess is None:
            raise RuntimeError("Could not find any finite initialization point after scanning dataset.")
            
        print(f"Found best start: LogProb={best_lp:.2f}")
        return best_guess



    def run_mcmc(self, nwalkers=64, nsteps=1000, burnin=500, initial_positions=None, n_cores=1, **kwargs):
        _ndim = self._mcmc_ndim
        _bounds = self._mcmc_param_bounds
        _names = self._mcmc_param_names
        center = np.array([(lo + hi) / 2.0 for (lo, hi) in _bounds], dtype=float)

        def _sample_from_prior() -> np.ndarray:
            vals = np.empty(_ndim, dtype=float)
            for di, (lo, hi) in enumerate(_bounds):
                if np.isfinite(lo) and np.isfinite(hi):
                    vals[di] = np.random.uniform(lo, hi)
                else:
                    vals[di] = np.random.normal(0.0, 1.0)
            return vals

        if initial_positions is None:
            # 1. Smart Initialization using Grid Search (KDE mode only)
            if self.use_kde_likelihood:
                print("Finding valid initial positions using grid search...")
            else:
                print("KDE likelihood disabled: initializing walkers from priors + limits only.")

            try:
                if self.use_kde_likelihood:
                    center = self._find_smart_initial_guess()
                    print(f"Best grid point: {center}")
                else:
                    raise RuntimeError("Skipping smart init because use_kde_likelihood=False")
                
                # Check if center itself is valid
                center_lp = self._log_probability(center)
                if not np.isfinite(center_lp):
                    print(f"Warning: Smart init 'center' has non-finite log-prob ({center_lp}). Searching for valid neighbor...")
                    # raise ValueError("Smart init center is invalid") -- Don't raise, try to find neighbor

                # Jitter strategy
                print(f"Initializing {nwalkers} walkers around center (jitter=0.01)...")
                # Try simple Gaussian ball first
                # But we need to ensure they are valid
                # So we draw candidates and check
                
                initial_positions = []
                attempts = 0
                max_attempts = 10000  # Increased for robust sampling
                scale = 0.01 
                
                # Identify relative abundance indices for uniform sampling
                _all_ratio_keys = set()
                for _r2i in self._per_comp_ratio_to_ions:
                    _all_ratio_keys.update(_r2i.keys())
                rel_abund_indices = []
                rel_abund_bounds_list = []
                for idx, p_name in enumerate(_names):
                    if "_" in p_name:
                        base, comp_suffix = p_name.rsplit("_", 1)
                        if base in _all_ratio_keys and comp_suffix.isdigit():
                            rel_abund_indices.append(idx)
                            rel_abund_bounds_list.append(_bounds[idx])
                
                if rel_abund_indices:
                     print(f"Initializing relative abundances (indices {rel_abund_indices}) uniformly from prior.")

                # Use a loop to fill walkers
                while len(initial_positions) < nwalkers and attempts < max_attempts:
                    proposal = center + scale * np.random.randn(_ndim)
                    
                    # Overwrite relative abundances with uniform draws from prior
                    for idx, (lo, hi) in zip(rel_abund_indices, rel_abund_bounds_list):
                        proposal[idx] = np.random.uniform(lo, hi)
                    
                    # Check validity
                    if np.isfinite(self._log_probability(proposal)):
                         initial_positions.append(proposal)
                    attempts += 1
                
                if len(initial_positions) < nwalkers:
                     print(f"Only found {len(initial_positions)} valid walkers after {attempts} attempts. Prior volume might be constrained.")
                     # Fallback: We must output *something*, but warn heavily.
                     # We fill the rest with the last valid point found, or just random tries?
                     # Better to fill with copies of valid points if we found any.
                     if len(initial_positions) > 0:
                         print("Filling remaining walkers by resampling valid ones.")
                         while len(initial_positions) < nwalkers:
                             # Resample from valid with slight jitter
                             base = initial_positions[np.random.randint(len(initial_positions))]
                             proposal = base + 1e-4 * np.random.randn(_ndim)
                             if np.isfinite(self._log_probability(proposal)):
                                 initial_positions.append(proposal)
                     else:
                         raise ValueError("Could not find ANY valid initial positions. Check priors/upper limits.")

                initial_positions = np.array(initial_positions)
                print(f"Valid walkers found: {len(initial_positions)}/{nwalkers}")

            except Exception as e:
                print(f"Smart init failed ({e}), falling back to random rejection sampling.")
                # Fallback to random
                initial_positions = []
                attempts = 0
                while len(initial_positions) < nwalkers and attempts < 10000:
                    try:
                        proposal = _sample_from_prior()
                        if np.isfinite(self._log_probability(proposal)):
                            initial_positions.append(proposal)
                    except:
                        pass
                    attempts += 1
                
                if len(initial_positions) < nwalkers:
                    raise RuntimeError("Could not find enough valid initial positions from Prior.")
                initial_positions = np.array(initial_positions)

        # Ensure distinct points (emcee requires distinct walkers)
        # Add tiny distinct jitter if duplicates exist
        if len(np.unique(initial_positions, axis=0)) < nwalkers:
             initial_positions += 1e-6 * np.random.randn(*initial_positions.shape)
        
        # Verify valid again? (optional, skip for speed)
        # Checking validity of *final* initial_positions
        for i, val in enumerate(initial_positions):
            if not np.isfinite(self._log_probability(val)):
                # If invalid, retry finding a neighbor
                valid = False
                attempts = 0
                while not valid and attempts < 100:
                    proposal = center + 1e-2 * np.random.randn(_ndim)
                    if np.isfinite(self._log_probability(proposal)):
                        initial_positions[i] = proposal
                        valid = True
                    attempts += 1
                if not valid: initial_positions[i] = proposal # Give up

        # Run MCMC
        if n_cores > 1:
            print(f"Running MCMC with {n_cores} cores...")
            
            # Set global variable for workers to inherit (fork) or access
            global _GLOBAL_FITTER_INSTANCE
            _GLOBAL_FITTER_INSTANCE = self
            
            # Use the global wrapper function
            with Pool(n_cores) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, _ndim, _global_log_prob, pool=pool, **kwargs)
                sampler.run_mcmc(initial_positions, nsteps, progress=True)
                
            # Cleanup global reference (optional)
            _GLOBAL_FITTER_INSTANCE = None
        else:
            sampler = emcee.EnsembleSampler(nwalkers, _ndim, self._log_probability, **kwargs)
            sampler.run_mcmc(initial_positions, nsteps, progress=True)
            
        self.sampler = sampler
        self.samples = sampler.get_chain(discard=burnin, flat=True)
        return sampler

    # ------------------------------------------------------------------
    # Helper: batch-evaluate a grid output for all posterior samples
    # ------------------------------------------------------------------

    def _batch_evaluate_grid(
        self,
        comp_index: int,
        samples: np.ndarray,
        ions: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Batch-evaluate a Cloudy grid for one component across many posterior samples.

        Parameters
        ----------
        comp_index
            Index into ``self.grids`` / ``self.component_param_templates``.
        samples
            The (possibly thinned) posterior ``self.samples`` array.
        ions
            Subset of grid ion names to return.  ``None`` ⇒ all.

        Returns
        -------
        Dict mapping ion name → 1-D array of log10 values.
        """
        grid = self.grids[comp_index]
        sl = self.param_slices[comp_index]
        p_order = self.component_param_templates[comp_index]
        comp_samples = samples[:, sl]
        n_pts = comp_samples.shape[0]

        eval_points = np.zeros((n_pts, len(grid.axis_names)))
        for dim, axis_name in enumerate(grid.axis_names):
            if axis_name == "z":
                eval_points[:, dim] = self.config.redshift
            elif axis_name in p_order:
                p_idx = p_order.index(axis_name)
                eval_points[:, dim] = comp_samples[:, p_idx]

        all_vals = grid.interpolator(eval_points)  # (n_pts, n_ions)
        target_ions = ions if ions is not None else list(grid.ion_order)
        result: Dict[str, np.ndarray] = {}
        for ion in target_ions:
            if ion in grid.ion_order:
                idx = grid.ion_order.index(ion)
                result[ion] = all_vals[:, idx]
        return result

    # ------------------------------------------------------------------
    # Enhanced corner plot
    # ------------------------------------------------------------------

    def corner_plot(
        self,
        include_derived: bool = True,
        include_temperature: bool = True,
        include_totals: bool = True,
        include_observation_ions: bool = False,
        predicted_ions: Optional[Sequence[str]] = None,
        include_b_total: bool = False,
        b_total_mass_amu: float = 1.008,
        include_kinematics: bool = True,
        jeans_overlay: bool = False,
        jeans_kwargs: Optional[Mapping[str, Any]] = None,
        labels: Optional[Sequence[str]] = None,
        truths: Optional[Sequence[float]] = None,
        thin: int = 1,
        max_samples: Optional[int] = None,
        **corner_kwargs,
    ):
        """Corner plot of MCMC samples with optional derived physical quantities.

        Derived quantities (``include_derived=True``):

        * Per component – cloud path-length *L* (kpc), cloud mass *M* (M☉),
          PIE equilibrium temperature *T_cloudy*.
        * Summed across components – total Σ\ *L*, total Σ\ *M*.

        A Jeans instability boundary can be overlaid on relevant panels with
        ``jeans_overlay=True``.

        Parameters
        ----------
        include_derived : bool
            Append derived columns (L, M, T) to the corner samples.
        include_temperature : bool
            Include PIE *T_cloudy* per component (requires grid column).
        include_totals : bool
            Show total Σ\ L and Σ\ M when more than one component.
        include_kinematics : bool
            If ``False``, strip kinematic pass-through parameters (v, b_other)
            and b_total from the corner plot.  Default ``True``.
        jeans_overlay : bool
            Draw Jeans instability boundary on (n_H, N_H), (n_H, L),
            and (N_H, L) panels per component.
        jeans_kwargs : dict, optional
            Forwarded to ``_overlay_jeans_boundary_joint``.
        labels, truths
            Override labels / truth markers.
        thin : int
            Keep one in every *thin* posterior samples (default 1).
        max_samples : int, optional
            Cap the number of samples used for the plot.
        **corner_kwargs
            Passed through to :func:`corner.corner`.
        """
        if self.samples is None:
            raise RuntimeError("Run MCMC before calling corner_plot")
        try:
            import corner as corner_pkg
        except ImportError as exc:
            raise RuntimeError("Install 'corner' to create corner plots") from exc

        # ---- subsample / thin ------------------------------------------------
        raw_samples = self._expand_samples(self.samples)
        n_total = raw_samples.shape[0]
        indices = np.arange(0, n_total, max(thin, 1))
        if max_samples is not None and max_samples > 0 and len(indices) > max_samples:
            rng = np.random.default_rng()
            indices = np.sort(rng.choice(indices, size=max_samples, replace=False))
        base_samples = raw_samples[indices]
        base_dim = base_samples.shape[1]
        n_pts = base_samples.shape[0]

        derived_arrays: List[np.ndarray] = []
        derived_labels: List[str] = []

        # per-component caches (needed for Jeans overlay later)
        per_comp_T_cloudy: Dict[int, np.ndarray] = {}
        per_comp_L: Dict[int, np.ndarray] = {}
        per_comp_M: Dict[int, np.ndarray] = {}

        if include_derived:
            for i in range(len(self.grids)):
                grid = self.grids[i]
                sl = self.param_slices[i]
                p_order = self.component_param_templates[i]
                comp_id = self.component_ids[i]
                comp_samples = base_samples[:, sl]

                # --- L and M from chain parameters (vectorised) ---------------
                n_H_idx = p_order.index("n_H")
                col_key = "N_H" if "N_H" in p_order else "NHI"
                col_idx = p_order.index(col_key)

                log_n_H = comp_samples[:, n_H_idx]
                log_col = comp_samples[:, col_idx]

                # When the grid axis is NHI, prefer N_H_total for L/M if available
                log_col_for_LM = log_col
                if col_key == "NHI" and "N_H_total" in grid.ion_order:
                    batch = self._batch_evaluate_grid(i, base_samples, ions=["N_H_total"])
                    if "N_H_total" in batch:
                        log_col_for_LM = batch["N_H_total"]

                log_L_kpc = log_col_for_LM - log_n_H - LOG10_CM_PER_KPC
                log_M_solar = (
                    3.0 * log_col_for_LM - 2.0 * log_n_H
                    + LOG10_PROTON_MASS_G - LOG10_SOLAR_MASS_G
                )
                per_comp_L[i] = log_L_kpc
                per_comp_M[i] = log_M_solar

                derived_arrays.append(log_L_kpc[:, np.newaxis])
                derived_labels.append(f"log L$_{{{comp_id}}}$ (kpc)")
                derived_arrays.append(log_M_solar[:, np.newaxis])
                derived_labels.append(f"log M$_{{{comp_id}}}$ (M$_\\odot$)")

                # --- Element masses per component (vectorised) -----------------
                _constrained_els = self._constrained_elements()
                Z_idx = p_order.index("Z") if "Z" in p_order else None
                for elem in _constrained_els:
                    if elem not in SOLAR_LOG_ABUNDANCE or elem not in LOG10_ATOMIC_MASS_G:
                        continue
                    ref = self._per_comp_abundance_reference[i]
                    log_Z = comp_samples[:, Z_idx] if Z_idx is not None else np.zeros(n_pts)
                    if elem == ref:
                        log_X_over_H = SOLAR_LOG_ABUNDANCE[elem] + log_Z
                    else:
                        ratio_key = _build_ratio_key(elem, ref)
                        ab_fixed = self._per_comp_abundance_fixed[i]
                        if ratio_key in p_order:
                            offset = comp_samples[:, p_order.index(ratio_key)]
                        else:
                            offset = ab_fixed.get(ratio_key, 0.0)
                        log_X_over_H = SOLAR_LOG_ABUNDANCE[elem] + log_Z + offset
                    log_Me_solar = (
                        log_X_over_H
                        + LOG10_ATOMIC_MASS_G[elem]
                        + 3.0 * log_col_for_LM
                        - 2.0 * log_n_H
                        - LOG10_SOLAR_MASS_G
                    )
                    _elem_mass_key = f"_emass_{elem}_{i}"
                    per_comp_M[_elem_mass_key] = log_Me_solar
                    derived_arrays.append(log_Me_solar[:, np.newaxis])
                    derived_labels.append(
                        f"log M$_{{{elem},{comp_id}}}$ (M$_\\odot$)"
                    )

                # --- T_cloudy from grid (batch) --------------------------------
                if include_temperature and "T_cloudy" in grid.ion_order:
                    batch_t = self._batch_evaluate_grid(i, base_samples, ions=["T_cloudy"])
                    t_arr = batch_t.get("T_cloudy")
                    if t_arr is not None:
                        per_comp_T_cloudy[i] = t_arr
                        derived_arrays.append(t_arr[:, np.newaxis])
                        derived_labels.append(f"log T$_{{{comp_id}}}$ (PIE)")

            # --- Summed quantities across components --------------------------
            if include_totals and len(self.grids) > 1:
                total_L_linear = np.zeros(n_pts)
                total_M_linear = np.zeros(n_pts)
                for i in range(len(self.grids)):
                    total_L_linear += 10.0 ** per_comp_L[i]
                    total_M_linear += 10.0 ** per_comp_M[i]
                log_total_L = np.log10(total_L_linear)
                log_total_M = np.log10(total_M_linear)
                derived_arrays.append(log_total_L[:, np.newaxis])
                derived_labels.append(r"log $\Sigma$L (kpc)")
                derived_arrays.append(log_total_M[:, np.newaxis])
                derived_labels.append(r"log $\Sigma$M (M$_\odot$)")

                # Total element masses across components
                _constrained_els = self._constrained_elements()
                for elem in _constrained_els:
                    total_elem_linear = np.zeros(n_pts)
                    any_found = False
                    for ci in range(len(self.grids)):
                        key = f"_emass_{elem}_{ci}"
                        if key in per_comp_M:
                            total_elem_linear += 10.0 ** per_comp_M[key]
                            any_found = True
                    if any_found:
                        log_total_elem = np.log10(np.clip(total_elem_linear, 1e-300, None))
                        derived_arrays.append(log_total_elem[:, np.newaxis])
                        derived_labels.append(
                            f"log $\\Sigma$M$_{{{elem}}}$ (M$_\\odot$)"
                        )

        # ---- observation-ion panels (upper / lower limit ions) ---------------
        obs_truths_map: Dict[str, float] = {}  # label -> limit value
        if include_observation_ions and self.observations:
            for obs_ion, obs_obj in self.observations.items():
                # Sum predicted columns across components for this ion
                total_linear = np.zeros(n_pts)
                any_found = False
                for ci in range(len(self.grids)):
                    batch = self._batch_evaluate_grid(ci, base_samples, ions=[obs_ion])
                    if obs_ion in batch:
                        total_linear += 10.0 ** batch[obs_ion]
                        any_found = True
                if any_found:
                    log_total = np.log10(np.clip(total_linear, 1e-40, None))
                    derived_arrays.append(log_total[:, np.newaxis])
                    lbl = f"log N({obs_ion.replace('N_', '')}) [obs]"
                    derived_labels.append(lbl)
                    obs_truths_map[lbl] = obs_obj.value

        # ---- predicted-ion panels (ions not in GMM or observations) ---------
        if predicted_ions:
            for pred_ion in predicted_ions:
                for ci in range(len(self.grids)):
                    comp_id = self.component_ids[ci]
                    batch = self._batch_evaluate_grid(ci, base_samples, ions=[pred_ion])
                    if pred_ion in batch:
                        col_arr = batch[pred_ion].copy()
                        p_order = self.component_param_templates[ci]
                        comp_samp = base_samples[:, self.param_slices[ci]]
                        col_arr = self._apply_abundance_offsets_batch(
                            ci, pred_ion, col_arr, comp_samp, p_order,
                        )
                        derived_arrays.append(col_arr[:, np.newaxis])
                        ion_short = pred_ion.replace('N_', '')
                        lbl = f"log N({ion_short})$_{{{comp_id}}}$ [pred]"
                        derived_labels.append(lbl)

        # ---- derived cross-ratio abundance constraint posteriors --------------
        # For bounded (non-fixed) constraints, compute the derived ratio from
        # the MCMC samples and add it as a corner-plot panel.  Fixed constraints
        # are skipped (single value, nothing to plot).
        if self._has_abundance_constraints():
            for ci in range(len(self.grids)):
                comp_id = self.component_ids[ci]
                constraints = self._per_comp_abundance_constraints[ci]
                if not constraints:
                    continue
                p_order = self.component_param_templates[ci]
                comp_samp = base_samples[:, self.param_slices[ci]]
                ref = self._per_comp_abundance_reference[ci]
                ab_fixed_ci = self._per_comp_abundance_fixed[ci]

                for ratio_key, num_el, den_el, is_fixed, fixed_val, bounds_val in constraints:
                    if is_fixed:
                        # Don't plot: single value, not informative
                        continue

                    # Compute derived ratio = num_offset - den_offset
                    # (both in solar-relative log space wrt ref)
                    if num_el == ref:
                        num_arr = np.zeros(n_pts)
                    else:
                        nk = _build_ratio_key(num_el, ref)
                        if nk in p_order:
                            num_arr = comp_samp[:, p_order.index(nk)]
                        else:
                            num_arr = np.full(n_pts, ab_fixed_ci.get(nk, 0.0))

                    if den_el == ref:
                        den_arr = np.zeros(n_pts)
                    else:
                        dk = _build_ratio_key(den_el, ref)
                        if dk in p_order:
                            den_arr = comp_samp[:, p_order.index(dk)]
                        else:
                            den_arr = np.full(n_pts, ab_fixed_ci.get(dk, 0.0))

                    derived_ratio = num_arr - den_arr
                    derived_arrays.append(derived_ratio[:, np.newaxis])
                    derived_labels.append(
                        f"log {num_el}/{den_el}$_{{{comp_id}}}$ [constr]"
                    )

        # ---- b_total as derived quantity (per component) ---------------------
        if include_b_total and include_kinematics and self.has_kinematics_in_likelihood:
            # b_thermal = 0.12845 * sqrt(T) / sqrt(mass_amu)  [km/s]
            _B_CONST = 0.12845  # sqrt(2 k_B / m_p) in km/(s sqrt(K*amu))
            for i in range(len(self.grids)):
                comp_id = self.component_ids[i]
                p_order = self.component_param_templates[i]
                comp_samples_local = base_samples[:, self.param_slices[i]]
                if "b_other" not in p_order:
                    continue
                b_other_idx = p_order.index("b_other")
                b_other_vals = comp_samples_local[:, b_other_idx]
                # Determine T: prefer PIE T_cloudy, fall back to chain T
                if i in per_comp_T_cloudy:
                    log_T = per_comp_T_cloudy[i]
                elif "T" in p_order:
                    log_T = comp_samples_local[:, p_order.index("T")]
                else:
                    continue
                T_linear = 10.0 ** log_T
                b_thermal = _B_CONST * np.sqrt(T_linear) / np.sqrt(b_total_mass_amu)
                b_total = np.sqrt(b_thermal ** 2 + b_other_vals ** 2)
                derived_arrays.append(b_total[:, np.newaxis])
                derived_labels.append(f"b$_{{total,{comp_id}}}$ (km/s)")

        # ---- optionally strip kinematic columns (v_*, b_other_*) from base ---
        _KIN_PREFIXES = ("v_", "b_other_")
        if not include_kinematics:
            _keep = [
                not any(n.startswith(p) for p in _KIN_PREFIXES)
                for n in self.param_names
            ]
            _keep_idx = [i for i, k in enumerate(_keep) if k]
            base_samples = base_samples[:, _keep_idx]
            _base_param_names: List[str] = [n for n, k in zip(self.param_names, _keep) if k]
        else:
            _base_param_names = list(self.param_names)

        base_dim = base_samples.shape[1]  # refresh after kinematic filtering

        # ---- assemble plot_samples -------------------------------------------
        if derived_arrays:
            plot_samples = np.hstack([base_samples] + derived_arrays)
        else:
            plot_samples = base_samples

        finite_mask = np.all(np.isfinite(plot_samples), axis=1)
        if not np.all(finite_mask):
            n_dropped = int(n_pts - np.count_nonzero(finite_mask))
            logger.warning(
                "Dropping %d samples with non-finite derived values in corner plot",
                n_dropped,
            )
            plot_samples = plot_samples[finite_mask]
            # also filter caches for jeans overlay
            for key in per_comp_T_cloudy:
                per_comp_T_cloudy[key] = per_comp_T_cloudy[key][finite_mask]
            if plot_samples.shape[0] == 0:
                raise RuntimeError(
                    "All samples contain non-finite derived values; check grid completeness."
                )

        # ---- labels ----------------------------------------------------------
        if labels is None:
            base_labels = [self._prettify_param_label(n) for n in _base_param_names]
            plot_labels: List[str] = base_labels + derived_labels
        else:
            plot_labels = list(labels)
            if len(plot_labels) == base_dim:
                plot_labels = plot_labels + derived_labels

        if len(plot_labels) != plot_samples.shape[1]:
            raise ValueError(
                f"Number of labels ({len(plot_labels)}) does not match "
                f"corner-plot dimensionality ({plot_samples.shape[1]})"
            )

        # ---- draw corner -----------------------------------------------------
        # Merge observation-ion limit values into the truths array so that
        # vertical / horizontal lines appear on the relevant panels.
        if truths is None:
            truths = [None] * plot_samples.shape[1]
        else:
            truths = list(truths)
            while len(truths) < plot_samples.shape[1]:
                truths.append(None)
        for lbl, limit_val in obs_truths_map.items():
            if lbl in plot_labels:
                truths[plot_labels.index(lbl)] = limit_val

        defaults = dict(
            plot_datapoints=False,
            plot_density=False,
            plot_contours=True,
            no_fill_contours=True,
            levels=(0.68, 0.95, 0.99),
            hist_kwargs={"density": True},
        )
        defaults.update(corner_kwargs)

        fig = corner_pkg.corner(
            plot_samples,
            labels=plot_labels,
            truths=truths,
            range=[0.995] * plot_samples.shape[1],
            **defaults,
        )

        # Colour diagonal histograms to match labelled panel meanings
        try:
            import matplotlib.patches as mpatches
            import matplotlib as mpl
            for ax in fig.get_axes():
                rects = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)]
                if not rects:
                    continue
                lbl = (ax.get_xlabel() or ax.get_title() or "")
                if "[pred]" in lbl:
                    c = "#de2d26"
                elif "[obs]" in lbl:
                    c = "#000000"
                else:
                    c = None
                if c is not None:
                    for r in rects:
                        r.set_facecolor(c)
                        r.set_edgecolor("none")
                        r.set_alpha(0.6)
        except Exception:
            pass

        # ---- Jeans overlay ---------------------------------------------------
        if jeans_overlay and include_derived:
            self._overlay_jeans_boundary_joint(
                fig,
                plot_labels,
                plot_samples,
                per_comp_T_cloudy,
                **(dict(jeans_kwargs) if jeans_kwargs else {}),
            )

        self._last_corner_labels = list(plot_labels)
        return fig

    # ------------------------------------------------------------------
    # Jeans overlay helper for joint corner plot
    # ------------------------------------------------------------------

    def _overlay_jeans_boundary_joint(
        self,
        fig,
        plot_labels: List[str],
        plot_samples: np.ndarray,
        per_comp_T_cloudy: Dict[int, np.ndarray],
        *,
        jeans_color: str = "forestgreen",
        jeans_alpha: float = 0.15,
        jeans_label: str = "Jeans limit",
    ) -> None:
        """Draw Jeans instability boundaries on the joint corner plot.

        The boundary ``L < L_Jeans`` is plotted on the (n_H, N_H),
        (n_H, L) and (N_H, L) panels for each component using the
        16/50/84-th percentile of the PIE *T_cloudy* posterior.
        """
        ndim = plot_samples.shape[1]
        try:
            axes = np.array(fig.axes).reshape(ndim, ndim)
        except ValueError:
            logger.warning("Cannot reshape axes for Jeans overlay; skipping")
            return

        label_used = False

        def _add_jeans_line(ax, x_range, y_func, log_T_vals):
            nonlocal label_used
            x = np.linspace(x_range[0], x_range[1], 300)
            y_med = y_func(x, _JEANS_A_CONST + log_T_vals[1])
            y_lo = y_func(x, _JEANS_A_CONST + log_T_vals[0])
            y_hi = y_func(x, _JEANS_A_CONST + log_T_vals[2])
            line_label = jeans_label if not label_used else None
            ax.plot(
                x, y_med, ":", color=jeans_color, linewidth=1.5,
                label=line_label, zorder=10,
            )
            ax.fill_between(
                x, y_lo, y_hi, alpha=jeans_alpha, color=jeans_color, zorder=5,
            )
            label_used = True

        for i in range(len(self.grids)):
            if i not in per_comp_T_cloudy:
                continue
            comp_id = self.component_ids[i]
            t_samples = per_comp_T_cloudy[i]
            finite_t = t_samples[np.isfinite(t_samples)]
            if finite_t.size < 3:
                continue
            T_pcts = tuple(np.percentile(finite_t, [16, 50, 84]))

            # Locate axes indices in the label list
            idx_nH: Optional[int] = None
            idx_NH: Optional[int] = None
            idx_L: Optional[int] = None
            for j, lbl in enumerate(plot_labels):
                if lbl == f"n_H_{comp_id}":
                    idx_nH = j
                elif lbl == f"N_H_{comp_id}" or lbl == f"NHI_{comp_id}":
                    idx_NH = j
                elif "log L" in lbl and str(comp_id) in lbl:
                    idx_L = j

            if idx_nH is None:
                continue

            # (n_H, N_H): N_H < 0.5*n_H + 0.5*(A + T)
            if idx_NH is not None and idx_NH > idx_nH:
                ax = axes[idx_NH, idx_nH]
                xlim = ax.get_xlim()
                _add_jeans_line(
                    ax, xlim,
                    lambda x, b: 0.5 * x + 0.5 * b + LOG10_CM_PER_KPC,
                    T_pcts,
                )
                ax.set_xlim(xlim)

            # (n_H, L): L < -0.5*n_H + 0.5*(A+T) - LOG_CM
            if idx_L is not None and idx_L > idx_nH:
                ax = axes[idx_L, idx_nH]
                xlim = ax.get_xlim()
                _add_jeans_line(
                    ax, xlim,
                    lambda x, b: -0.5 * x + 0.5 * b - LOG10_CM_PER_KPC,
                    T_pcts,
                )
                ax.set_xlim(xlim)

            # (N_H, L): L < -N_H + (A+T) - LOG_CM
            if idx_NH is not None and idx_L is not None and idx_L > idx_NH:
                ax = axes[idx_L, idx_NH]
                xlim = ax.get_xlim()
                _add_jeans_line(
                    ax, xlim,
                    lambda x, b: -x + b - LOG10_CM_PER_KPC,
                    T_pcts,
                )
                ax.set_xlim(xlim)

    # ------------------------------------------------------------------
    # Temperature comparison plot
    # ------------------------------------------------------------------

    def temperature_comparison_plot(
        self,
        voigt_temperature_samples: Optional[Sequence[Optional[np.ndarray]]] = None,
        thin: int = 1,
        max_samples: Optional[int] = 5000,
        figsize: Optional[Tuple[float, float]] = None,
        bins: int = 50,
    ):
        """Compare PIE temperature from Cloudy with Voigt-profile temperature posteriors.

        This plot is always available, regardless of whether the Voigt
        temperature prior is enabled.

        Parameters
        ----------
        voigt_temperature_samples
            List of ``log10(T)`` arrays from the Voigt profile posterior,
            one entry per component. Entries may be ``None``.
        thin : int
            Keep every *thin*-th posterior sample.
        max_samples : int, optional
            Cap the number of samples evaluated against the grid.
        figsize : tuple, optional
            Figure size; defaults to ``(5*n_components, 4)``.
        bins : int
            Number of histogram bins.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.samples is None:
            raise RuntimeError("Run MCMC before calling temperature_comparison_plot")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib required for temperature_comparison_plot") from exc

        n_comp = len(self.grids)
        if figsize is None:
            figsize = (5 * n_comp, 4)

        fig, axes = plt.subplots(1, n_comp, figsize=figsize, squeeze=False)
        axes = axes[0]

        # subsample (expand to full space)
        full_samples = self._expand_samples(self.samples)
        n_total = full_samples.shape[0]
        idx_arr = np.arange(0, n_total, max(thin, 1))
        if max_samples and len(idx_arr) > max_samples:
            rng = np.random.default_rng()
            idx_arr = np.sort(rng.choice(idx_arr, size=max_samples, replace=False))
        samples = full_samples[idx_arr]

        for i in range(n_comp):
            ax = axes[i]
            comp_id = self.component_ids[i]
            grid = self.grids[i]

            # Batch evaluate T_cloudy
            if "T_cloudy" in grid.ion_order:
                batch = self._batch_evaluate_grid(i, samples, ions=["T_cloudy"])
                t_cloudy = batch.get("T_cloudy")
                if t_cloudy is not None:
                    t_finite = t_cloudy[np.isfinite(t_cloudy)]
                    if t_finite.size > 0:
                        ax.hist(
                            t_finite, bins=bins, density=True, histtype="step",
                            color="red", linewidth=2, label="Cloudy PIE",
                        )
            elif "T" in [p for p in self.component_param_templates[i]]:
                # Free T mode: temperature is a fit parameter
                sl = self.param_slices[i]
                p_order = self.component_param_templates[i]
                t_idx_in_comp = p_order.index("T")
                t_chain = samples[:, sl][:, t_idx_in_comp]
                t_finite = t_chain[np.isfinite(t_chain)]
                if t_finite.size > 0:
                    ax.hist(
                        t_finite, bins=bins, density=True, histtype="step",
                        color="red", linewidth=2, label="Cloudy (free T)",
                    )

            # Voigt temperature overlay
            if voigt_temperature_samples is not None and i < len(voigt_temperature_samples):
                t_voigt = voigt_temperature_samples[i]
                if t_voigt is not None:
                    t_voigt = np.asarray(t_voigt, dtype=float)
                    t_voigt_finite = t_voigt[np.isfinite(t_voigt)]
                    if t_voigt_finite.size > 0:
                        ax.hist(
                            t_voigt_finite, bins=bins, density=True, histtype="step",
                            color="blue", linewidth=2, label="Voigt Profile",
                        )

            ax.set_xlabel("log T (K)")
            ax.set_ylabel("Density")
            ax.set_title(f"Component {comp_id}")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        fig.tight_layout()
        return fig

    def posterior_check_plot(
        self,
        n_samples=1000,
        include_observation_ions=True,
        predicted_ions: Optional[Sequence[str]] = None,
        include_kinematics: bool = True,
    ):
        """Plot 'True' KDE samples vs 'Predicted' samples from the Cloudy chain.

        Parameters
        ----------
        n_samples : int
            Number of posterior draws to evaluate.
        include_observation_ions : bool
            Append Cloudy-predicted columns for upper/lower-limit ions.
        predicted_ions : list of str, optional
            Extra ions (e.g. ``["N_OVI"]``) predicted by the Cloudy grid but
            not present in the GMM or observation set.  These are shown as
            predicted-only panels (no "True" contours, just the red
            predicted histogram).
        include_kinematics : bool
            If ``False``, strip kinematic dimensions (v, b_other) from the
            corner comparison.  Default ``True``.
        """
        import corner
        import matplotlib.pyplot as plt

        # 1. Get subset of Cloudy posterior (expand to full space)
        full_samples = self._expand_samples(self.samples)
        if len(full_samples) > n_samples:
            indices = np.random.choice(len(full_samples), size=n_samples, replace=False)
            subset = full_samples[indices]
        else:
            subset = full_samples
            indices = np.arange(len(full_samples))

        # 2. Evaluate predicted column densities for GMM dimensions
        predicted_samples = []
        valid_mask = []
        for theta in subset:
            vector = np.zeros(len(self.kde_obs.param_names))
            valid = True
            for i, sl in enumerate(self.param_slices):
                comp_theta = theta[sl]
                grid = self.grids[i]
                p_order = self.component_param_templates[i]

                params = {}
                if "z" in grid.axis_names:
                    params["z"] = self.config.redshift

                for idx, name in enumerate(p_order):
                    params[name] = comp_theta[idx]

                model = grid.evaluate(params)

                # Apply abundance offsets (per-component)
                self._apply_abundance_offsets_for_component(i, model, params)

                for ion_name, kde_idx in zip(self.ions_per_component[i], self.kde_indices[i]):
                    val = model.get(ion_name)
                    if val is None or not np.isfinite(val):
                        valid = False; break
                    vector[kde_idx] = val
                if not valid: break

                if self.has_temperature_in_likelihood and i in self._temp_kde_idx_for_component:
                    t_val = model.get("T_cloudy")
                    if t_val is None:
                        t_val = model.get("T")
                    if t_val is None or not np.isfinite(t_val):
                        valid = False; break
                    vector[self._temp_kde_idx_for_component[i]] = t_val

                if self.has_kinematics_in_likelihood and i in self._kinematic_kde_idx_for_component:
                    for kin_param, kde_idx in self._kinematic_kde_idx_for_component[i].items():
                        if kin_param in params:
                            vector[kde_idx] = params[kin_param]
                        else:
                            valid = False; break
                if not valid: break
            valid_mask.append(valid)
            if valid:
                predicted_samples.append(vector)

        predicted_samples = np.array(predicted_samples)
        n_valid = len(predicted_samples)

        # Filtered posterior samples (valid only) for batch grid evaluation
        valid_indices = np.array(indices)[np.array(valid_mask)]
        valid_samples = full_samples[valid_indices]

        # ------------------------------------------------------------------
        # 3. Build observation-ion columns (upper / lower limit ions)
        # ------------------------------------------------------------------
        obs_labels: List[str] = []
        obs_truths: List[float] = []          # limit values for vertical lines
        obs_predicted_cols: List[np.ndarray] = []

        if include_observation_ions and self.observations:
            for obs_ion, obs_obj in self.observations.items():
                total_linear = np.zeros(n_valid)
                any_found = False
                for ci in range(len(self.grids)):
                    batch = self._batch_evaluate_grid(ci, valid_samples, ions=[obs_ion])
                    if obs_ion in batch:
                        col = batch[obs_ion].copy()
                        p_order = self.component_param_templates[ci]
                        comp_samp = valid_samples[:, self.param_slices[ci]]
                        col = self._apply_abundance_offsets_batch(
                            ci, obs_ion, col, comp_samp, p_order,
                        )
                        total_linear += 10.0 ** col
                        any_found = True
                if any_found:
                    log_total = np.log10(np.clip(total_linear, 1e-40, None))
                    obs_predicted_cols.append(log_total)
                    suffix = ""
                    if obs_obj.is_upper_limit:
                        suffix = " (UL)"
                    elif obs_obj.is_lower_limit:
                        suffix = " (LL)"
                    lbl = f"log N({obs_ion.replace('N_', '')}){suffix}"
                    obs_labels.append(lbl)
                    obs_truths.append(obs_obj.value)

        # ------------------------------------------------------------------
        # 3b. Build predicted-ion columns per component (no "True" data)
        # ------------------------------------------------------------------
        pred_labels: List[str] = []
        pred_cols: List[np.ndarray] = []

        if predicted_ions:
            for pred_ion in predicted_ions:
                for ci in range(len(self.grids)):
                    comp_id = self.component_ids[ci]
                    batch = self._batch_evaluate_grid(ci, valid_samples, ions=[pred_ion])
                    if pred_ion in batch:
                        col = batch[pred_ion].copy()
                        p_order = self.component_param_templates[ci]
                        comp_samp = valid_samples[:, self.param_slices[ci]]
                        col = self._apply_abundance_offsets_batch(
                            ci, pred_ion, col, comp_samp, p_order,
                        )
                        pred_cols.append(col)
                        ion_short = pred_ion.replace('N_', '')
                        lbl = f"log N({ion_short})$_{{{comp_id}}}$ [pred]"
                        pred_labels.append(lbl)

        # ------------------------------------------------------------------
        # 4. Merge all columns
        # ------------------------------------------------------------------
        labels = list(self.kde_obs.param_names)
        true_data = self.kde_obs.dataset.copy()      # (n_true, n_kde)
        n_true = true_data.shape[0]
        rng = np.random.default_rng(42)

        # Observation-ion columns
        if obs_predicted_cols:
            obs_pred_block = np.column_stack(obs_predicted_cols)
            predicted_samples = np.hstack([predicted_samples, obs_pred_block])

            dummy_cols = np.empty((n_true, len(obs_predicted_cols)))
            for j, col in enumerate(obs_predicted_cols):
                lo, hi = col.min(), col.max()
                pad = max(0.1 * (hi - lo), 0.5)
                dummy_cols[:, j] = rng.uniform(lo - pad, hi + pad, size=n_true)
            true_data = np.hstack([true_data, dummy_cols])
            labels.extend(obs_labels)

        # Predicted-ion columns (predicted-only — fill true_data with uniform)
        if pred_cols:
            pred_block = np.column_stack(pred_cols)
            predicted_samples = np.hstack([predicted_samples, pred_block])

            dummy_pred = np.empty((n_true, len(pred_cols)))
            for j, col in enumerate(pred_cols):
                lo, hi = col.min(), col.max()
                pad = max(0.1 * (hi - lo), 0.5)
                dummy_pred[:, j] = rng.uniform(lo - pad, hi + pad, size=n_true)
            true_data = np.hstack([true_data, dummy_pred])
            labels.extend(pred_labels)

        # ------------------------------------------------------------------
        # 4b. Optionally strip kinematic dimensions (v_*, b_other_*)
        # ------------------------------------------------------------------
        if not include_kinematics:
            _KIN_PFX = ("v_", "b_other_")
            _keep = [
                not any(lbl.startswith(p) for p in _KIN_PFX)
                for lbl in labels
            ]
            _keep_idx = [i for i, k in enumerate(_keep) if k]
            true_data = true_data[:, _keep_idx]
            predicted_samples = predicted_samples[:, _keep_idx]
            labels = [l for l, k in zip(labels, _keep) if k]

        # ------------------------------------------------------------------
        # 5. Plot overlay
        # ------------------------------------------------------------------
        n_dim = len(labels)
        fig_size = max(12, n_dim * 2)
        fig = plt.figure(figsize=(fig_size, fig_size))

        # Compute shared ranges from combined data so both corner calls
        # use identical bin edges on the diagonal histograms.
        _combined = np.vstack([true_data, predicted_samples])
        _shared_range = []
        for _d in range(n_dim):
            _lo = float(np.percentile(_combined[:, _d], 0.25))
            _hi = float(np.percentile(_combined[:, _d], 99.75))
            _shared_range.append((_lo, _hi))

        common_kwargs = dict(
            plot_datapoints=False,
            plot_density=False,
            plot_contours=True,
            no_fill_contours=True,
            levels=(0.68, 0.95, 0.99),
            bins=20,
        )

        # True (Black)
        corner.corner(
            true_data,
            labels=labels,
            color="black",
            fig=fig,
            range=_shared_range,
            contour_kwargs={"linestyles": "-"},
            hist_kwargs={"density": True, "color": "black", "alpha": 0.5},
            **common_kwargs,
        )

        # Predicted (Red)
        corner.corner(
            predicted_samples,
            color="red",
            fig=fig,
            range=_shared_range,
            contour_kwargs={"linestyles": "--"},
            hist_kwargs={"density": True, "color": "red", "alpha": 0.5},
            **common_kwargs,
        )

        # ------------------------------------------------------------------
        # 6. Remove meaningless "True" (black) histograms / contours from
        #    obs-ion and pred-ion panels (those columns are uniform noise
        #    in the True dataset).
        # ------------------------------------------------------------------
        axes = np.array(fig.axes).reshape(n_dim, n_dim)
        # Number of "real" (non-noise) KDE dimensions (may be reduced if
        # kinematic columns were stripped).
        if not include_kinematics:
            _KIN_PFX2 = ("v_", "b_other_")
            n_kde = sum(
                1 for p in self.kde_obs.param_names
                if not any(p.startswith(pfx) for pfx in _KIN_PFX2)
            )
        else:
            n_kde = len(self.kde_obs.param_names)
        n_extra = len(obs_labels) + len(pred_labels)

        if n_extra > 0:
            _black_rgba = np.array([0.0, 0.0, 0.0, 1.0])
            for d in range(n_kde, n_dim):
                # Diagonal: remove the black step-polygon (uniform noise hist)
                ax_diag = axes[d, d]
                for patch in list(ax_diag.patches):
                    ec = np.asarray(patch.get_edgecolor()).ravel()[:4]
                    if np.allclose(ec[:3], _black_rgba[:3], atol=0.05):
                        patch.remove()
                # Off-diagonal panels in the extra row (d, col<d):
                # remove black contours/collections (uniform noise in y dim)
                for col in range(d):
                    ax = axes[d, col]
                    if col >= n_kde:
                        # Both axes are extra → remove ALL black artists
                        for coll in list(ax.collections):
                            ec = np.asarray(coll.get_edgecolors())
                            if ec.size and np.allclose(ec.ravel()[:3], _black_rgba[:3], atol=0.05):
                                coll.remove()
                    elif col < n_kde:
                        # x is KDE, y is extra → black contours are noise
                        for coll in list(ax.collections):
                            ec = np.asarray(coll.get_edgecolors())
                            if ec.size and np.allclose(ec.ravel()[:3], _black_rgba[:3], atol=0.05):
                                coll.remove()
                # Off-diagonal panels in the extra column (row>d, d):
                # remove black contours (uniform noise in x dim)
                for row in range(d + 1, n_dim):
                    ax = axes[row, d]
                    # Only remove if the current dim d is extra
                    for coll in list(ax.collections):
                        ec = np.asarray(coll.get_edgecolors())
                        if ec.size and np.allclose(ec.ravel()[:3], _black_rgba[:3], atol=0.05):
                            coll.remove()

        # ------------------------------------------------------------------
        # 7. Truth lines for observation-ion limits
        # ------------------------------------------------------------------
        if obs_truths:
            for j, truth_val in enumerate(obs_truths):
                dim = n_kde + j
                # Diagonal (1-D histogram): VERTICAL line
                ax_diag = axes[dim, dim]
                ax_diag.axvline(truth_val, color="blue", linewidth=1.5,
                                linestyle="--", zorder=10)
                # Off-diagonal panels where this dimension is the X-axis
                for row in range(dim + 1, n_dim):
                    axes[row, dim].axvline(truth_val, color="blue",
                                           linewidth=0.8, linestyle="--",
                                           alpha=0.5)
                # Off-diagonal panels where this dimension is the Y-axis
                for col in range(dim):
                    axes[dim, col].axhline(truth_val, color="blue",
                                           linewidth=0.8, linestyle="--",
                                           alpha=0.5)

        # ------------------------------------------------------------------
        # 8. Copy x-axis tick labels to the TOP of diagonal panels
        # ------------------------------------------------------------------
        for d in range(n_dim):
            ax_diag = axes[d, d]
            ax_diag.tick_params(axis="x", top=True, labeltop=True,
                                bottom=True, labelbottom=True)
            ax_diag.set_xlabel("")           # keep bottom label from corner
            ax_diag.xaxis.set_label_position("bottom")
            # Add a compact title with the parameter name above each diagonal
            ax_diag.set_title(labels[d], fontsize=8, pad=4)

        return fig

    # ------------------------------------------------------------------
    # Posterior absorption profile plot (Feature 2)
    # ------------------------------------------------------------------

    def posterior_absorption_plot(
        self,
        spectrum,
        line_list_per_ion: Mapping[str, Sequence[str]],
        z: float,
        *,
        n_samples: int = 500,
        vrange: float = 300.0,
        thin: int = 1,
        include_predicted_ions: bool = True,
        figsize_per_panel: Tuple[float, float] = (8.0, 2.5),
        band_alpha: float = 0.25,
        show_components: bool = False,
        stis_min_wave: float = 1900.0,
        lifetime_position: int = 4,
        binning: Optional[Union[int, Mapping[str, int]]] = None,
    ):
        """Plot posterior Voigt absorption profiles predicted by the Cloudy fit.

        For each posterior sample the method:

        1. Evaluates the Cloudy grid to obtain predicted log N for every
           requested ion (applying abundance offsets).
        2. Reads the kinematic parameters (T, b_other, v) from the MCMC
           walker state (passed through from the Voigt chain).
        3. Builds an ``Absorber1D`` model (from ``model_builder``) per
           component and evaluates optical depth on the spectrum's pixel grid.
        4. Converts to normalised flux ``exp(-tau)`` and convolves with the
           appropriate instrument LSF (COS or STIS via ``lsf.py``).

        The resulting ensemble of profiles is summarised as median + 68 %
        and 95 % bands and overlaid on the observed spectrum.

        Parameters
        ----------
        spectrum : specutils.Spectrum1D
            Observed (normalised) spectrum.
        line_list_per_ion : dict
            Mapping from Cloudy ion name (e.g. ``"N_CIV"``) to a list of
            SEARCH_LINES ``tempname`` keys (e.g. ``["CIV_1", "CIV_2"]``).
        z : float
            Absorber redshift.
        n_samples : int
            Number of posterior draws to evaluate (default 500).
        vrange : float
            Half-width of the velocity window (km/s) for each panel.
        thin : int
            Keep every *thin*-th posterior sample before sub-sampling.
        include_predicted_ions : bool
            Allow ions not in the Voigt GMM but available in the grid.
        figsize_per_panel : tuple
            ``(width, height)`` in inches per line panel.
        band_alpha : float
            Transparency of the confidence bands.
        show_components : bool
            If ``True``, draw individual component profiles.
        stis_min_wave : float
            Observed wavelength (Angstrom) above which the STIS LSF is
            used instead of the COS LSF.
        lifetime_position : int
            COS lifetime position for LSF lookup (default 4).
        binning : int or dict, optional
            Pixel binning for the *observed* spectrum in each panel.
            * ``int`` — bin every panel by this factor.
            * ``dict`` — mapping from SEARCH_LINES ``tempname`` key to an
              integer bin factor, e.g. ``{"CIV_1": 3, "OVI_1": 5}``.
              Lines not listed fall back to 1 (no binning).
            Default ``None`` means no rebinning (native pixels).

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.samples is None:
            raise RuntimeError("Run MCMC before calling posterior_absorption_plot")

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required") from exc

        # Lazy imports — use the same libraries as the rest of the package
        from scipy.special import wofz
        from scipy.ndimage import convolve1d
        from line_info import SEARCH_LINES
        from lsf import getCOSlsf, getSTISlsf
        import astropy.units as u
        from astropy.constants import c as _c, k_B as _kB, m_p as _mp, m_e as _me, e as _e

        # Physical constants (matching model_builder.py exactly)
        _c_kms = _c.to(u.km / u.s).value
        _c_As = _c.to(u.Angstrom / u.s).value
        _constant1 = (np.sqrt(2 * _kB / (_mp * _c ** 2)) * _c).to(
            u.km / u.s / u.K ** 0.5
        ).value
        _prefactor_tau = (np.pi * _e.gauss ** 2 / (_me * _c)).cgs.value
        _sqrt_pi = np.sqrt(np.pi)
        # NOTE: the Cloudy grid already stores log10(N / cm^-2), so we do NOT
        # apply the model_builder ``multiple`` factor (which converts from
        # astropy parameter units to cm^-2).  10**log_N is already in cm^-2.

        # ---- Subsample posterior (expand to full space) -------------------------
        full_samples = self._expand_samples(self.samples)
        n_total = full_samples.shape[0]
        indices = np.arange(0, n_total, max(thin, 1))
        rng = np.random.default_rng()
        if len(indices) > n_samples:
            indices = np.sort(rng.choice(indices, size=n_samples, replace=False))
        samples = full_samples[indices]
        n_pts = samples.shape[0]

        # ---- Pre-compute line constants --------------------------------------
        all_panels: List[Tuple[str, str]] = []  # (ion, line_key)
        for ion, lines in line_list_per_ion.items():
            for lk in lines:
                all_panels.append((ion, lk))

        _line_cache: Dict[str, Tuple[float, float, float, float, float]] = {}
        for _, lk in all_panels:
            if lk in _line_cache:
                continue
            info = SEARCH_LINES[SEARCH_LINES["tempname"] == lk]
            if len(info) == 0:
                raise ValueError(f"Line key '{lk}' not found in SEARCH_LINES")
            lam0 = info["wave"][0]
            lam0_val = lam0.to_value(u.Angstrom) if isinstance(lam0, u.Quantity) else float(lam0)
            gamma = info["Gamma"][0]
            gamma_val = gamma.to_value(u.Hz) if isinstance(gamma, u.Quantity) else float(gamma)
            f_val = info["f"][0]
            f_val = f_val.value if isinstance(f_val, u.Quantity) else float(f_val)
            mass = info["mass"][0]
            mass_val = mass.to_value(u.u) if hasattr(mass, "to") else float(mass)
            nu0 = _c_As / lam0_val
            _line_cache[lk] = (f_val, lam0_val, gamma_val, mass_val, nu0)

        _ion_for_line: Dict[str, str] = {}
        for ion, lines in line_list_per_ion.items():
            for lk in lines:
                _ion_for_line[lk] = ion

        # ---- Build pixel masks on the spectrum grid per panel ----------------
        spec_wave = np.asarray(spectrum.spectral_axis.value, dtype=float)  # Angstrom
        spec_flux_arr = np.asarray(spectrum.flux.value, dtype=float)
        spec_err_arr = None
        if hasattr(spectrum, "uncertainty") and spectrum.uncertainty is not None:
            spec_err_arr = np.asarray(spectrum.uncertainty.array, dtype=float)

        # For each line, find the spectrum pixels within ±vrange and a padded
        # region for LSF convolution (pad ≈ 3× the LSF kernel half-width).
        _LSF_PAD_PIX = 150  # extra pixels each side for convolution edge effects

        panel_pixel_info: Dict[str, Dict] = {}
        for _, lk in all_panels:
            if lk in panel_pixel_info:
                continue
            _, lam0_val, _, _, _ = _line_cache[lk]
            lam_obs = lam0_val * (1.0 + z)

            # velocity of each spectrum pixel relative to this line
            vel_all = (spec_wave - lam_obs) / lam_obs * _c_kms

            # Tight mask: pixels within ±vrange (for plotting)
            tight = np.where((vel_all >= -vrange) & (vel_all <= vrange))[0]
            if tight.size == 0:
                panel_pixel_info[lk] = None
                continue

            # Padded mask: extend for LSF convolution
            i_lo = max(tight[0] - _LSF_PAD_PIX, 0)
            i_hi = min(tight[-1] + _LSF_PAD_PIX + 1, len(spec_wave))
            padded_idx = np.arange(i_lo, i_hi)
            wave_padded = spec_wave[padded_idx]

            # Offset of the tight region within the padded array
            tight_start = tight[0] - i_lo
            tight_end = tight[-1] - i_lo + 1

            # Get LSF kernel for this observed wavelength
            if lam_obs >= stis_min_wave:
                lsf_data = getSTISlsf(lam_obs)
            else:
                lsf_data = getCOSlsf(lam_obs, lifetimePosition=lifetime_position)
            lsf_kernel = lsf_data["kernel"].astype(float)
            # Ensure normalised
            lsf_kernel = lsf_kernel / lsf_kernel.sum()

            panel_pixel_info[lk] = {
                "padded_idx": padded_idx,
                "wave_padded": wave_padded,
                "tight_start": tight_start,
                "tight_end": tight_end,
                "tight_idx": tight,
                "vel_tight": vel_all[tight],
                "lam_obs": lam_obs,
                "lsf_kernel": lsf_kernel,
            }

        # ---- Evaluate tau for each posterior sample --------------------------
        # Accumulate tau on the padded spectrum pixel grid (per line)
        tau_total: Dict[str, np.ndarray] = {}
        tau_comp: Dict[Tuple[str, int], np.ndarray] = {}
        for _, lk in all_panels:
            pinfo = panel_pixel_info.get(lk)
            if pinfo is None:
                continue
            n_pix_padded = len(pinfo["wave_padded"])
            if lk not in tau_total:
                tau_total[lk] = np.zeros((n_pts, n_pix_padded))
            if show_components:
                for ci in range(len(self.grids)):
                    key = (lk, ci)
                    if key not in tau_comp:
                        tau_comp[key] = np.zeros((n_pts, n_pix_padded))

        _groups = self.config.component_groups
        _voigt_chain = self.config.voigt_chain_for_splitting
        _voigt_pnames = self.config.voigt_chain_param_names

        def _compute_tau_for_absorber(
            log_N_arr, T_arr_lin, b_other_arr, v_arr, lk, wave,
        ):
            """Vectorised Voigt tau computation for one absorber on one line."""
            N_cm2 = 10.0 ** log_N_arr
            f_val, lam0_val, gamma_val, mass_val, nu0 = _line_cache[lk]
            b_thermal = _constant1 * np.sqrt(T_arr_lin) / np.sqrt(mass_val)
            b_other_safe = np.where(np.isfinite(b_other_arr), b_other_arr, 0.0)
            b_tot = np.sqrt(b_thermal ** 2 + b_other_safe ** 2)
            doppler_width = b_tot * nu0 / _c_kms
            doppler_width = np.where(doppler_width > 0, doppler_width, 1.0)
            a_param = gamma_val / (4.0 * np.pi * doppler_width)
            one_plus_z_eff = (1.0 + z) * (1.0 + v_arr / _c_kms)
            wave_rest = wave[np.newaxis, :] / one_plus_z_eff[:, np.newaxis]
            nu_rest = _c_As / wave_rest
            u_arg = (nu_rest - nu0) / doppler_width[:, np.newaxis]
            H = wofz(u_arg + 1j * a_param[:, np.newaxis]).real
            return (
                N_cm2[:, np.newaxis]
                * _prefactor_tau * f_val
                / (_sqrt_pi * doppler_width[:, np.newaxis])
                * H
            )

        for ci in range(len(self.grids)):
            comp_id = self.component_ids[ci]
            p_order = self.component_param_templates[ci]
            sl = self.param_slices[ci]
            comp_samples = samples[:, sl]

            needed_ions = set(line_list_per_ion.keys())
            batch = self._batch_evaluate_grid(ci, samples, ions=list(needed_ions))

            ion_cols: Dict[str, np.ndarray] = {}
            for ion in needed_ions:
                if ion not in batch:
                    continue
                col = batch[ion].copy()
                col = self._apply_abundance_offsets_batch(
                    ci, ion, col, comp_samples, p_order,
                )
                ion_cols[ion] = col

            if _groups is not None and ci in _groups and len(_groups[ci]) > 1:
                # --- Grouped mode: split total N back to individual absorbers ---
                absorber_ids = _groups[ci]
                for _, lk in all_panels:
                    pinfo = panel_pixel_info.get(lk)
                    if pinfo is None:
                        continue
                    ion = _ion_for_line[lk]
                    if ion not in ion_cols:
                        continue
                    total_log_N = ion_cols[ion]
                    wave = pinfo["wave_padded"]

                    split = split_group_columns(
                        total_log_N, _voigt_chain, _voigt_pnames,
                        ion, absorber_ids,
                    )

                    for aid in absorber_ids:
                        t_name = f"T_{aid}"
                        b_name = f"b_other_{aid}"
                        v_name = f"v_{aid}"
                        T_med = np.full(n_pts, 4.0)
                        b_med = np.zeros(n_pts)
                        v_med = np.zeros(n_pts)
                        if _voigt_pnames is not None:
                            if t_name in _voigt_pnames:
                                T_med[:] = np.median(np.log10(_voigt_chain[:, _voigt_pnames.index(t_name)]))
                            if b_name in _voigt_pnames:
                                b_med[:] = np.median(_voigt_chain[:, _voigt_pnames.index(b_name)])
                            if v_name in _voigt_pnames:
                                v_med[:] = np.median(_voigt_chain[:, _voigt_pnames.index(v_name)])

                        tau = _compute_tau_for_absorber(
                            split[aid], 10.0 ** T_med, b_med, v_med, lk, wave,
                        )
                        tau_total[lk] += tau
                        if show_components:
                            tau_comp[(lk, ci)] += tau
            else:
                # --- Standard (ungrouped) mode ---
                T_arr = np.full(n_pts, np.nan)
                b_other_arr = np.full(n_pts, np.nan)
                v_arr = np.zeros(n_pts)

                if "b_other" in p_order:
                    b_other_arr = comp_samples[:, p_order.index("b_other")]
                if "v" in p_order:
                    v_arr = comp_samples[:, p_order.index("v")]

                if "T_cloudy" in self.grids[ci].ion_order:
                    t_batch = self._batch_evaluate_grid(ci, samples, ions=["T_cloudy"])
                    T_arr = t_batch.get("T_cloudy", T_arr)
                elif "T" in p_order:
                    T_arr = comp_samples[:, p_order.index("T")]

                T_linear = 10.0 ** T_arr

                for _, lk in all_panels:
                    pinfo = panel_pixel_info.get(lk)
                    if pinfo is None:
                        continue
                    ion = _ion_for_line[lk]
                    if ion not in ion_cols:
                        continue
                    wave = pinfo["wave_padded"]

                    tau = _compute_tau_for_absorber(
                        ion_cols[ion], T_linear, b_other_arr, v_arr, lk, wave,
                    )
                    tau_total[lk] += tau
                    if show_components:
                        tau_comp[(lk, ci)] += tau

        # ---- Convert tau → flux and convolve with LSF ------------------------
        flux_total: Dict[str, np.ndarray] = {}
        flux_comp_dict: Dict[Tuple[str, int], np.ndarray] = {}

        for _, lk in all_panels:
            pinfo = panel_pixel_info.get(lk)
            if pinfo is None:
                continue
            kernel = pinfo["lsf_kernel"]
            ts = pinfo["tight_start"]
            te = pinfo["tight_end"]

            # Total flux: exp(-tau) then convolve with LSF along pixel axis
            raw_flux = np.exp(-tau_total[lk])                     # (n_pts, n_pix_padded)
            convolved = convolve1d(raw_flux, kernel, axis=1,
                                   mode="constant", cval=1.0)     # LSF convolution
            flux_total[lk] = convolved[:, ts:te]                  # trim to tight region

            if show_components:
                for ci in range(len(self.grids)):
                    raw_c = np.exp(-tau_comp[(lk, ci)])
                    conv_c = convolve1d(raw_c, kernel, axis=1,
                                        mode="constant", cval=1.0)
                    flux_comp_dict[(lk, ci)] = conv_c[:, ts:te]

        # ---- Plot ------------------------------------------------------------
        n_panels = len(all_panels)
        fig_w, fig_h = figsize_per_panel
        fig, axes_arr = plt.subplots(
            n_panels, 1, figsize=(fig_w, fig_h * n_panels), squeeze=False, sharex=False,
        )
        axes_flat = axes_arr[:, 0]

        _comp_colors = [
            "tab:blue", "tab:orange", "tab:green",
            "tab:purple", "tab:brown", "tab:pink",
        ]

        for panel_idx, (ion, lk) in enumerate(all_panels):
            ax = axes_flat[panel_idx]
            pinfo = panel_pixel_info.get(lk)
            if pinfo is None:
                ax.text(0.5, 0.5, f"{lk}: no spectrum coverage",
                        transform=ax.transAxes, ha="center", va="center")
                continue

            vel = pinfo["vel_tight"]
            tight_idx = pinfo["tight_idx"]
            flux_ens = flux_total[lk]

            # Percentiles
            p2_5  = np.percentile(flux_ens,  2.5, axis=0)
            p16   = np.percentile(flux_ens, 16,   axis=0)
            p50   = np.percentile(flux_ens, 50,   axis=0)
            p84   = np.percentile(flux_ens, 84,   axis=0)
            p97_5 = np.percentile(flux_ens, 97.5, axis=0)

            # Observed spectrum in velocity space (with optional binning)
            obs_vel = vel
            obs_flux = spec_flux_arr[tight_idx]
            obs_err = spec_err_arr[tight_idx] if spec_err_arr is not None else None

            # Determine per-line bin factor
            _n_bin = 1
            if isinstance(binning, dict):
                _n_bin = int(binning.get(lk, 1))
            elif binning is not None:
                _n_bin = int(binning)
            _n_bin = max(_n_bin, 1)

            if _n_bin > 1 and len(obs_vel) >= _n_bin:
                remainder = len(obs_vel) % _n_bin
                trimmed = len(obs_vel) - remainder
                if trimmed > 0:
                    obs_vel = np.mean(obs_vel[remainder:remainder + trimmed].reshape(-1, _n_bin), axis=1)
                    obs_flux = np.mean(obs_flux[remainder:remainder + trimmed].reshape(-1, _n_bin), axis=1)
                    if obs_err is not None:
                        obs_err = np.sqrt(
                            np.sum(obs_err[remainder:remainder + trimmed].reshape(-1, _n_bin) ** 2, axis=1)
                        ) / _n_bin

            ax.step(obs_vel, obs_flux,
                    where="mid", color="black", linewidth=0.8, label="Observed")
            if obs_err is not None:
                ax.fill_between(
                    obs_vel,
                    obs_flux - obs_err,
                    obs_flux + obs_err,
                    color="gray", alpha=0.2, step="mid",
                )

            # Posterior bands
            ax.fill_between(vel, p2_5, p97_5, color="red",
                            alpha=band_alpha * 0.6, label="95%")
            ax.fill_between(vel, p16, p84, color="red",
                            alpha=band_alpha, label="68%")
            ax.plot(vel, p50, color="red", linewidth=1.2, label="Median")

            # Per-component median profiles
            if show_components:
                for ci in range(len(self.grids)):
                    comp_id = self.component_ids[ci]
                    cflux = flux_comp_dict.get((lk, ci))
                    if cflux is None:
                        continue
                    c_p2_5 = np.percentile(cflux, 2.5, axis=0)
                    c_p16 = np.percentile(cflux, 16, axis=0)
                    c_med = np.percentile(cflux, 50, axis=0)
                    c_p84 = np.percentile(cflux, 84, axis=0)
                    c_p97_5 = np.percentile(cflux, 97.5, axis=0)
                    cc = _comp_colors[ci % len(_comp_colors)]
                    ax.fill_between(
                        vel,
                        c_p2_5,
                        c_p97_5,
                        color=cc,
                        alpha=max(0.08, band_alpha * 0.20),
                    )
                    ax.fill_between(
                        vel,
                        c_p16,
                        c_p84,
                        color=cc,
                        alpha=max(0.12, band_alpha * 0.35),
                    )
                    ax.plot(vel, c_med, color=cc, linewidth=0.8, linestyle=":",
                            label=f"Comp {comp_id}", alpha=0.7)

            # Formatting
            ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
            ax.axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
            ax.axvline(0.0, color="gray", linewidth=0.5, linestyle=":")
            ax.set_xlim(-vrange, vrange)
            ax.set_ylim(-0.1, 1.5)
            ion_short = ion.replace("N_", "")
            _, lam0_val, _, _, _ = _line_cache[lk]
            ax.set_ylabel("Normalised Flux")
            ax.set_title(f"{ion_short}  {lk}  ({lam0_val:.1f} Ang)", fontsize=10)
            if panel_idx == 0:
                ax.legend(fontsize=8, loc="upper right", ncol=3)

        axes_flat[-1].set_xlabel("Velocity (km/s)")
        fig.tight_layout()
        return fig

    def ion_distribution_plot(
        self,
        ions: Optional[Sequence[str]] = None,
        *,
        thin: int = 1,
        max_samples: Optional[int] = 2000,
        n_draw_lines: int = 300,
        include_total: bool = True,
        component_colors: Optional[Sequence[str]] = None,
        total_color: str = "black",
        prior_color: str = "dimgray",
        figsize: Tuple[float, float] = (14.0, 4.8),
        random_seed: Optional[int] = 42,
        title: Optional[str] = None,
    ):
        """Plot posterior ion-column ladders for each Cloudy component.

        The figure samples the fitted Cloudy posterior and draws faint
        sample-wise ion ladders for each component, plus median and 16-84%
        envelopes. Optionally, the summed (total) column-density ladder is
        overplotted. If ``config.total_column_density_bounds`` is set, prior
        bounds are drawn for each constrained ion as short horizontal bands at
        the corresponding ion position.

        Parameters
        ----------
        ions : sequence of str, optional
            Ion names in Cloudy notation (e.g. ``["N_CII", "N_CIII"]``).
            Default: inferred from ``ions_per_component`` plus any ions in
            ``config.total_column_density_bounds``.
        thin : int
            Keep every ``thin``-th posterior sample.
        max_samples : int, optional
            Cap number of posterior samples used in this plot.
        n_draw_lines : int
            Number of faint posterior ladders to draw per component.
        include_total : bool
            Also draw the total (sum across components, in linear space).
        component_colors : sequence of str, optional
            Colors for components. Defaults to matplotlib ``tab10`` cycle.
        total_color : str
            Color used for total ladder statistics.
        prior_color : str
            Color used for total-column prior overlays.
        figsize : tuple
            Matplotlib figure size.
        random_seed : int, optional
            RNG seed for reproducible posterior subsampling.
        title : str, optional
            Optional figure title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.samples is None:
            raise RuntimeError("Run MCMC before calling ion_distribution_plot")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib required for ion_distribution_plot") from exc

        rng = np.random.default_rng(random_seed)

        # Thin/cap posterior samples (expand to full space).
        full_samples = self._expand_samples(self.samples)
        idx = np.arange(0, full_samples.shape[0], max(thin, 1))
        if max_samples is not None and max_samples > 0 and len(idx) > max_samples:
            idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
        base_samples = full_samples[idx]
        n_pts = base_samples.shape[0]
        if n_pts == 0:
            raise RuntimeError("No posterior samples available after thinning.")

        # Ion order: preserve first appearance across components, then append
        # any prior-constrained ions not already present.
        if ions is None:
            ion_order: List[str] = []
            for ion_list in self.ions_per_component:
                for ion in ion_list:
                    if ion not in ion_order:
                        ion_order.append(ion)
            if self.config.total_column_density_bounds:
                for ion in self.config.total_column_density_bounds.keys():
                    if ion not in ion_order:
                        ion_order.append(ion)
        else:
            ion_order = list(ions)

        if not ion_order:
            raise ValueError("No ions available to plot.")

        n_ions = len(ion_order)
        n_comp = len(self.grids)
        is_single_component = n_comp == 1

        # Component colors.
        if component_colors is None:
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i % 10) for i in range(n_comp)]
        else:
            colors = [component_colors[i % len(component_colors)] for i in range(n_comp)]

        # Evaluate ion columns for each component across posterior samples.
        per_comp_matrix: List[np.ndarray] = []
        for ci in range(n_comp):
            grid = self.grids[ci]
            p_order = self.component_param_templates[ci]
            comp_samples = base_samples[:, self.param_slices[ci]]
            batch = self._batch_evaluate_grid(ci, base_samples, ions=ion_order)

            mat = np.full((n_pts, n_ions), np.nan, dtype=float)
            for j, ion in enumerate(ion_order):
                if ion not in batch:
                    continue
                col = batch[ion].copy()
                col = self._apply_abundance_offsets_batch(
                    ci, ion, col, comp_samples, p_order,
                )
                mat[:, j] = col
            per_comp_matrix.append(mat)

        # Optional total matrix in log-space after linear summation.
        total_mat: Optional[np.ndarray] = None
        if include_total:
            total_lin = np.zeros((n_pts, n_ions), dtype=float)
            any_valid = np.zeros((n_pts, n_ions), dtype=bool)
            for mat in per_comp_matrix:
                valid = np.isfinite(mat)
                if np.any(valid):
                    total_lin[valid] += 10.0 ** mat[valid]
                    any_valid |= valid
            total_mat = np.full((n_pts, n_ions), np.nan, dtype=float)
            ok = any_valid & (total_lin > 0.0)
            total_mat[ok] = np.log10(total_lin[ok])

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        x = np.arange(n_ions, dtype=float)

        # Draw per-component sample ladders + summary stats.
        for ci, mat in enumerate(per_comp_matrix):
            comp_id = self.component_ids[ci]
            color = colors[ci]

            valid_rows = np.where(np.any(np.isfinite(mat), axis=1))[0]
            if valid_rows.size > 0 and n_draw_lines > 0:
                draw_n = min(n_draw_lines, valid_rows.size)
                draw_idx = rng.choice(valid_rows, size=draw_n, replace=False)
                line_alpha = 0.16 if is_single_component else 0.08
                for ri in draw_idx:
                    ax.plot(x, mat[ri], color=color, alpha=line_alpha, linewidth=0.8, zorder=1)

            med = np.nanmedian(mat, axis=0)
            p16 = np.nanpercentile(mat, 16.0, axis=0)
            p84 = np.nanpercentile(mat, 84.0, axis=0)
            ax.fill_between(x, p16, p84, color=color, alpha=0.18, zorder=2)
            ax.plot(
                x,
                med,
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=4.0,
                label=f"Component {comp_id}",
                zorder=3,
            )

        # Draw total ladder summary.
        if include_total and total_mat is not None:
            med_t = np.nanmedian(total_mat, axis=0)
            p16_t = np.nanpercentile(total_mat, 16.0, axis=0)
            p84_t = np.nanpercentile(total_mat, 84.0, axis=0)
            total_fill_alpha = 0.04 if is_single_component else 0.10
            total_line_alpha = 0.85 if is_single_component else 1.0
            total_line_style = "--" if is_single_component else "-"
            total_line_width = 1.8 if is_single_component else 2.8
            total_label = "Total (= component)" if is_single_component else "Total"
            ax.fill_between(x, p16_t, p84_t, color=total_color, alpha=total_fill_alpha, zorder=2)
            ax.plot(
                x,
                med_t,
                color=total_color,
                linewidth=total_line_width,
                linestyle=total_line_style,
                alpha=total_line_alpha,
                marker="D",
                markersize=4.5,
                label=total_label,
                zorder=4,
            )

        # Prior overlays: show each constrained ion as a local horizontal band.
        prior_cfg = self.config.total_column_density_bounds or {}
        prior_legend_added = False
        for j, ion in enumerate(ion_order):
            if ion not in prior_cfg:
                continue
            lo, hi = prior_cfg[ion]
            x0 = j - 0.34
            x1 = j + 0.34
            ax.fill_between([x0, x1], [lo, lo], [hi, hi], color=prior_color, alpha=0.14, zorder=5)
            ax.hlines(
                [lo, hi],
                x0,
                x1,
                colors=prior_color,
                linestyles="--",
                linewidth=1.2,
                zorder=6,
                label="Total column prior" if not prior_legend_added else None,
            )
            prior_legend_added = True

        xlabels = [ion.replace("N_", "") for ion in ion_order]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=11)
        ax.set_ylabel(r"log($N_{ion}$ / cm$^{-2}$)", fontsize=12)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=10, frameon=True)

        if title is None:
            title = "Posterior Ion Distribution by Component"
        ax.set_title(title, fontsize=13)
        fig.tight_layout()
        return fig

    def total_parameters_plot(
        self,
        ions: Optional[Sequence[str]] = None,
        *,
        include_total_columns: bool = True,
        include_total_length: bool = True,
        include_total_gas_mass: bool = True,
        include_total_element_masses: bool = True,
        include_total_ion_masses: bool = True,
        element_symbols: Optional[Sequence[str]] = None,
        thin: int = 1,
        max_samples: Optional[int] = 5000,
        bins: int = 40,
        panel_size: Tuple[float, float] = (4.0, 3.0),
        ncols: int = 4,
        title: Optional[str] = None,
    ):
        """Plot totals-only posterior summaries for summed physical quantities.

        This figure includes only quantities summed across all fitted components,
        such as total ion columns, total cloud length, total gas mass, total
        element masses, and total ion masses. For total ion columns, priors
        (``config.total_column_density_bounds``) and observation limits
        (``self.observations``) are overlaid when available.

        Notes
        -----
        Total ion masses are computed with the same geometric scaling used in
        the cloud-mass proxies, i.e. per component:
            log M_ion ~ log N_ion + 2*(log N_H - log n_H) + log(m_element) - log(M_sun)
        and then summed in linear space across components.
        """
        if self.samples is None:
            raise RuntimeError("Run MCMC before calling total_parameters_plot")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib required for total_parameters_plot") from exc

        if thin < 1:
            raise ValueError("thin must be >= 1")

        full_samples = self._expand_samples(self.samples)
        idx = np.arange(0, full_samples.shape[0], thin, dtype=int)
        if idx.size == 0:
            raise RuntimeError("No posterior samples available after thinning")
        if max_samples is not None and max_samples > 0 and idx.size > max_samples:
            rng = np.random.default_rng()
            idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
        base_samples = full_samples[idx]
        n_pts = base_samples.shape[0]

        # Ion list defaults to union of modeled ions, observation ions, and prior-bounded ions.
        if ions is None:
            ion_order: List[str] = []
            for ion_list in self.ions_per_component:
                for ion in ion_list:
                    if ion not in ion_order:
                        ion_order.append(ion)
            for ion in self.observations.keys():
                if ion not in ion_order:
                    ion_order.append(ion)
            if self.config.total_column_density_bounds:
                for ion in self.config.total_column_density_bounds.keys():
                    if ion not in ion_order:
                        ion_order.append(ion)
        else:
            ion_order = list(ions)

        # Determine which element totals to show.
        if element_symbols is not None:
            elem_list = [_normalise_element_symbol(e) for e in element_symbols]
        else:
            elem_list = self._constrained_elements()

        # Per-component geometric terms and ion predictions.
        comp_log_n_H: List[np.ndarray] = []
        comp_log_col_for_mass: List[np.ndarray] = []
        comp_ion_cols: List[Dict[str, np.ndarray]] = []

        for ci, grid in enumerate(self.grids):
            p_order = self.component_param_templates[ci]
            comp_samples = base_samples[:, self.param_slices[ci]]

            if "n_H" not in p_order:
                raise ValueError(f"Component {self.component_ids[ci]} missing n_H parameter")
            n_H_idx = p_order.index("n_H")
            log_n_H = comp_samples[:, n_H_idx]

            col_key = "N_H" if "N_H" in p_order else "NHI"
            log_col = comp_samples[:, p_order.index(col_key)]
            log_col_for_mass = log_col.copy()
            if col_key == "NHI" and "N_H_total" in grid.ion_order:
                batch_nh = self._batch_evaluate_grid(ci, base_samples, ions=["N_H_total"])
                if "N_H_total" in batch_nh:
                    log_col_for_mass = batch_nh["N_H_total"]

            batch = self._batch_evaluate_grid(ci, base_samples, ions=ion_order)
            ion_cols_i: Dict[str, np.ndarray] = {}
            for ion in ion_order:
                if ion not in batch:
                    continue
                col_arr = batch[ion].copy()
                col_arr = self._apply_abundance_offsets_batch(ci, ion, col_arr, comp_samples, p_order)
                ion_cols_i[ion] = col_arr

            comp_log_n_H.append(log_n_H)
            comp_log_col_for_mass.append(log_col_for_mass)
            comp_ion_cols.append(ion_cols_i)

        # Totals in linear space.
        total_columns_linear: Dict[str, np.ndarray] = {ion: np.zeros(n_pts) for ion in ion_order}
        for ion in ion_order:
            for ci in range(len(self.grids)):
                col = comp_ion_cols[ci].get(ion)
                if col is not None:
                    total_columns_linear[ion] += 10.0 ** col

        total_length_linear = np.zeros(n_pts)
        total_gas_mass_linear = np.zeros(n_pts)
        for ci in range(len(self.grids)):
            log_L_kpc = comp_log_col_for_mass[ci] - comp_log_n_H[ci] - LOG10_CM_PER_KPC
            log_M_gas = (
                3.0 * comp_log_col_for_mass[ci]
                - 2.0 * comp_log_n_H[ci]
                + LOG10_PROTON_MASS_G
                - LOG10_SOLAR_MASS_G
            )
            total_length_linear += 10.0 ** log_L_kpc
            total_gas_mass_linear += 10.0 ** log_M_gas

        total_element_mass_linear: Dict[str, np.ndarray] = {}
        if include_total_element_masses and elem_list:
            for elem in elem_list:
                parts = np.zeros(n_pts)
                for ci in range(len(self.grids)):
                    p_order = self.component_param_templates[ci]
                    comp_samples = base_samples[:, self.param_slices[ci]]
                    if elem not in SOLAR_LOG_ABUNDANCE or elem not in LOG10_ATOMIC_MASS_G:
                        continue
                    if "Z" not in p_order:
                        continue
                    log_Z = comp_samples[:, p_order.index("Z")]
                    ref = self._per_comp_abundance_reference[ci]
                    if elem == ref:
                        log_X_over_H = SOLAR_LOG_ABUNDANCE[elem] + log_Z
                    else:
                        ratio_key = _build_ratio_key(elem, ref)
                        ab_fixed = self._per_comp_abundance_fixed[ci]
                        if ratio_key in p_order:
                            offset = comp_samples[:, p_order.index(ratio_key)]
                        else:
                            offset = np.full(n_pts, ab_fixed.get(ratio_key, 0.0))
                        log_X_over_H = SOLAR_LOG_ABUNDANCE[elem] + log_Z + offset

                    log_Me = (
                        log_X_over_H
                        + LOG10_ATOMIC_MASS_G[elem]
                        + 3.0 * comp_log_col_for_mass[ci]
                        - 2.0 * comp_log_n_H[ci]
                        - LOG10_SOLAR_MASS_G
                    )
                    parts += 10.0 ** log_Me
                total_element_mass_linear[elem] = parts

        total_ion_mass_linear: Dict[str, np.ndarray] = {}
        if include_total_ion_masses:
            for ion in ion_order:
                elem = ION_TO_ELEMENT.get(ion)
                if elem is None or elem not in LOG10_ATOMIC_MASS_G:
                    continue
                parts = np.zeros(n_pts)
                for ci in range(len(self.grids)):
                    col = comp_ion_cols[ci].get(ion)
                    if col is None:
                        continue
                    log_Mion = (
                        col
                        + 2.0 * (comp_log_col_for_mass[ci] - comp_log_n_H[ci])
                        + LOG10_ATOMIC_MASS_G[elem]
                        - LOG10_SOLAR_MASS_G
                    )
                    parts += 10.0 ** log_Mion
                total_ion_mass_linear[ion] = parts

        # Assemble panel definitions.
        panels: List[Dict[str, Any]] = []
        if include_total_columns:
            for ion in ion_order:
                total_lin = total_columns_linear.get(ion)
                if total_lin is None:
                    continue
                vals = np.log10(np.clip(total_lin, 1e-300, None))
                panels.append(
                    {
                        "label": f"log N({ion.replace('N_', '')}) total",
                        "values": vals,
                        "prior_bounds": None if self.config.total_column_density_bounds is None else self.config.total_column_density_bounds.get(ion),
                        "obs": self.observations.get(ion),
                    }
                )

        if include_total_length:
            panels.append(
                {
                    "label": r"log $\Sigma$L (kpc)",
                    "values": np.log10(np.clip(total_length_linear, 1e-300, None)),
                    "prior_bounds": self.config.total_length_bounds,
                    "obs": None,
                }
            )

        if include_total_gas_mass:
            panels.append(
                {
                    "label": r"log $\Sigma$M$_{gas}$ (M$_\odot$)",
                    "values": np.log10(np.clip(total_gas_mass_linear, 1e-300, None)),
                    "prior_bounds": self.config.total_mass_bounds,
                    "obs": None,
                }
            )

        if include_total_element_masses:
            for elem in elem_list:
                vals_lin = total_element_mass_linear.get(elem)
                if vals_lin is None:
                    continue
                eb = None
                if self.config.element_mass_bounds is not None:
                    eb = self.config.element_mass_bounds.get(elem)
                panels.append(
                    {
                        "label": f"log $\\Sigma$M$_{{{elem}}}$ (M$_\\odot$)",
                        "values": np.log10(np.clip(vals_lin, 1e-300, None)),
                        "prior_bounds": eb,
                        "obs": None,
                    }
                )

        if include_total_ion_masses:
            for ion in ion_order:
                vals_lin = total_ion_mass_linear.get(ion)
                if vals_lin is None:
                    continue
                panels.append(
                    {
                        "label": f"log $\\Sigma$M$_{{{ion.replace('N_', '')}}}$ (M$_\\odot$)",
                        "values": np.log10(np.clip(vals_lin, 1e-300, None)),
                        "prior_bounds": None,
                        "obs": None,
                    }
                )

        if not panels:
            raise RuntimeError("No total parameters available to plot")

        n_pan = len(panels)
        ncols_eff = max(1, min(ncols, n_pan))
        nrows = int(np.ceil(n_pan / ncols_eff))
        fig, axes = plt.subplots(
            nrows,
            ncols_eff,
            figsize=(panel_size[0] * ncols_eff, panel_size[1] * nrows),
            squeeze=False,
        )
        ax_flat = axes.ravel()

        for i, panel in enumerate(panels):
            ax = ax_flat[i]
            vals = np.asarray(panel["values"], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                ax.text(0.5, 0.5, "No finite samples", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(panel["label"], fontsize=10)
                continue

            ax.hist(vals, bins=bins, density=True, color="firebrick", alpha=0.55)
            p16, p50, p84 = np.percentile(vals, [16.0, 50.0, 84.0])
            ax.axvline(p50, color="firebrick", linewidth=1.6)
            ax.axvline(p16, color="firebrick", linewidth=1.0, linestyle=":")
            ax.axvline(p84, color="firebrick", linewidth=1.0, linestyle=":")

            pb = panel.get("prior_bounds")
            if pb is not None:
                lo, hi = pb
                ax.axvspan(lo, hi, color="dimgray", alpha=0.12)
                ax.axvline(lo, color="dimgray", linestyle="--", linewidth=1.1)
                ax.axvline(hi, color="dimgray", linestyle="--", linewidth=1.1)

            obs = panel.get("obs")
            if obs is not None:
                if getattr(obs, "is_upper_limit", False):
                    ax.axvline(obs.value, color="black", linestyle="--", linewidth=1.2)
                elif getattr(obs, "is_lower_limit", False):
                    ax.axvline(obs.value, color="black", linestyle="-.", linewidth=1.2)
                else:
                    ax.axvline(obs.value, color="black", linestyle="-", linewidth=1.2)

            ax.set_title(panel["label"], fontsize=10)
            ax.grid(alpha=0.25)

        for j in range(n_pan, len(ax_flat)):
            ax_flat[j].axis("off")

        if title is None:
            title = "Summed Parameters Posterior"
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    # ------------------------------------------------------------------
    # Combined corner plot: physical params + predicted columns w/ Voigt overlay
    # ------------------------------------------------------------------

    def combined_corner_plot(
        self,
        include_derived: bool = True,
        include_temperature: bool = True,
        include_totals: bool = True,
        include_observation_ions: bool = True,
        predicted_ions: Optional[Sequence[str]] = None,
        include_b_total: bool = False,
        b_total_mass_amu: float = 1.008,
        include_kinematics: bool = True,
        jeans_overlay: bool = False,
        jeans_kwargs: Optional[Mapping[str, Any]] = None,
        thin: int = 1,
        max_samples: Optional[int] = None,
        voigt_color: str = "steelblue",
        cloudy_color: str = "firebrick",
        **corner_kwargs,
    ):
        """Corner plot combining Cloudy physical parameters with predicted column
        densities, overlaying the true Voigt posterior where available.

        Along the diagonal, parameters that have a Voigt-profile counterpart
        (column densities, temperature, kinematics) show both the Cloudy-
        predicted histogram and the input Voigt posterior.  In off-diagonal
        panels where **both** axes have Voigt counterparts the 2-D Voigt
        contours are overlaid as well, making it easy to see which ions drive
        the inferred metallicity, density and abundances.

        Parameters
        ----------
        include_derived : bool
            Append cloud path-length *L* and mass *M* per component.
        include_temperature : bool
            Include PIE *T_cloudy* per component (requires grid column).
        include_totals : bool
            Show total |Sigma| L and |Sigma| M when more than one component.
        include_observation_ions : bool
            Append Cloudy-predicted columns for upper/lower-limit ions.
        predicted_ions : list of str, optional
            Extra ions predicted by the grid but not in the GMM.
        include_b_total : bool
            Derived total Doppler parameter per component.
        include_kinematics : bool
            Include pass-through v and b_other parameters.
        jeans_overlay : bool
            Draw Jeans instability boundary on relevant panels.
        thin, max_samples
            Posterior thinning / capping parameters.
        voigt_color : str
            Colour for the Voigt-profile truth overlay (default steelblue).
        cloudy_color : str
            Colour for the Cloudy posterior (default firebrick).
        **corner_kwargs
            Forwarded to :func:`corner.corner`.
        """
        if self.samples is None:
            raise RuntimeError("Run MCMC before calling combined_corner_plot")
        try:
            import corner as corner_pkg
        except ImportError as exc:
            raise RuntimeError("Install 'corner' to create corner plots") from exc
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # ---- subsample / thin -----------------------------------------------
        raw_samples = self._expand_samples(self.samples)
        n_total = raw_samples.shape[0]
        indices = np.arange(0, n_total, max(thin, 1))
        if max_samples is not None and max_samples > 0 and len(indices) > max_samples:
            rng = np.random.default_rng()
            indices = np.sort(rng.choice(indices, size=max_samples, replace=False))
        base_samples = raw_samples[indices]
        n_pts = base_samples.shape[0]

        # Parallel lists: one entry per corner-plot dimension
        plot_columns: List[np.ndarray] = []
        plot_labels: List[str] = []
        voigt_kde_idx: List[Optional[int]] = []   # index into kde dataset, or None

        # Caches for Jeans overlay and derived quantities
        per_comp_T_cloudy: Dict[int, np.ndarray] = {}
        per_comp_L: Dict[int, np.ndarray] = {}
        per_comp_M: Dict[int, np.ndarray] = {}

        _KIN_PARAMS = ("v", "b_other")

        # ==================================================================
        # Group 1: Cloudy physical parameters (no Voigt counterpart)
        # ==================================================================
        for dim_idx, full_name in enumerate(self.param_names):
            parts = full_name.rsplit("_", 1)
            base_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else full_name
            if base_name in _KIN_PARAMS:
                continue
            plot_columns.append(base_samples[:, dim_idx])
            plot_labels.append(self._prettify_param_label(full_name))
            voigt_kde_idx.append(None)

        # ==================================================================
        # Group 2: Per-component predicted column densities (Voigt truth)
        # ==================================================================
        for ci in range(len(self.grids)):
            grid = self.grids[ci]
            comp_id = self.component_ids[ci]
            p_order = self.component_param_templates[ci]
            comp_samples = base_samples[:, self.param_slices[ci]]
            ion_list = self.ions_per_component[ci]

            batch = self._batch_evaluate_grid(ci, base_samples, ions=ion_list)
            for ion in ion_list:
                if ion not in batch:
                    continue
                col = batch[ion].copy()
                col = self._apply_abundance_offsets_batch(
                    ci, ion, col, comp_samples, p_order,
                )
                plot_columns.append(col)
                ion_short = ion.replace("N_", "")
                plot_labels.append(f"log N({ion_short})$_{{{comp_id}}}$")
                # Voigt counterpart
                kde_name = f"{ion}_{comp_id}"
                if kde_name in self.kde_obs.param_names:
                    voigt_kde_idx.append(self.kde_obs.param_names.index(kde_name))
                else:
                    voigt_kde_idx.append(None)

        # ==================================================================
        # Group 3: T_cloudy per component (may have Voigt counterpart)
        # ==================================================================
        if include_temperature:
            for ci in range(len(self.grids)):
                grid = self.grids[ci]
                comp_id = self.component_ids[ci]
                if "T_cloudy" not in grid.ion_order:
                    continue
                batch_t = self._batch_evaluate_grid(ci, base_samples, ions=["T_cloudy"])
                t_arr = batch_t.get("T_cloudy")
                if t_arr is None:
                    continue
                per_comp_T_cloudy[ci] = t_arr
                plot_columns.append(t_arr)
                plot_labels.append(f"log T$_{{{comp_id}}}$ (PIE)")
                t_kde_name = f"T_{comp_id}"
                if t_kde_name in self.kde_obs.param_names:
                    voigt_kde_idx.append(self.kde_obs.param_names.index(t_kde_name))
                else:
                    voigt_kde_idx.append(None)

        # ==================================================================
        # Group 4: Derived L, M per component (no Voigt counterpart)
        # ==================================================================
        if include_derived:
            for ci in range(len(self.grids)):
                grid = self.grids[ci]
                comp_id = self.component_ids[ci]
                p_order = self.component_param_templates[ci]
                comp_samples = base_samples[:, self.param_slices[ci]]

                n_H_idx = p_order.index("n_H")
                col_key = "N_H" if "N_H" in p_order else "NHI"
                col_idx = p_order.index(col_key)

                log_n_H = comp_samples[:, n_H_idx]
                log_col = comp_samples[:, col_idx]

                log_col_for_LM = log_col
                if col_key == "NHI" and "N_H_total" in grid.ion_order:
                    batch_nh = self._batch_evaluate_grid(ci, base_samples, ions=["N_H_total"])
                    if "N_H_total" in batch_nh:
                        log_col_for_LM = batch_nh["N_H_total"]

                log_L_kpc = log_col_for_LM - log_n_H - LOG10_CM_PER_KPC
                log_M_solar = (
                    3.0 * log_col_for_LM - 2.0 * log_n_H
                    + LOG10_PROTON_MASS_G - LOG10_SOLAR_MASS_G
                )
                per_comp_L[ci] = log_L_kpc
                per_comp_M[ci] = log_M_solar

                plot_columns.append(log_L_kpc)
                plot_labels.append(f"log L$_{{{comp_id}}}$ (kpc)")
                voigt_kde_idx.append(None)

                plot_columns.append(log_M_solar)
                plot_labels.append(f"log M$_{{{comp_id}}}$ (M$_\\odot$)")
                voigt_kde_idx.append(None)

                # --- Element masses per component ---
                _constrained_els = self._constrained_elements()
                Z_idx = p_order.index("Z") if "Z" in p_order else None
                for elem in _constrained_els:
                    if elem not in SOLAR_LOG_ABUNDANCE or elem not in LOG10_ATOMIC_MASS_G:
                        continue
                    ref = self._per_comp_abundance_reference[ci]
                    log_Z = comp_samples[:, Z_idx] if Z_idx is not None else np.zeros(n_pts)
                    if elem == ref:
                        log_X_over_H = SOLAR_LOG_ABUNDANCE[elem] + log_Z
                    else:
                        ratio_key = _build_ratio_key(elem, ref)
                        ab_fixed = self._per_comp_abundance_fixed[ci]
                        if ratio_key in p_order:
                            offset = comp_samples[:, p_order.index(ratio_key)]
                        else:
                            offset = ab_fixed.get(ratio_key, 0.0)
                        log_X_over_H = SOLAR_LOG_ABUNDANCE[elem] + log_Z + offset
                    log_Me_solar = (
                        log_X_over_H + LOG10_ATOMIC_MASS_G[elem]
                        + 3.0 * log_col_for_LM - 2.0 * log_n_H
                        - LOG10_SOLAR_MASS_G
                    )
                    per_comp_M[f"_emass_{elem}_{ci}"] = log_Me_solar
                    plot_columns.append(log_Me_solar)
                    plot_labels.append(f"log M$_{{{elem},{comp_id}}}$ (M$_\\odot$)")
                    voigt_kde_idx.append(None)

        # ==================================================================
        # Group 5: Totals across components (no Voigt counterpart)
        # ==================================================================
        if include_totals and include_derived and len(self.grids) > 1:
            total_L_linear = np.zeros(n_pts)
            total_M_linear = np.zeros(n_pts)
            for ci in range(len(self.grids)):
                total_L_linear += 10.0 ** per_comp_L[ci]
                total_M_linear += 10.0 ** per_comp_M[ci]
            plot_columns.append(np.log10(total_L_linear))
            plot_labels.append(r"log $\Sigma$L (kpc)")
            voigt_kde_idx.append(None)
            plot_columns.append(np.log10(total_M_linear))
            plot_labels.append(r"log $\Sigma$M (M$_\odot$)")
            voigt_kde_idx.append(None)

            # Total element masses
            _constrained_els = self._constrained_elements()
            for elem in _constrained_els:
                total_elem_linear = np.zeros(n_pts)
                any_found = False
                for ci in range(len(self.grids)):
                    key = f"_emass_{elem}_{ci}"
                    if key in per_comp_M:
                        total_elem_linear += 10.0 ** per_comp_M[key]
                        any_found = True
                if any_found:
                    log_total_elem = np.log10(np.clip(total_elem_linear, 1e-300, None))
                    plot_columns.append(log_total_elem)
                    plot_labels.append(f"log $\\Sigma$M$_{{{elem}}}$ (M$_\\odot$)")
                    voigt_kde_idx.append(None)

        # ==================================================================
        # Group 6: Observation-ion predictions (no Voigt counterpart)
        # ==================================================================
        obs_truths_map: Dict[str, float] = {}
        if include_observation_ions and self.observations:
            for obs_ion, obs_obj in self.observations.items():
                total_linear = np.zeros(n_pts)
                any_found = False
                for ci in range(len(self.grids)):
                    batch = self._batch_evaluate_grid(ci, base_samples, ions=[obs_ion])
                    if obs_ion in batch:
                        col = batch[obs_ion].copy()
                        p_order = self.component_param_templates[ci]
                        comp_samp = base_samples[:, self.param_slices[ci]]
                        col = self._apply_abundance_offsets_batch(
                            ci, obs_ion, col, comp_samp, p_order,
                        )
                        total_linear += 10.0 ** col
                        any_found = True
                if any_found:
                    log_total = np.log10(np.clip(total_linear, 1e-40, None))
                    plot_columns.append(log_total)
                    suffix = " (UL)" if obs_obj.is_upper_limit else (" (LL)" if obs_obj.is_lower_limit else "")
                    lbl = f"log N({obs_ion.replace('N_', '')}){suffix}"
                    plot_labels.append(lbl)
                    voigt_kde_idx.append(None)
                    obs_truths_map[lbl] = obs_obj.value

        # ==================================================================
        # Group 7: Predicted-only ion columns per component (no Voigt counterpart)
        # ==================================================================
        if predicted_ions:
            for pred_ion in predicted_ions:
                for ci in range(len(self.grids)):
                    comp_id = self.component_ids[ci]
                    batch = self._batch_evaluate_grid(ci, base_samples, ions=[pred_ion])
                    if pred_ion in batch:
                        col = batch[pred_ion].copy()
                        p_order = self.component_param_templates[ci]
                        comp_samp = base_samples[:, self.param_slices[ci]]
                        col = self._apply_abundance_offsets_batch(
                            ci, pred_ion, col, comp_samp, p_order,
                        )
                        plot_columns.append(col)
                        ion_short = pred_ion.replace('N_', '')
                        plot_labels.append(f"log N({ion_short})$_{{{comp_id}}}$ [pred]")
                        voigt_kde_idx.append(None)

        # ==================================================================
        # Group 7.5: Derived cross-ratio abundance constraints
        # ==================================================================
        if self._has_abundance_constraints():
            for ci in range(len(self.grids)):
                comp_id = self.component_ids[ci]
                constraints = self._per_comp_abundance_constraints[ci]
                if not constraints:
                    continue
                p_order = self.component_param_templates[ci]
                comp_samp = base_samples[:, self.param_slices[ci]]
                ref = self._per_comp_abundance_reference[ci]
                ab_fixed_ci = self._per_comp_abundance_fixed[ci]

                for ratio_key, num_el, den_el, is_fixed, fixed_val, bounds_val in constraints:
                    if is_fixed:
                        continue

                    if num_el == ref:
                        num_arr = np.zeros(n_pts)
                    else:
                        nk = _build_ratio_key(num_el, ref)
                        if nk in p_order:
                            num_arr = comp_samp[:, p_order.index(nk)]
                        else:
                            num_arr = np.full(n_pts, ab_fixed_ci.get(nk, 0.0))

                    if den_el == ref:
                        den_arr = np.zeros(n_pts)
                    else:
                        dk = _build_ratio_key(den_el, ref)
                        if dk in p_order:
                            den_arr = comp_samp[:, p_order.index(dk)]
                        else:
                            den_arr = np.full(n_pts, ab_fixed_ci.get(dk, 0.0))

                    derived_ratio = num_arr - den_arr
                    plot_columns.append(derived_ratio)
                    plot_labels.append(
                        f"log {num_el}/{den_el}$_{{{comp_id}}}$ [constr]"
                    )
                    voigt_kde_idx.append(None)

        # ==================================================================
        # Group 8: Kinematic pass-through (Voigt counterpart if in KDE)
        # ==================================================================
        if include_kinematics:
            for ci in range(len(self.grids)):
                comp_id = self.component_ids[ci]
                p_order = self.component_param_templates[ci]
                comp_samples = base_samples[:, self.param_slices[ci]]
                for kin_param in ("v", "b_other"):
                    if kin_param not in p_order:
                        continue
                    kin_idx = p_order.index(kin_param)
                    plot_columns.append(comp_samples[:, kin_idx])
                    plot_labels.append(f"{kin_param}$_{{{comp_id}}}$")
                    kin_kde_name = f"{kin_param}_{comp_id}"
                    if kin_kde_name in self.kde_obs.param_names:
                        voigt_kde_idx.append(
                            self.kde_obs.param_names.index(kin_kde_name)
                        )
                    else:
                        voigt_kde_idx.append(None)

        # ==================================================================
        # Group 9: b_total (derived, no direct Voigt counterpart)
        # ==================================================================
        if include_b_total and include_kinematics and self.has_kinematics_in_likelihood:
            _B_CONST = 0.12845
            for ci in range(len(self.grids)):
                comp_id = self.component_ids[ci]
                p_order = self.component_param_templates[ci]
                comp_samples = base_samples[:, self.param_slices[ci]]
                if "b_other" not in p_order:
                    continue
                b_other_vals = comp_samples[:, p_order.index("b_other")]
                if ci in per_comp_T_cloudy:
                    log_T = per_comp_T_cloudy[ci]
                elif "T" in p_order:
                    log_T = comp_samples[:, p_order.index("T")]
                else:
                    continue
                T_lin = 10.0 ** log_T
                b_thermal = _B_CONST * np.sqrt(T_lin) / np.sqrt(b_total_mass_amu)
                b_total = np.sqrt(b_thermal ** 2 + b_other_vals ** 2)
                plot_columns.append(b_total)
                plot_labels.append(f"b$_{{total,{comp_id}}}$ (km/s)")
                voigt_kde_idx.append(None)

        # ==================================================================
        # Assemble plot_samples and filter non-finite rows
        # ==================================================================
        n_dim = len(plot_columns)
        plot_samples = np.column_stack(plot_columns)
        finite_mask = np.all(np.isfinite(plot_samples), axis=1)
        if not np.all(finite_mask):
            n_dropped = int(n_pts - np.count_nonzero(finite_mask))
            logger.warning(
                "combined_corner_plot: dropping %d non-finite samples", n_dropped
            )
            plot_samples = plot_samples[finite_mask]
            for key in per_comp_T_cloudy:
                per_comp_T_cloudy[key] = per_comp_T_cloudy[key][finite_mask]
        n_pts = plot_samples.shape[0]
        if n_pts == 0:
            raise RuntimeError("All samples non-finite; check grid completeness.")

        has_voigt = [v is not None for v in voigt_kde_idx]

        # ==================================================================
        # Build Voigt truth samples array
        # ==================================================================
        voigt_data = self.kde_obs.dataset.copy()
        n_voigt = voigt_data.shape[0]
        rng_v = np.random.default_rng(42)
        # Cap Voigt samples for performance
        cap = max(n_pts * 2, 50000)
        if n_voigt > cap:
            vi = rng_v.choice(n_voigt, size=cap, replace=False)
            voigt_data = voigt_data[vi]
            n_voigt = voigt_data.shape[0]

        voigt_samples = np.empty((n_voigt, n_dim))
        for d in range(n_dim):
            if voigt_kde_idx[d] is not None:
                voigt_samples[:, d] = voigt_data[:, voigt_kde_idx[d]]
            else:
                lo = float(np.percentile(plot_samples[:, d], 0.5))
                hi = float(np.percentile(plot_samples[:, d], 99.5))
                pad = max(0.1 * (hi - lo), 0.2)
                voigt_samples[:, d] = rng_v.uniform(lo - pad, hi + pad, size=n_voigt)

        # ==================================================================
        # Shared axis ranges
        # ==================================================================
        _combined = np.vstack([plot_samples, voigt_samples])
        _shared_range = []
        for d in range(n_dim):
            lo = float(np.percentile(_combined[:, d], 0.25))
            hi = float(np.percentile(_combined[:, d], 99.75))
            _shared_range.append((lo, hi))

        # ==================================================================
        # Truths array (observation-ion limit values)
        # ==================================================================
        truths = [None] * n_dim
        for lbl, lim_val in obs_truths_map.items():
            if lbl in plot_labels:
                truths[plot_labels.index(lbl)] = lim_val

        # ==================================================================
        # Draw corner plots
        # ==================================================================
        # Build per-band RGBA color lists with graded alpha:
        #   band 0 (outside 3σ):  transparent
        #   band 1 (3σ → 2σ):    line only (transparent fill)
        #   band 2 (2σ → 1σ):    lighter shade
        #   band 3 (inside 1σ):   stronger shade
        # levels (0.68, 0.95, 0.99) → 4 contourf bands
        def _graded_colors(base_color, alphas=(0.0, 0.0, 0.20, 0.35)):
            rgba = list(mcolors.to_rgba(base_color))
            return [[rgba[0], rgba[1], rgba[2], a] for a in alphas]

        _voigt_fill_colors = _graded_colors(voigt_color)
        _cloudy_fill_colors = _graded_colors(cloudy_color)

        common_kwargs = dict(
            plot_datapoints=False,
            plot_density=False,
            plot_contours=True,
            fill_contours=True,
            no_fill_contours=True,
            levels=(0.68, 0.95, 0.99),
            bins=25,
        )
        common_kwargs.update(corner_kwargs)

        # --- Voigt truth (draw first so Cloudy is on top) ---
        fig = corner_pkg.corner(
            voigt_samples,
            labels=plot_labels,
            color=voigt_color,
            range=_shared_range,
            contour_kwargs={"linestyles": "-", "linewidths": 0.8},
            contourf_kwargs={"colors": _voigt_fill_colors},
            hist_kwargs={"density": True, "color": voigt_color, "alpha": 0.4},
            **common_kwargs,
        )

        # Snapshot Voigt collections so we can identify them later
        _voigt_fill_collections = set()
        for ax in fig.axes:
            for coll in ax.collections:
                _voigt_fill_collections.add(id(coll))

        # --- Cloudy posterior (overlay) ---
        corner_pkg.corner(
            plot_samples,
            color=cloudy_color,
            fig=fig,
            truths=truths,
            range=_shared_range,
            contour_kwargs={"linestyles": "-", "linewidths": 1.0},
            contourf_kwargs={"colors": _cloudy_fill_colors},
            hist_kwargs={"density": True, "color": cloudy_color, "alpha": 0.5},
            **common_kwargs,
        )

        # ==================================================================
        # Clean up: remove Voigt artists from panels without counterparts
        # ==================================================================
        axes = np.array(fig.axes).reshape(n_dim, n_dim)
        voigt_rgba = np.array(mcolors.to_rgba(voigt_color))

        for d_row in range(n_dim):
            for d_col in range(d_row + 1):
                if d_row == d_col:
                    # Diagonal: remove Voigt histogram if no counterpart
                    if not has_voigt[d_row]:
                        ax = axes[d_row, d_row]
                        import matplotlib.patches as mpatches
                        for patch in list(ax.patches):
                            fc = np.asarray(mcolors.to_rgba(patch.get_facecolor())).ravel()[:4]
                            if np.allclose(fc[:3], voigt_rgba[:3], atol=0.1):
                                patch.remove()
                        for line in list(ax.lines):
                            try:
                                lc = np.asarray(mcolors.to_rgba(line.get_color())).ravel()[:4]
                            except Exception:
                                continue
                            if np.allclose(lc[:3], voigt_rgba[:3], atol=0.1):
                                line.remove()
                else:
                    # Off-diagonal: remove Voigt contours + fills unless BOTH
                    # axes have a Voigt counterpart
                    if not (has_voigt[d_row] and has_voigt[d_col]):
                        ax = axes[d_row, d_col]
                        for coll in list(ax.collections):
                            if id(coll) not in _voigt_fill_collections:
                                continue
                            # Check edge color (line contour) or face color (fill)
                            ec = np.asarray(coll.get_edgecolors())
                            fc = np.asarray(coll.get_facecolors())
                            is_voigt = False
                            if ec.size and np.allclose(ec.ravel()[:3], voigt_rgba[:3], atol=0.1):
                                is_voigt = True
                            if fc.size and np.allclose(fc.ravel()[:3], voigt_rgba[:3], atol=0.1):
                                is_voigt = True
                            if is_voigt:
                                coll.remove()

        # ==================================================================
        # Compact diagonal titles
        # ==================================================================
        for d in range(n_dim):
            ax_diag = axes[d, d]
            ax_diag.set_title(plot_labels[d], fontsize=7, pad=4)
            ax_diag.tick_params(axis="x", top=True, labeltop=True,
                                bottom=True, labelbottom=True)

        # ==================================================================
        # Observation-ion truth lines (blue dashed)
        # ==================================================================
        for lbl, lim_val in obs_truths_map.items():
            if lbl not in plot_labels:
                continue
            dim = plot_labels.index(lbl)
            ax_diag = axes[dim, dim]
            ax_diag.axvline(lim_val, color="blue", linewidth=1.5,
                            linestyle="--", zorder=10)
            for row in range(dim + 1, n_dim):
                axes[row, dim].axvline(lim_val, color="blue",
                                       linewidth=0.8, linestyle="--", alpha=0.5)
            for col in range(dim):
                axes[dim, col].axhline(lim_val, color="blue",
                                       linewidth=0.8, linestyle="--", alpha=0.5)

        # ==================================================================
        # Colour special-panel diagonal histograms
        # ==================================================================
        try:
            import matplotlib.patches as mpatches
            for d in range(n_dim):
                lbl = plot_labels[d]
                if "[pred]" in lbl:
                    c = "#de2d26"
                elif "(UL)" in lbl or "(LL)" in lbl:
                    c = "#000000"
                else:
                    continue
                ax = axes[d, d]
                for r in ax.patches:
                    fc = np.asarray(mcolors.to_rgba(r.get_facecolor())).ravel()[:3]
                    if np.allclose(fc, np.asarray(mcolors.to_rgba(cloudy_color))[:3], atol=0.1):
                        r.set_facecolor(c)
                        r.set_edgecolor("none")
                        r.set_alpha(0.6)
        except Exception:
            pass

        # ==================================================================
        # Legend
        # ==================================================================
        try:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=cloudy_color, lw=2, label="Cloudy posterior"),
                Line2D([0], [0], color=voigt_color, lw=2, label="Voigt posterior"),
            ]
            if obs_truths_map:
                legend_elements.append(
                    Line2D([0], [0], color="blue", lw=1.5, ls="--", label="Obs limit")
                )
            fig.legend(handles=legend_elements, loc="upper right", fontsize=10,
                       frameon=True, framealpha=0.8)
        except Exception:
            pass

        # ==================================================================
        # Jeans overlay
        # ==================================================================
        if jeans_overlay and include_derived:
            self._overlay_jeans_boundary_joint(
                fig,
                plot_labels,
                plot_samples,
                per_comp_T_cloudy,
                **(dict(jeans_kwargs) if jeans_kwargs else {}),
            )

        return fig

    def debug_highlight_ovi(self, output_path="debug_ovi_highlight.png"):
        """
        Highlight samples where 13.5 < logN(OVI) < 14.5 for component 1 (index 1).
        Plot these on the parameter corner plot.
        """
        import corner
        import matplotlib.pyplot as plt

        # Identify component index for ID 1
        target_comp_id = 1
        if target_comp_id not in self.component_ids:
            print(f"Component ID {target_comp_id} not found. Skipping OVI highlight.")
            return

        comp_idx = self.component_ids.index(target_comp_id)
        
        # Subsample for speed if needed (expand to full space)
        full_samples = self._expand_samples(self.samples)
        n_max = 5000
        if len(full_samples) > n_max:
            indices = np.random.choice(len(full_samples), size=n_max, replace=False)
            subset = full_samples[indices]
        else:
            subset = full_samples

        highlighted_samples = []
        
        # Scan samples
        print("Scanning samples for high OVI...")
        for theta in subset:
            # We need to map theta to params for component 1
            sl = self.param_slices[comp_idx]
            comp_theta = theta[sl]
            
            # Map array to dict for grid evaluation
            p_temp = self.component_param_templates[comp_idx]
            params = {}
            # Start with z if grid has it
            grid = self.grids[comp_idx]
            if "z" in grid.axis_names: params["z"] = self.config.redshift
            
            # Fill params from theta
            for i, name in enumerate(p_temp):
                params[name] = comp_theta[i]
                
            # Evaluate grid
            # Note: Ion name generally "OVI" or "N_OVI" depending on grid.
            # Usually "OVI" if based on ions_per_component logic.
            # But let's check grid evaluate return keys or just handle both.
            pred = grid.evaluate(params)
        
            # Apply abundance offsets (per-component)
            self._apply_abundance_offsets_for_component(comp_idx, pred, params)
            
            val = pred.get("N_OVI")
            if val is None: val = pred.get("OVI")
            
            if val is not None and 13.5 < val < 14.5:
                highlighted_samples.append(theta)

        highlighted_samples = np.array(highlighted_samples)
        print(f"Found {len(highlighted_samples)} / {len(subset)} samples in OVI range.")

        if len(highlighted_samples) > 0:
            print("\nStatistics for samples with 13.5 < logN(OVI) < 14.5:")
            print(f"{'Parameter':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
            print("-" * 60)
            for i, name in enumerate(self.param_names):
                vals = highlighted_samples[:, i]
                print(f"{name:<15} {np.mean(vals):.2f}      {np.std(vals):.2f}      {np.min(vals):.2f}      {np.max(vals):.2f}")
            print("-" * 60)
            print("Check if 'O_C' (or abundance params) span a wide range. If so, they are breaking the OVI constraint.\n")

        # Plot
        print("Generating debug corner plot...")
        fig = plt.figure(figsize=(15, 15))
        
        # Base contour (all samples)
        corner.corner(
            subset,
            labels=self.param_names,
            fig=fig,
            color="black",
            plot_datapoints=False,
            plot_density=False,
            plot_contours=True,
            no_fill_contours=True,
            range=[0.995] * subset.shape[1],
            levels=(0.68, 0.95),
            hist_kwargs={"density": True},
        )
        
        # Highlighted points (Red scatter)
        if len(highlighted_samples) > 0:
            corner.corner(
                highlighted_samples,
                fig=fig,
                color="red",
                plot_datapoints=True,
                plot_density=False,
                plot_contours=False,
                no_fill_contours=True,
                range=[0.995] * highlighted_samples.shape[1],
                hist_kwargs={"density": True},
                weights=np.ones(len(highlighted_samples)) * (len(subset) / len(highlighted_samples)) # Attempt to normalize histogram scale? corner might handle density automatically
            )
        
        plt.savefig(output_path)
        print(f"Saved debug plot to {output_path}")


# ---- Absorber-grouping helpers -------------------------------------------

def split_group_columns(
    total_log_N: np.ndarray,
    voigt_chain: np.ndarray,
    param_names: List[str],
    ion: str,
    group_absorber_ids: List[int],
) -> Dict[int, np.ndarray]:
    """Split a total (log10) column density back to individual absorbers
    proportionally, using the original Voigt chain fractions.

    Parameters
    ----------
    total_log_N : 1-D array (n_samples,)
        Predicted total log10 column density from the Cloudy fit.
    voigt_chain : 2-D array (n_chain, n_params)
        Full Voigt MCMC chain (linear column densities * 1e-8).
    param_names : list of str
        Parameter names for *voigt_chain* columns.
    ion : str
        Ion name, e.g. ``"N_HI"``.
    group_absorber_ids : list of int
        Voigt absorber IDs that form this group (e.g. ``[0, 1]``).

    Returns
    -------
    dict mapping absorber_id -> 1-D array of log10 column densities.
    """
    n_samp = len(total_log_N)
    total_linear = 10.0 ** total_log_N

    col_indices = []
    for aid in group_absorber_ids:
        name = f"{ion}_{aid}"
        if name not in param_names:
            raise ValueError(f"{name} not found in Voigt chain param_names")
        col_indices.append(param_names.index(name))

    voigt_cols = voigt_chain[:, col_indices]  # linear * 1e-8
    voigt_fracs = voigt_cols / voigt_cols.sum(axis=1, keepdims=True)
    median_fracs = np.nanmedian(voigt_fracs, axis=0)  # (n_absorbers,)
    median_fracs = median_fracs / median_fracs.sum()

    result: Dict[int, np.ndarray] = {}
    for j, aid in enumerate(group_absorber_ids):
        linear_j = total_linear * median_fracs[j]
        result[aid] = np.log10(np.clip(linear_j, 1e-40, None))
    return result


class GMMObservation:
    """
    Wraps a Gaussian Mixture Model (sklearn) for likelihood estimation.
    Drop-in replacement for KDEObservation.
    """

    def __init__(self, samples: ArrayLike, param_names: Sequence[str], n_components: int = 10, max_samples: int = 100_000, covariance_type: str = 'full') -> None:
        """
        :param samples: (N_samples, N_params) array of posterior chain samples
        :param param_names: List of parameter names
        :param n_components: Number of Gaussian components to fit.
        :param max_samples: Maximum number of samples to use for training (randomly subsampled).
        :param covariance_type: 'full' (default), 'tied', 'diag', 'spherical'.
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            raise RuntimeError("scikit-learn is required for GMMObservation")

        self.param_names = list(param_names)
        
        # Subsample if too large
        self.dataset = np.asarray(samples)
        n_total = self.dataset.shape[0]
        if n_total > max_samples:
            indices = np.random.choice(n_total, max_samples, replace=False)
            self.dataset = self.dataset[indices, :]
            
        self.n = self.dataset.shape[0]
        self.d = self.dataset.shape[1]
        
        print(f"Fitting GMM with {n_components} components on {self.n} samples...")
        self.gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
        self.gmm.fit(self.dataset)
        
        self.converged_ = self.gmm.converged_
        print(f"GMM Converged: {self.converged_}")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Pickle the GMMObservation to *path*."""
        import pickle
        path = Path(path)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("GMMObservation saved to %s", path)
        print(f"GMMObservation saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GMMObservation":
        """Load a previously pickled GMMObservation from *path*."""
        import pickle
        path = Path(path)
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected GMMObservation, got {type(obj).__name__}"
            )
        logger.info("GMMObservation loaded from %s", path)
        print(f"GMMObservation loaded from {path}")
        return obj

    def log_prob(self, values: ArrayLike) -> float:
        """Evaluate log likelihood at a single point or array of points."""
        # gmm.score_samples returns log likelihood for each sample
        # Expects (n_samples, n_features)
        
        values = np.asarray(values)
        if values.ndim == 1:
            values_2d = values.reshape(1, -1)
            try:
                # score_samples returns array of shape (n_samples,)
                log_prob = self.gmm.score_samples(values_2d)
                return float(log_prob[0])
            except Exception:
                return -np.inf
        else:
            try:
                return self.gmm.score_samples(values)
            except Exception:
                return np.full(values.shape[0], -np.inf)

    def plot_corner(self, figsize: Tuple[float, float] = (10, 10), **corner_kwargs):
        """Plot the GMM samples (resampled) vs original dataset."""
        try:
            import corner
            import matplotlib.pyplot as plt
        except ImportError:
            raise RuntimeError("corner and matplotlib required for plotting")
            
        fig = plt.figure(figsize=figsize)
        
        # Levels for contours
        levels = (0.68, 0.95, 0.99)
        
        # Plot original samples (True Posterior) in Black
        corner.corner(
            self.dataset, 
            labels=self.param_names, 
            color="black", 
            fig=fig, 
            plot_datapoints=False,
            plot_density=False,
            plot_contours=True, # Explicitly true
            no_fill_contours=True,
            range=[0.995] * self.dataset.shape[1],
            levels=levels,
            contour_kwargs={"linestyles": "-"},
            hist_kwargs={"density": True},
            **corner_kwargs
        )
        
        # Sample from GMM (Red)
        # sample() returns (X, y)
        gmm_samples, _ = self.gmm.sample(n_samples=self.dataset.shape[0])
        
        corner.corner(
            gmm_samples, 
            fig=fig, 
            color="red", 
            plot_datapoints=False,
            plot_density=False,
            plot_contours=True,
            no_fill_contours=True,
            range=[0.995] * gmm_samples.shape[1],
            levels=levels,
            contour_kwargs={"linestyles": "--"},
            hist_kwargs={"density": True},
        )
        
        return fig

    @staticmethod
    def optimize_n_components(data: ArrayLike, max_components: int = 20, plot: bool = True, covariance_type: str = 'full') -> int:
        """
        Iterate over n_components from 1 to max_components, calculate BIC, and plot the result.
        Returns the optimal n_components (minimizing BIC).
        
        :param data: (N_samples, N_features) array
        :param max_components: Maximum number of components to test
        :param plot: Whether to generate a plot
        :param covariance_type: 'full' (default), 'tied', 'diag', 'spherical'
        :return: optimal n_components
        """
        try:
            from sklearn.mixture import GaussianMixture
            import matplotlib.pyplot as plt
        except ImportError:
            raise RuntimeError("scikit-learn and matplotlib are required")

        data = np.asarray(data)
        # Subsample for speed if huge
        if data.shape[0] > 50000:
             indices = np.random.choice(data.shape[0], 20000, replace=False)
             data = data[indices]

        n_components_range = range(1, max_components + 1)
        bics = []
        aics = []

        print(f"Optimizing GMM components (1 to {max_components})...")
        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=42)
            gmm.fit(data)
            bics.append(gmm.bic(data))
            aics.append(gmm.aic(data))
            print(f"  n={n}: BIC={bics[-1]:.2f}")

        optimal_n = n_components_range[np.argmin(bics)]
        print(f"Optimal n_components (BIC): {optimal_n}")

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(n_components_range, bics, label='BIC', marker='o')
            plt.plot(n_components_range, aics, label='AIC', marker='x', alpha=0.6)
            plt.axvline(optimal_n, color='r', linestyle='--', label=f'Optimal BIC ({optimal_n})')
            plt.xlabel('Number of Components')
            plt.ylabel('Score')
            plt.title(f'GMM Model Selection (BIC/AIC) - {covariance_type}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig("gmm_bic_optimization.png")
            print("Saved plot to 'gmm_bic_optimization.png'")
            plt.close()

        return optimal_n

__all__ = [
    "VoigtComponent",
    "Observation",
    "ZScoreObservation",
    "ZScoreGridConfig",
    "KDEObservation",
    "GMMObservation",
    "FitterConfig",
    "CloudyGridInterpolator",
    "CloudyComponentFitter",
    "JointCloudyComponentFitter",
    "build_observations_from_chain",
    "build_zscore_observations_from_chain",
    "load_voigt_component",
]
