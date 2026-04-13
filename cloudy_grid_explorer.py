"""Explore a Cloudy grid as interpolated 2-D heatmaps.

This module builds on :class:`cloudy_voigt_inference.CloudyGridInterpolator`
to produce publication-quality heatmap figures of predicted ion column
densities over any pair of grid axes, with optional contour overlays and
multi-panel tiling over a third axis.

Example
-------
>>> from cloudy_grid_explorer import CloudyGridExplorer
>>> explorer = CloudyGridExplorer.from_table_path(
...     "/data/mishran/cloudy_outputs/HM12/J2135_example/full_grid.dat"
... )
>>> fig, axes = explorer.plot_grid(
...     x_axis="n_H", y_axis="Z", panel_axis="N_H",
...     ion="N_CIII", contour_levels=[12, 13, 14],
... )
>>> fig.savefig("ciii_heatmap_grid.pdf")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator

from cloudy_voigt_inference import CloudyGridInterpolator

# numpy / astropy compatibility shim (same as other VAPORS modules)
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: np.ndarray.item(a)  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pretty axis labels (internal name -> LaTeX)
# ---------------------------------------------------------------------------

AXIS_LABELS: Dict[str, str] = {
    "z": r"$z$",
    "Z": r"$\log\,Z\;[\mathrm{Z}_\odot]$",
    "n_H": r"$\log\,n_{\mathrm{H}}\;[\mathrm{cm}^{-3}]$",
    "N_H": r"$\log\,N_{\mathrm{H}}\;[\mathrm{cm}^{-2}]$",
    "NHI": r"$\log\,N_{\mathrm{HI}}\;[\mathrm{cm}^{-2}]$",
    "T": r"$\log\,T\;[\mathrm{K}]$",
}

ION_LABELS: Dict[str, str] = {
    "N_HI": r"$\log\,N_{\mathrm{HI}}$",
    "N_H_total": r"$\log\,N_{\mathrm{H,\,total}}$",
    "N_CII": r"$\log\,N_{\mathrm{CII}}$",
    "N_CIII": r"$\log\,N_{\mathrm{CIII}}$",
    "N_CIV": r"$\log\,N_{\mathrm{CIV}}$",
    "N_OII": r"$\log\,N_{\mathrm{OII}}$",
    "N_OIII": r"$\log\,N_{\mathrm{OIII}}$",
    "N_OIV": r"$\log\,N_{\mathrm{OIV}}$",
    "N_OVI": r"$\log\,N_{\mathrm{OVI}}$",
    "N_NII": r"$\log\,N_{\mathrm{NII}}$",
    "N_NIII": r"$\log\,N_{\mathrm{NIII}}$",
    "N_NV": r"$\log\,N_{\mathrm{NV}}$",
    "N_SiII": r"$\log\,N_{\mathrm{SiII}}$",
    "N_SiIII": r"$\log\,N_{\mathrm{SiIII}}$",
    "N_SiIV": r"$\log\,N_{\mathrm{SiIV}}$",
    "N_SIII": r"$\log\,N_{\mathrm{SIII}}$",
    "T_cloudy": r"$\log\,T_{\mathrm{Cloudy}}$",
}

# Default number of interpolation points per axis
DEFAULT_N_FINE = 200


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CloudyGridExplorer:
    """High-level wrapper for exploring a Cloudy grid with heatmaps.

    Parameters
    ----------
    grid : CloudyGridInterpolator
        An already-constructed interpolator (values in log-space).
    """

    def __init__(self, grid: CloudyGridInterpolator) -> None:
        self.grid = grid

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_table_path(
        cls,
        path: Union[str, Path],
        ion_columns: Optional[Sequence[str]] = None,
        format: Optional[str] = None,
    ) -> "CloudyGridExplorer":
        """Load a ``.dat`` / ``.ecsv`` grid file and wrap it.

        Parameters
        ----------
        format : str, optional
            Astropy table read format.  Defaults to ``'ascii'`` for ``.dat``
            files, otherwise auto-detected.
        """
        path = Path(path)
        if format is None and path.suffix.lower() == ".dat":
            format = "ascii"
        table = Table.read(str(path), format=format)
        grid = CloudyGridInterpolator.from_table(table, ion_columns=ion_columns)
        return cls(grid)

    @classmethod
    def from_table(
        cls,
        table: Table,
        ion_columns: Optional[Sequence[str]] = None,
    ) -> "CloudyGridExplorer":
        grid = CloudyGridInterpolator.from_table(table, ion_columns=ion_columns)
        return cls(grid)

    @classmethod
    def from_grid(cls, grid: CloudyGridInterpolator) -> "CloudyGridExplorer":
        return cls(grid)

    # -- discovery helpers --------------------------------------------------

    def available_axes(self) -> Tuple[str, ...]:
        """Return the names of the grid parameter axes."""
        return self.grid.axis_names

    def available_ions(self) -> Tuple[str, ...]:
        """Return the names of all ion / quantity columns in the grid."""
        return self.grid.ion_order

    def axis_values(self, axis: str) -> np.ndarray:
        """Return the unique grid-point values along *axis*."""
        idx = list(self.grid.axis_names).index(axis)
        return np.array(self.grid.points[idx])

    # -- 2-D slice extraction -----------------------------------------------

    def get_heatmap_data(
        self,
        x_axis: str,
        y_axis: str,
        ion: str,
        fixed_params: Mapping[str, float],
        *,
        n_fine: int = DEFAULT_N_FINE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the grid on a fine 2-D mesh for a given ion.

        Parameters
        ----------
        x_axis, y_axis : str
            Names of the two axes to vary (must be in ``available_axes()``).
        ion : str
            Ion column name (must be in ``available_ions()``).
        fixed_params : mapping
            Values for every remaining axis (pinned during evaluation).
        n_fine : int
            Number of points along each interpolated axis.

        Returns
        -------
        x_fine : 1-D array, shape (n_fine,)
        y_fine : 1-D array, shape (n_fine,)
        values  : 2-D array, shape (n_fine, n_fine)
            Log10 ion column density at each (x, y) mesh point.
        """
        axes = list(self.grid.axis_names)
        if x_axis not in axes:
            raise ValueError(f"x_axis '{x_axis}' not in grid axes {axes}")
        if y_axis not in axes:
            raise ValueError(f"y_axis '{y_axis}' not in grid axes {axes}")
        if x_axis == y_axis:
            raise ValueError("x_axis and y_axis must be different")
        if ion not in self.grid.ion_order:
            raise ValueError(f"ion '{ion}' not in grid ions {list(self.grid.ion_order)}")

        # Validate that all other axes are covered
        for ax in axes:
            if ax not in (x_axis, y_axis) and ax not in fixed_params:
                raise ValueError(
                    f"Axis '{ax}' is not on x or y and not in fixed_params. "
                    f"You must supply a value for it."
                )

        ion_idx = list(self.grid.ion_order).index(ion)

        x_idx = axes.index(x_axis)
        y_idx = axes.index(y_axis)
        x_fine = np.linspace(
            float(self.grid.points[x_idx][0]),
            float(self.grid.points[x_idx][-1]),
            n_fine,
        )
        y_fine = np.linspace(
            float(self.grid.points[y_idx][0]),
            float(self.grid.points[y_idx][-1]),
            n_fine,
        )

        # Build meshgrid of query points
        xx, yy = np.meshgrid(x_fine, y_fine, indexing="ij")
        n_total = xx.size
        query = np.empty((n_total, len(axes)), dtype=float)
        for dim, ax_name in enumerate(axes):
            if ax_name == x_axis:
                query[:, dim] = xx.ravel()
            elif ax_name == y_axis:
                query[:, dim] = yy.ravel()
            else:
                query[:, dim] = fixed_params[ax_name]

        # Evaluate – result has shape (n_total, n_ions)
        result = self.grid.interpolator(query)
        values = result[:, ion_idx].reshape(xx.shape)  # (n_fine_x, n_fine_y)

        return x_fine, y_fine, values

    # -- single heatmap plot ------------------------------------------------

    def plot_heatmap(
        self,
        ax: "plt.Axes",
        x_axis: str,
        y_axis: str,
        ion: str,
        fixed_params: Mapping[str, float],
        *,
        n_fine: int = DEFAULT_N_FINE,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        contour_levels: Optional[Sequence[float]] = None,
        contour_colors: Union[str, Sequence[str]] = "white",
        contour_linewidths: float = 1.0,
        contour_label_fontsize: float = 8,
        colorbar: bool = True,
        colorbar_label: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> Any:
        """Draw a single smooth heatmap on a Matplotlib axes.

        Parameters
        ----------
        contour_levels : sequence of float, optional
            Log-column-density values at which to draw labelled contours.

        Returns
        -------
        im : AxesImage
            The image handle from ``pcolormesh``.
        """
        x_fine, y_fine, values = self.get_heatmap_data(
            x_axis, y_axis, ion, fixed_params, n_fine=n_fine,
        )

        # pcolormesh expects (y, x) ordering when data has shape (nx, ny)
        im = ax.pcolormesh(
            x_fine,
            y_fine,
            values.T,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Contours
        if contour_levels is not None:
            cs = ax.contour(
                x_fine,
                y_fine,
                values.T,
                levels=sorted(contour_levels),
                colors=contour_colors,
                linewidths=contour_linewidths,
            )
            ax.clabel(cs, inline=True, fontsize=contour_label_fontsize, fmt="%.1f")

        # Labels
        if xlabel is None:
            xlabel = AXIS_LABELS.get(x_axis, x_axis)
        if ylabel is None:
            ylabel = AXIS_LABELS.get(y_axis, y_axis)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

        # Colorbar
        if colorbar:
            cbar_lbl = colorbar_label
            if cbar_lbl is None:
                cbar_lbl = ION_LABELS.get(ion, ion)
            plt.colorbar(im, ax=ax, label=cbar_lbl)

        return im

    # -- multi-panel grid ---------------------------------------------------

    def plot_grid(
        self,
        x_axis: str,
        y_axis: str,
        panel_axis: str,
        ion: str,
        fixed_params: Optional[Mapping[str, float]] = None,
        *,
        panel_values: Optional[Sequence[float]] = None,
        n_fine: int = DEFAULT_N_FINE,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        contour_levels: Optional[Sequence[float]] = None,
        contour_colors: Union[str, Sequence[str]] = "white",
        contour_linewidths: float = 1.0,
        contour_label_fontsize: float = 8,
        ncols: int = 4,
        panel_size: Tuple[float, float] = (4.0, 3.5),
        shared_colorbar: bool = True,
        suptitle: Optional[str] = None,
        figkwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple["plt.Figure", np.ndarray]:
        """Create a grid of heatmap panels tiled over *panel_axis*.

        Parameters
        ----------
        panel_axis : str
            Grid axis whose values generate individual panels.
        panel_values : sequence of float, optional
            Subset of ``panel_axis`` values to plot.  Defaults to all unique
            grid-point values along that axis.
        shared_colorbar : bool
            If ``True``, share one colorbar across all panels and auto-compute
            a common vmin/vmax.
        ncols : int
            Maximum number of columns in the panel grid.
        panel_size : (width, height)
            Size of each individual panel in inches.

        Returns
        -------
        fig, axes : Figure, ndarray of Axes
        """
        if fixed_params is None:
            fixed_params = {}
        fixed_params = dict(fixed_params)

        axes_list = list(self.grid.axis_names)
        if panel_axis not in axes_list:
            raise ValueError(f"panel_axis '{panel_axis}' not in grid axes {axes_list}")

        if panel_values is None:
            panel_values = self.axis_values(panel_axis)
        panel_values = np.asarray(panel_values, dtype=float)
        n_panels = len(panel_values)

        nrows = max(1, int(np.ceil(n_panels / ncols)))
        actual_ncols = min(n_panels, ncols)
        fw, fh = panel_size
        fig_kwargs = dict(figsize=(fw * actual_ncols + 1.5, fh * nrows + 0.8))
        if figkwargs:
            fig_kwargs.update(figkwargs)
        fig, axes = plt.subplots(nrows, actual_ncols, squeeze=False, **fig_kwargs)

        # Compute shared colour limits if requested
        if shared_colorbar and (vmin is None or vmax is None):
            all_vals: List[np.ndarray] = []
            for pv in panel_values:
                fp = dict(fixed_params, **{panel_axis: float(pv)})
                _, _, vals = self.get_heatmap_data(
                    x_axis, y_axis, ion, fp, n_fine=n_fine,
                )
                all_vals.append(vals)
            stacked = np.concatenate([v.ravel() for v in all_vals])
            finite = stacked[np.isfinite(stacked)]
            if vmin is None:
                vmin = float(np.nanmin(finite))
            if vmax is None:
                vmax = float(np.nanmax(finite))

        last_im = None
        panel_label = AXIS_LABELS.get(panel_axis, panel_axis)
        for idx, pv in enumerate(panel_values):
            row, col = divmod(idx, actual_ncols)
            ax = axes[row, col]
            fp = dict(fixed_params, **{panel_axis: float(pv)})
            last_im = self.plot_heatmap(
                ax,
                x_axis,
                y_axis,
                ion,
                fp,
                n_fine=n_fine,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                contour_levels=contour_levels,
                contour_colors=contour_colors,
                contour_linewidths=contour_linewidths,
                contour_label_fontsize=contour_label_fontsize,
                colorbar=not shared_colorbar,
                title=f"{panel_label} = {pv:.2f}",
            )

        # Hide unused panels
        for idx in range(n_panels, nrows * actual_ncols):
            row, col = divmod(idx, actual_ncols)
            axes[row, col].set_visible(False)

        if shared_colorbar and last_im is not None:
            cbar_label = ION_LABELS.get(ion, ion)
            fig.colorbar(last_im, ax=axes.ravel().tolist(), label=cbar_label,
                         fraction=0.02, pad=0.04)

        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=14, y=1.01)

        fig.tight_layout()
        return fig, axes

    def plot_multi_ion_grid(
        self,
        x_axis: str,
        y_axis: str,
        panel_axis: str,
        ions: Sequence[str],
        fixed_params: Optional[Mapping[str, float]] = None,
        *,
        panel_values: Optional[Sequence[float]] = None,
        n_fine: int = DEFAULT_N_FINE,
        cmap: str = "viridis",
        vmin_per_ion: Optional[Mapping[str, float]] = None,
        vmax_per_ion: Optional[Mapping[str, float]] = None,
        contour_levels_per_ion: Optional[Mapping[str, Sequence[float]]] = None,
        contour_colors: Union[str, Sequence[str]] = "white",
        contour_linewidths: float = 1.0,
        contour_label_fontsize: float = 8,
        panel_size: Tuple[float, float] = (4.0, 3.5),
        shared_colorbar: bool = True,
        suptitle: Optional[str] = None,
        figkwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple["plt.Figure", np.ndarray]:
        """Create a grid of heatmap panels, with one row per ion.

        Parameters
        ----------
        ions : sequence of str
            List of ions to plot. Each ion will have its own row.
        panel_axis : str
            Grid axis whose values generate individual panels (columns).
        panel_values : sequence of float, optional
            Subset of ``panel_axis`` values to plot across columns.
        vmin_per_ion, vmax_per_ion : mapping of str to float, optional
            Dictionaries specifying manual colorbar limits for specific ions.
        contour_levels_per_ion : mapping of str to sequence of float, optional
            Dictionary specifying contour levels for specific ions.
        """
        if fixed_params is None:
            fixed_params = {}
        fixed_params = dict(fixed_params)

        axes_list = list(self.grid.axis_names)
        if panel_axis not in axes_list:
            raise ValueError(f"panel_axis '{panel_axis}' not in grid axes {axes_list}")

        if panel_values is None:
            panel_values = self.axis_values(panel_axis)
        panel_values = np.asarray(panel_values, dtype=float)
        
        ncols = len(panel_values)
        nrows = len(ions)
        
        vmin_dict = vmin_per_ion or {}
        vmax_dict = vmax_per_ion or {}
        contour_dict = contour_levels_per_ion or {}

        fw, fh = panel_size
        fig_kwargs = dict(figsize=(fw * ncols + 1.5, fh * nrows + 0.8))
        if figkwargs:
            fig_kwargs.update(figkwargs)
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, **fig_kwargs)

        panel_label = AXIS_LABELS.get(panel_axis, panel_axis)
        
        for row, ion in enumerate(ions):
            # Compute shared colour limits for this ion if requested
            vmin = vmin_dict.get(ion)
            vmax = vmax_dict.get(ion)
            if shared_colorbar and (vmin is None or vmax is None):
                all_vals: List[np.ndarray] = []
                for pv in panel_values:
                    fp = dict(fixed_params, **{panel_axis: float(pv)})
                    _, _, vals = self.get_heatmap_data(
                        x_axis, y_axis, ion, fp, n_fine=n_fine,
                    )
                    all_vals.append(vals)
                stacked = np.concatenate([v.ravel() for v in all_vals])
                finite = stacked[np.isfinite(stacked)]
                if vmin is None:
                    vmin = float(np.nanmin(finite)) if finite.size > 0 else 0.0
                if vmax is None:
                    vmax = float(np.nanmax(finite)) if finite.size > 0 else 1.0

            last_im = None
            contour_levels = contour_dict.get(ion)
            
            for col, pv in enumerate(panel_values):
                ax = axes[row, col]
                fp = dict(fixed_params, **{panel_axis: float(pv)})
                
                title = None
                if row == 0:
                    title = f"{panel_label} = {pv:.2f}"
                
                # Only show x labels on the bottom row
                xlabel = AXIS_LABELS.get(x_axis, x_axis) if row == nrows - 1 else ""
                
                # Only show y labels on the first column
                ylabel = AXIS_LABELS.get(y_axis, y_axis) if col == 0 else ""
                
                last_im = self.plot_heatmap(
                    ax,
                    x_axis,
                    y_axis,
                    ion,
                    fp,
                    n_fine=n_fine,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    contour_levels=contour_levels,
                    contour_colors=contour_colors,
                    contour_linewidths=contour_linewidths,
                    contour_label_fontsize=contour_label_fontsize,
                    colorbar=False,  # We'll add row colorbars manually
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                )

            if shared_colorbar and last_im is not None:
                cbar_label = ION_LABELS.get(ion, ion)
                # Add a colorbar for this specific row (ion)
                fig.colorbar(last_im, ax=axes[row, :].ravel().tolist(), label=cbar_label,
                             fraction=0.02, pad=0.02)

        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=14, y=1.01)

        fig.tight_layout()
        return fig, axes


__all__ = [
    "CloudyGridExplorer",
    "AXIS_LABELS",
    "ION_LABELS",
    "DEFAULT_N_FINE",
]
