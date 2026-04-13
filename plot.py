from header import *
from line_info import *
from spectral import *
from model_builder import *


def _is_continuum_component(model):
    """Return True if the model looks like a continuum spline component."""
    if model is None:
        return False
    return (
        hasattr(model, '_knot_location_param_names')
        and hasattr(model, 'knot_locations')
        and hasattr(model, 'knot_values')
    )


def _find_continuum_component(model):
    """Locate the first continuum component inside a possibly compound model."""
    if _is_continuum_component(model):
        return model

    finder = getattr(model, 'traverse_postorder', None)
    if callable(finder):
        for sub_model in finder():
            if _is_continuum_component(sub_model):
                return sub_model
    return None


def _collect_continuum_components(model):
    """Return every continuum component present in the compound model."""
    components = []
    for sub_model in _iter_model_components(model):
        if _is_continuum_component(sub_model):
            components.append(sub_model)
    return components


def _format_redshift(z, precision=5):
    """Return a nicely formatted redshift string."""
    if z is None:
        return "unknown"
    return f"{float(z):.{precision}f}"


def _normalize_redshift_value(z, precision=6):
    """Round a redshift to a fixed precision for hashing/comparison."""
    if z is None:
        return None
    return round(float(z), precision)


def _build_redshift_color_map(main_z, mask_dict=None, explicit_colors=None):
    """Construct a consistent color mapping for each redshift of interest."""
    palette = [
        'tab:red', 'tab:purple', 'tab:orange', 'tab:blue', 'tab:brown',
        'tab:pink', 'tab:gray', 'tab:olive'
    ]

    color_map = {}

    if isinstance(explicit_colors, dict):
        for key, value in explicit_colors.items():
            normalized = _normalize_redshift_value(key)
            if normalized is None:
                continue
            color_map[normalized] = value

    if main_z is not None:
        normalized_main = _normalize_redshift_value(main_z)
        if normalized_main not in color_map:
            color_map[normalized_main] = 'green'

    entries = []
    if mask_dict is not None:
        entries = mask_dict.get('redshifts', []) or []

    for entry in entries:
        z_entry = _normalize_redshift_value(entry.get('z'))
        if z_entry is None or z_entry in color_map:
            continue
        if palette:
            color_map[z_entry] = palette.pop(0)
        else:
            # fallback to matplotlib cycle default
            color_map[z_entry] = mpl.rcParams['axes.prop_cycle'].by_key()['color'][0]

    return color_map


def _compute_binned_spectrum(spectrum, line_wave, z, n):
    """Rebin the spectrum onto an averaged velocity grid for a given line and redshift."""
    full_velocity_axis = to_velocity(spectrum.spectral_axis, line_wave, z).to('km/s')
    length = spectrum.shape[0]
    n = max(int(n), 1)

    if length < n:
        # Insufficient samples to average; return raw arrays.
        velocity = full_velocity_axis
        flux = spectrum.flux
        std = spectrum.uncertainty.array
        return velocity.value, flux.value, std

    remainder = length % n
    trimmed = length - remainder

    if trimmed <= 0:
        velocity = full_velocity_axis
        flux = spectrum.flux
        std = spectrum.uncertainty.array
        return velocity.value, flux.value, std

    velocity = np.average(full_velocity_axis[remainder:remainder+trimmed].reshape(-1, n), axis=1)
    flux = np.average(spectrum.flux[remainder:remainder+trimmed].reshape(-1, n), axis=1)
    std = np.sqrt(
        np.sum((spectrum.uncertainty.array[remainder:remainder+trimmed].reshape(-1, n)) ** 2, axis=1) / n ** 2
    )
    return velocity.value, flux.value, np.asarray(std)


def _to_velocity_value(value):
    """Convert an input value (possibly an astropy Quantity) to a float in km/s."""
    if value is None:
        return None
    if hasattr(value, 'to_value'):
        try:
            return float(value.to_value('km/s'))
        except Exception:
            pass
    if hasattr(value, 'to'):
        try:
            return float(value.to('km/s').value)
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return None


def _ensure_span_list(data):
    """Normalize one or more span specifications into a list of (low, high) floats."""
    spans = []

    if data is None:
        return spans

    if isinstance(data, dict):
        for value in data.values():
            spans.extend(_ensure_span_list(value))
        return spans

    if isinstance(data, (list, tuple, np.ndarray)):
        if len(data) == 0:
            return spans

        first = data[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            for item in data:
                spans.extend(_ensure_span_list(item))
            return spans

        if len(data) == 2:
            lower = _to_velocity_value(data[0])
            upper = _to_velocity_value(data[1])
            if lower is not None and upper is not None:
                if upper < lower:
                    lower, upper = upper, lower
                spans.append((lower, upper))
            return spans

        for item in data:
            spans.extend(_ensure_span_list(item))
        return spans

    # Single scalar values are ignored because a span needs two endpoints
    return spans


def _collect_mask_regions(mask_dict, line, target_z):
    """Extract mask velocity spans for a given line and redshift from mask_dict."""
    spans = []

    if mask_dict is None:
        return spans

    normalized_target = _normalize_redshift_value(target_z)
    if normalized_target is None:
        return spans

    for entry in (mask_dict.get('redshifts', []) or []):
        entry_z = entry.get('z')
        if entry_z is None:
            continue

        if _normalize_redshift_value(entry_z) != normalized_target:
            continue

        vrange = entry.get('vrange') or {}
        if not isinstance(vrange, dict):
            continue

        for bounds in (vrange.get(line) or []):
            if bounds is None:
                continue
            if not isinstance(bounds, (list, tuple, np.ndarray)) or len(bounds) != 2:
                continue

            lower = _to_velocity_value(bounds[0])
            upper = _to_velocity_value(bounds[1])

            if lower is None or upper is None:
                continue

            if upper < lower:
                lower, upper = upper, lower

            spans.append((lower, upper))

    return spans


def _iter_model_components(model):
    """Yield all unique components inside a possibly compound model."""
    if model is None:
        return

    seen = set()
    stack = [model]

    while stack:
        current = stack.pop()
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)

        yield current

        # Explore common compound-model attributes
        if hasattr(current, 'submodel_names'):
            for name in getattr(current, 'submodel_names') or []:
                try:
                    child = getattr(current, name)
                except AttributeError:
                    continue
                if child is not None:
                    stack.append(child)

        for attr in ('left', 'right'):
            child = getattr(current, attr, None)
            if child is not None:
                stack.append(child)

        if hasattr(current, 'models'):
            for child in getattr(current, 'models') or []:
                if child is not None:
                    stack.append(child)


def _collect_voigt_components(model):
    """Return a list of absorber submodels that carry per-line information."""
    components = []
    for component in _iter_model_components(model):
        if hasattr(component, '_line_list') and hasattr(component, '_z'):
            components.append(component)
    return components


def _find_lsf_component(model):
    """Attempt to locate a line-spread function component within the model."""
    for component in _iter_model_components(model):
        type_name = type(component).__name__.lower()
        comp_name = (getattr(component, 'name', '') or '').lower()
        if 'lsf' in comp_name or 'linespread' in type_name or 'line_spread' in type_name:
            return component
        if hasattr(component, 'kernel') or hasattr(component, 'convolution_kernel'):
            return component

    # Fallback: try final component if model is indexable
    try:
        candidate = model[-1]
        if candidate is not model:
            return candidate
    except Exception:
        pass

    return None

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['figure.facecolor'] = 'white'
# mpl.use('PDF')
plt.rc('text', usetex=False)
plt.rc('font', family='serif',size=10)
plt.rc('axes', linewidth=1.5)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15, direction='in')
plt.rc('ytick', labelsize=15, direction='in')
plt.rc('xtick', top=True)
plt.rc('ytick', right=True)
plt.rc('xtick.minor', visible=True)
plt.rc('ytick.minor', visible=True)
plt.rc('xtick.major',size=10, pad=4)
plt.rc('xtick.minor',size=5, pad=4)
plt.rc('ytick.major',size=10)
plt.rc('ytick.minor',size=5)
plt.rc('legend', fontsize=15)


def plot_absorber(fig, ax, spectrum, line_list, z, vrange=300, n=1, gray_out=None,
                  overall_alpha=1.0, color=None, plot_entries=None, show_legend=True,
                  n_cos=None, n_stis=None, stis_min_wave=2000, binning=None):
    """Plot the observed spectrum for each (line, z) panel using a fixed data color."""

    axes_array = np.atleast_1d(ax)
    axes = list(np.ravel(axes_array))

    if plot_entries is None:
        plot_entries = []
        lines = line_list or []
        for idx, line in enumerate(lines):
            gray_val = None
            if isinstance(gray_out, (list, tuple)) and idx < len(gray_out):
                gray_val = gray_out[idx]
            elif isinstance(gray_out, dict):
                gray_val = gray_out.get(line)
            plot_entries.append({
                'line': line,
                'z': z,
                'label': f"z={_format_redshift(z)}",
                'gray_out': gray_val,
                'alpha': overall_alpha,
            })

    for axis, entry in zip(axes, plot_entries):
        line = entry['line']
        entry_z = entry.get('z', z)
        label = entry.get('label', f"z={_format_redshift(entry_z)}")
        entry_alpha = entry.get('alpha', overall_alpha)
        gray_vals = entry.get('gray_out')
        mask_regions = entry.get('mask_regions')

        line_wave = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['wave'][0]
        ion_name = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['name'][0]

        n_local = n
        obs_wave = line_wave * (1 + entry_z)
        # Per-line binning override: binning dict maps line key → int
        if isinstance(binning, dict) and line in binning:
            n_local = int(binning[line])
        elif n_cos is not None or n_stis is not None:
            if obs_wave >= stis_min_wave:
                n_local = n_stis if n_stis is not None else n
            else:
                n_local = n_cos if n_cos is not None else n
        else:
            if obs_wave > 2000:
                n_local = max(int(np.floor(n / 2)), 1)

        vel, flux, std = _compute_binned_spectrum(spectrum, line_wave, entry_z, n_local)

        axis.step(
            vel,
            flux,
            alpha=0.5 * entry_alpha,
            color='blue',
            where='mid',
            rasterized=True,
        )
        axis.fill_between(
            vel,
            flux - std,
            flux + std,
            color='blue',
            alpha=0.15 * entry_alpha,
            step='mid',
            rasterized=True,
        )

        axis.vlines(0, 0, 1.5, color='black', linestyles='-.', alpha=0.5 * entry_alpha)
        axis.hlines(1, -vrange, vrange, color='slategrey', linestyles='dashed', alpha=0.9 * entry_alpha)

        shading_spans = []
        shading_spans.extend(_ensure_span_list(gray_vals))
        shading_spans.extend(_ensure_span_list(mask_regions))

        for lower, upper in shading_spans:
            if lower is None or upper is None:
                continue
            axis.fill_betweenx(
                [0, 1.5],
                [lower] * 2,
                [upper] * 2,
                color='lightgrey',
                alpha=0.35 * entry_alpha,
                rasterized=True,
                zorder=1,
            )

        axis.set_xlim(-vrange, vrange)
        axis.set_ylim(0, 1.5)

        axis.text(
            0.03,
            0.12,
            f"{ion_name} {np.round(line_wave, decimals=1)} $\\mathrm{{\\AA}}$\n{label}",
            ha='left',
            va='bottom',
            transform=axis.transAxes,
            fontsize='x-large',
        )

def plot_galaxy_for_paper(spectrum, fit, z, mask=None, vrange=300, n=1, save=True, title=None, name=None, gray_out=None, fig_return=False):
    # Define the line groups
    lyman_lines = ['LyA', 'LyB', 'LyD', 'LyG', 'LyE']
    carbon_lines = ['CII_1', 'CII_2', 'CIII', 'CIV_1', 'CIV_2']
    silicon_ii_lines = ['SiII_1', 'SiII_3', 'SiII_4', 'SiII_5', 'SiII_6']
    silicon_iii_iv_lines = ['SiIII', 'SiIV_1', 'SiIV_2','OVI_1', 'OVI_2']

    # Create a 5x5 grid of subplots
    fig, ax = plt.subplots(figsize=(15, 15), nrows=5, ncols=4, sharex=True, sharey=True)

    # Plot the absorber features for each line group
    plot_absorber(fig, ax[:, 0], spectrum, lyman_lines, z, vrange=vrange, n=n, gray_out=gray_out)
    plot_model(fig, ax[:, 0], spectrum, lyman_lines, fit, z, mask=mask, vrange=vrange)
    plot_absorber(fig, ax[:, 1], spectrum, carbon_lines, z, vrange=vrange, n=n, gray_out=gray_out)
    plot_model(fig, ax[:, 1], spectrum, carbon_lines, fit, z, mask=mask, vrange=vrange)
    plot_absorber(fig, ax[:, 2], spectrum, silicon_ii_lines, z, vrange=vrange, n=n, gray_out=gray_out)
    plot_model(fig, ax[:, 2], spectrum, silicon_ii_lines, fit, z, mask=mask, vrange=vrange)
    plot_absorber(fig, ax[:, 3], spectrum, silicon_iii_iv_lines, z, vrange=vrange, n=n, gray_out=gray_out)
    plot_model(fig, ax[:, 3], spectrum, silicon_iii_iv_lines, fit, z, mask=mask, vrange=vrange)
    # plot_absorber(fig, ax[:, 4], spectrum, oxygen_lines, z, vrange=vrange, n=n, gray_out=gray_out)
    # plot_model(fig, ax[:, 4], spectrum, oxygen_lines, fit, z, mask=mask, vrange=vrange)

    # # Set the labels for each column
    # ax[0, 0].set_title('Lyman Lines')
    # ax[0, 1].set_title('Carbon Lines')
    # ax[0, 2].set_title('Silicon II Lines')
    # ax[0, 3].set_title('Silicon III and IV Lines')
    # ax[0, 4].set_title('Oxygen Lines')

    # Remove empty subplots
    for i in range(5):
        for j in range(5):
            if j >= 1 and (j-1) >= len(lyman_lines):
                fig.delaxes(ax[i, j])

    # Adjust the layout
    # fig.tight_layout()

    plt.xlim(-vrange,vrange)
    plt.ylim(0, 1.5)
    plt.subplots_adjust(wspace=0.05, hspace=0.10)

    # Save or show the plot
    if save == True:
        if name is None:
            plt.savefig('absorber_plot_testing.pdf', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(name, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# def plot_model(fig, ax, spectrum, line_list, fit, z, mask=None, vrange=300):
#     """
#     Plot the fitted model for the absorber features in the given spectrum.

#     Parameters:
#     - fig (matplotlib.figure.Figure): The matplotlib figure for the plot.
#     - ax (array of matplotlib.axes.Axes): An array of subplots for individual absorber plots.
#     - spectrum (Spectrum1D): The input spectrum.
#     - line_list (list): List of spectral lines to plot.
#     - fit (callable): The fitted model.
#     - z (float): The redshift value.

#     Returns:
#     - None
#     """
#     if len(line_list) > 1:
#         for i, line in enumerate(line_list):
#             # Get the wavelength of the spectral line
#             line_wave = SEARCH_LINES[SEARCH_LINES['tempname']==line]['wave'][0]
#             full_velocity_axis = to_velocity(spectrum.spectral_axis, line_wave, z).to('km/s')
#             ax[i].plot(full_velocity_axis, fit(spectrum.spectral_axis.value), color='Green')
#             index = 0

#             # Find the index of the sub-model with None as the name
#             for j, sub in enumerate(fit):
#                 if sub.name is None:
#                     index = j
#                     break

#             for voigt in fit:
#                 if (voigt.name == 'exponential') or (voigt.name is None):
#                     continue

#                 voigt_line_name, voigt_line_z = voigt.name.split('__z=')
#                 voigt_line_wave = SEARCH_LINES[SEARCH_LINES['tempname']==voigt_line_name]['wave'][0]
#                 voigt_line_z = float(voigt_line_z)
#                 voigt_line_observed_wave = voigt_line_wave * (1 + voigt_line_z)
#                 voigt_line_observed_velocity = to_velocity(voigt_line_observed_wave * u.Angstrom, line_wave, z).to('km/s') 
#                 voigt_line_observed_velocity = voigt_line_observed_velocity.value   

#                 if (voigt_line_observed_velocity < -vrange) or (voigt_line_observed_velocity > vrange):
#                     continue            
#                 # if line not in voigt.name:
#                 #     continue
#                 lsf_voigt = generic_exp(voigt) | fit[index]
#                 if str(z) not in voigt.name:
#                     ax[i].plot(full_velocity_axis, lsf_voigt(spectrum.spectral_axis.value), '--', color='red', alpha=0.5)
#                 else:
#                     ax[i].plot(full_velocity_axis, lsf_voigt(spectrum.spectral_axis.value), '--', color='green', alpha=0.5)
#                     if mask is not None:
#                         ax[i].plot(full_velocity_axis, mask)
#             ax[i].set_xlim(-vrange,vrange)
#             ax[i].set_ylim(0, 1.5)
#     else:
#         for i, line in enumerate(line_list):
#             # Get the wavelength of the spectral line
#             line_wave = SEARCH_LINES[SEARCH_LINES['tempname']==line]['wave'][0]
#             full_velocity_axis = to_velocity(spectrum.spectral_axis, line_wave, z).to('km/s')
#             ax.plot(full_velocity_axis, fit(spectrum.spectral_axis.value), color='Green')
#             index = 0

#             # Find the index of the sub-model with None as the name
#             for j, sub in enumerate(fit):
#                 if sub.name is None:
#                     index = j
#                     break

#             for voigt in fit:
#                 if (voigt.name == 'exponential') or (voigt.name is None):
#                     continue

#                 voigt_line_name, voigt_line_z = voigt.name.split('__z=')
#                 voigt_line_wave = SEARCH_LINES[SEARCH_LINES['tempname']==voigt_line_name]['wave'][0]
#                 voigt_line_z = float(voigt_line_z)
#                 voigt_line_observed_wave = voigt_line_wave * (1 + voigt_line_z)
#                 voigt_line_observed_velocity = to_velocity(voigt_line_observed_wave * u.Angstrom, line_wave, z).to('km/s') 
#                 voigt_line_observed_velocity = voigt_line_observed_velocity.value   

#                 if (voigt_line_observed_velocity < -vrange) or (voigt_line_observed_velocity > vrange):
#                     continue   

#                 lsf_voigt = generic_exp(voigt) | fit[index]

#                 if str(z) not in voigt.name:
#                     ax.plot(full_velocity_axis, lsf_voigt(spectrum.spectral_axis.value), '--', color='red', alpha=0.5)
#                 else:
#                     ax.plot(full_velocity_axis, lsf_voigt(spectrum.spectral_axis.value), '--', color='green', alpha=0.5)
#                     if mask is not None:
#                         ax.plot(full_velocity_axis, mask)
#             ax.set_xlim(-vrange,vrange)
#             ax.set_ylim(0, 1.5)

def plot_model(fig, ax, spectrum, line_list, fit, z, mask=None, vrange=300, alpha=1,
               skip_submodels=False, submodel_colors=None, show_knots=False,
               show_continuum=True, redshift_colors=None):
    """
    Plot the fitted model for the absorber features in the given spectrum.

    Parameters:
    - fig (matplotlib.figure.Figure): The matplotlib figure for the plot.
    - ax (array of matplotlib.axes.Axes): An array of subplots for individual absorber plots.
    - spectrum (Spectrum1D): The input spectrum.
    - line_list (list): List of spectral lines to plot.
    - fit (callable): The fitted model.
    - z (float): The redshift value.

    Returns:
    - None
    """

    continuum_components = []
    continuum_knots_info = []
    continuum_curve = None

    z_color_lookup = {}
    if isinstance(redshift_colors, dict):
        for key, value in redshift_colors.items():
            normalized = _normalize_redshift_value(key)
            z_color_lookup[normalized] = value

    if show_knots or show_continuum:
        continuum_components = _collect_continuum_components(fit)
        if continuum_components:
            for component in continuum_components:
                knot_locations = np.asarray(component.knot_locations, dtype=float)
                knot_values = np.asarray(component.knot_values, dtype=float)
                sort_idx = np.argsort(knot_locations)
                continuum_knots_info.append({
                    'locations': knot_locations[sort_idx],
                    'values': knot_values[sort_idx],
                })
            if show_continuum:
                spectral_grid = np.asarray(spectrum.spectral_axis.value, dtype=float)
                continuum_curve = np.ones_like(spectral_grid, dtype=float)
                for component in continuum_components:
                    component_vals = np.asarray(component(spectral_grid), dtype=float)
                    continuum_curve += component_vals - 1.0
    model_values = np.asarray(fit(spectrum.spectral_axis.value))

    axes_array = np.atleast_1d(ax)
    axes = list(np.ravel(axes_array))

    voigt_components = _collect_voigt_components(fit)
    lsf_component = None if skip_submodels else _find_lsf_component(fit)

    if isinstance(line_list, list) and line_list and isinstance(line_list[0], dict):
        panel_entries = line_list
    elif isinstance(line_list, dict):
        panel_entries = [line_list]
    else:
        panel_entries = []
        for idx, line in enumerate(line_list or []):
            panel_entries.append({
                'line': line,
                'z': z,
                'color': z_color_lookup.get(_normalize_redshift_value(z), 'green'),
                'alpha': alpha,
            })

    for i, entry in enumerate(panel_entries):
        target_ax = axes[i]
        line = entry['line']
        entry_z = entry.get('z', z)
        entry_color = entry.get('color', 'green')
        entry_alpha = entry.get('alpha', alpha)

        line_wave = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['wave'][0]
        full_velocity_axis = to_velocity(spectrum.spectral_axis, line_wave, entry_z).to('km/s')

        target_ax.plot(full_velocity_axis, model_values, color=entry_color, alpha=entry_alpha)

        if show_continuum and continuum_curve is not None:
            target_ax.plot(full_velocity_axis.value, continuum_curve, color='purple', linestyle='--',
                           linewidth=1.2, alpha=entry_alpha, zorder=5)

        if show_knots and continuum_knots_info:
            for info in continuum_knots_info:
                knot_velocities = to_velocity(info['locations'] * u.Angstrom, line_wave, entry_z).to('km/s').value
                knot_mask = (knot_velocities >= -vrange) & (knot_velocities <= vrange)
                if not np.any(knot_mask):
                    continue
                knot_flux = info['values'][knot_mask]
                target_ax.scatter(
                    knot_velocities[knot_mask],
                    knot_flux,
                    color='purple',
                    marker='o',
                    s=26,
                    alpha=0.85,
                    zorder=8,
                )
                # if knot_velocities.size > 1:
                #     target_ax.plot(knot_velocities, knot_flux, color='purple', linewidth=1.0,
                #                    alpha=0.5, zorder=7)

        target_ax.set_xlim(-vrange, vrange)
        target_ax.set_ylim(0, 1.5)

        if skip_submodels:
            continue

        # prepare color mapping for submodels if requested
        # submodel_colors can be:
        #  - None : keep old behavior (green for same-z, red otherwise)
        #  - dict : mapping from voigt.name (or base name before '__') to color string
        #  - list/tuple : sequence of colors to cycle through for distinct submodels
        color_map = {}
        color_cycle = None
        if isinstance(submodel_colors, (list, tuple)):
            import itertools
            color_cycle = itertools.cycle(submodel_colors)

        for j, voigt in enumerate(voigt_components):
            voigt_line_z = float(getattr(voigt, '_z', entry_z))
            voigt_lines = getattr(voigt, '_line_list', ()) or ()

            # Determine if this is a contaminant (different redshift)
            is_contaminant = not np.isclose(voigt_line_z, entry_z, atol=1e-6)

            # For the main absorber, only plot components that include this line
            if not is_contaminant and line not in voigt_lines:
                continue

            # For contaminants, plot in every panel using the redshift color map
            if is_contaminant:
                lsf_voigt = generic_exp(voigt)
                if lsf_component is not None:
                    lsf_voigt = lsf_voigt | lsf_component

                model_values_contaminant = np.asarray(lsf_voigt(spectrum.spectral_axis.value))
                normalized_contaminant_z = _normalize_redshift_value(voigt_line_z)
                chosen_color = z_color_lookup.get(normalized_contaminant_z, entry_color)
                target_ax.plot(
                    full_velocity_axis,
                    model_values_contaminant,
                    color=chosen_color,
                    alpha=entry_alpha * 0.7,
                    linestyle='--',
                    linewidth=1.5,
                )
                continue

            lsf_voigt = generic_exp(voigt)
            if lsf_component is not None:
                lsf_voigt = lsf_voigt | lsf_component

            # choose color for this submodel
            chosen_color = None
            name_key = getattr(voigt, 'name', None)
            if name_key is None:
                name_key = f'comp_{j}'

            if isinstance(submodel_colors, dict):
                # exact match
                if name_key in submodel_colors:
                    chosen_color = submodel_colors[name_key]
                else:
                    # try base name before any '__'
                    base = name_key.split('__')[0]
                    if base in submodel_colors:
                        chosen_color = submodel_colors[base]

            elif color_cycle is not None:
                # assign a color per unique name_key
                if name_key not in color_map:
                    color_map[name_key] = next(color_cycle)
                chosen_color = color_map[name_key]

            # fallback to previous logic when no mapping provided/found
            if chosen_color is None:
                normalized_z = _normalize_redshift_value(voigt_line_z)
                chosen_color = z_color_lookup.get(normalized_z, entry_color)

            target_ax.plot(
                full_velocity_axis,
                lsf_voigt(spectrum.spectral_axis.value),
                linestyle=':',
                color=chosen_color,
                alpha=0.7 * entry_alpha,
            )

            if mask is not None and np.isclose(entry_z, voigt_line_z, atol=1e-6):
                target_ax.plot(full_velocity_axis, mask)



def plotter(spectrum, line_list, fit, z, mask=None, vrange=300, n=1, save=True,
            title=None, name=None, gray_out=None, fig_return=False, input_fig=None,
            input_ax=None, skip_submodels=False, alpha=1, show_knots=False,
            show_continuum=True, mask_dict=None, include_mask_lines=True,
            redshift_colors=None, show_legend=True, n_cos=None, n_stis=None,
            stis_min_wave=2000, binning=None):
    """
    Plot the absorber features and the fitted model.

    Parameters:
    - spectrum (Spectrum1D): The input spectrum.
    - line_list (list): List of spectral lines to plot.
    - fit (callable): The fitted model.
    - z (float): The redshift value.

    Returns:
    - None
    """
    color_map = _build_redshift_color_map(z, mask_dict, explicit_colors=redshift_colors)

    def _resolve_gray(gray_obj, line, entry_z, idx=None):
        if gray_obj is None:
            return None
        if isinstance(gray_obj, dict):
            key_exact = (line, _normalize_redshift_value(entry_z))
            if key_exact in gray_obj:
                return gray_obj[key_exact]
            if line in gray_obj:
                return gray_obj[line]
        if isinstance(gray_obj, (list, tuple)) and idx is not None and idx < len(gray_obj):
            return gray_obj[idx]
        return None

    main_lines = list(line_list or [])
    main_label = 'Main absorber'
    if mask_dict is not None:
        for entry in mask_dict.get('redshifts', []) or []:
            if np.isclose(entry.get('z', -np.inf), z, atol=1e-6):
                main_label = entry.get('label', main_label)
                for l in entry.get('lines', []) or []:
                    if l not in main_lines:
                        main_lines.append(l)
                break

    panel_entries = []
    main_color = color_map.get(_normalize_redshift_value(z), 'green')
    for idx, line in enumerate(main_lines):
        panel_entries.append({
            'line': line,
            'z': z,
            'color': main_color,
            'label': f"{main_label} (z={_format_redshift(z)})",
            'legend_label': f"{main_label} (z={_format_redshift(z)})",
            'alpha': alpha,
            'gray_out': _resolve_gray(gray_out, line, z, idx),
        })

    if mask_dict is not None and include_mask_lines:
        for entry in mask_dict.get('redshifts', []) or []:
            entry_z = entry.get('z')
            if entry_z is None or np.isclose(entry_z, z, atol=1e-6):
                continue
            entry_lines = entry.get('lines', []) or []
            entry_label = entry.get('label')
            label_text = entry.get('text_label')
            legend_text = entry.get('legend_label')
            default_text = f"z={_format_redshift(entry_z)}"
            if entry_label:
                default_text = f"{entry_label} (z={_format_redshift(entry_z)})"
            label_text = label_text or default_text
            legend_text = legend_text or default_text
            entry_color = color_map.get(_normalize_redshift_value(entry_z), None) or 'tab:red'
            for line in entry_lines:
                panel_entries.append({
                    'line': line,
                    'z': entry_z,
                    'color': entry_color,
                    'label': label_text,
                    'legend_label': legend_text,
                    'alpha': entry.get('alpha', 1.0),
                    'gray_out': _resolve_gray(gray_out, line, entry_z, None),
                })

    if not panel_entries:
        raise ValueError("No lines available to plot. Provide line_list or mask_dict entries.")

    if mask_dict is not None:
        for entry in panel_entries:
            spans = _collect_mask_regions(mask_dict, entry['line'], entry.get('z', z))
            entry['mask_regions'] = spans

    nrows = len(panel_entries)

    if input_fig is not None and input_ax is not None:
        fig = input_fig
        ax = input_ax
    else:
        fig, ax = plt.subplots(figsize=(8, int(2 * nrows)), nrows=nrows, sharex=True)

    axes_array = np.atleast_1d(ax)
    axes = list(np.ravel(axes_array))

    # Plot the observed spectra (always blue) per panel
    plot_absorber(
        fig,
        axes,
        spectrum,
        line_list=None,
        z=z,
        vrange=vrange,
        n=n,
        gray_out=None,
        overall_alpha=alpha,
        plot_entries=panel_entries,
        show_legend=False,
        n_cos=n_cos,
        n_stis=n_stis,
        stis_min_wave=stis_min_wave,
        binning=binning,
    )

    # Overlay model predictions with redshift-specific colors
    plot_model(
        fig,
        axes,
        spectrum,
        panel_entries,
        fit,
        z,
        mask=mask,
        vrange=vrange,
        skip_submodels=skip_submodels,
        alpha=alpha,
        show_knots=show_knots,
        show_continuum=show_continuum,
        redshift_colors=color_map,
    )

    if title is not None:
        plt.title(title, fontdict={'fontsize': 20})

    if save==True:
        if name is None:
            plt.savefig('absorber_plot.pdf', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(str(name) + '_absorber_plot.pdf', dpi=300, bbox_inches='tight')
    # else:
    #     plt.show()

    if fig_return==True:
        return fig, ax
    
def plotter_model_only(spectrum, line_list, fit, z, mask=None, vrange=300, n=1,
                       save=True, title=None, name=None, gray_out=None,
                       fig_return=False, input_fig=None, input_ax=None,
                       alpha=1.0, skip_submodels=False, show_knots=False,
                       show_continuum=True):
    """
    Plot the absorber features and the fitted model.

    Parameters:
    - spectrum (Spectrum1D): The input spectrum.
    - line_list (list): List of spectral lines to plot.
    - fit (callable): The fitted model.
    - z (float): The redshift value.

    Returns:
    - None
    """
    nrows = len(line_list)

    if input_fig is not None and input_ax is not None:
        fig = input_fig
        ax = input_ax
    else:
        fig, ax = plt.subplots(figsize=(8, int(2 * nrows)), nrows=nrows, sharex=True)

    # # Plot the absorber features
    # plot_absorber(fig, ax, spectrum, line_list, z, vrange=vrange, n=n, gray_out=gray_out)
    
    # Plot the fitted model
    plot_model(
        fig,
        ax,
        spectrum,
        line_list,
        fit,
        z,
        mask=mask,
        vrange=vrange,
        skip_submodels=skip_submodels,
        alpha=alpha,
        show_knots=show_knots,
        show_continuum=show_continuum,
    )

    if title is not None:
        plt.title(title, fontdict={'fontsize': 20})

    if save==True:
        if name is None:
            plt.savefig('absorber_plot.pdf', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(str(name) + '_absorber_plot.pdf', dpi=300, bbox_inches='tight')
    # else:
    #     plt.show()

    if fig_return==True:
        return fig, ax


def plotViolin(output_Ns, ions, subtable):
    """
    Creates a violin plot for the given data.

    Parameters:
    output_Ns : A 2D array where each row corresponds to an ion and each column corresponds to a measurement.
    ions : A list of ion names corresponding to the rows in output_Ns.
    subtable : A DataFrame containing the median, upper, and lower errors, and upper limit flags for each ion.

    The function plots a violin plot of the measurements for each ion. It also plots the median and error bars from the subtable, and marks any upper limits with a downward-pointing triangle.

    The plot is displayed using plt.show().

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(15,5))
    ax.violinplot(output_Ns.swapaxes(0,1), widths=0.7, showextrema=True, showmedians=True)

    ax.set_xticks(np.arange(1,13))
    ax.set_xticklabels(ions)

    expected = np.full(len(ions), np.nan)
    upper = np.full(len(ions), np.nan)
    lower = np.full(len(ions), np.nan)
    ul = np.full(len(ions), np.nan)
    ul_nomcmc = np.full(len(ions), np.nan)
    
    for i,ion in enumerate(ions):
        ion_table = subtable[subtable['ion'] == ion]
        if len(ion_table) == 0:
            continue
        if ion_table['upper_limit'] == False and ion_table['non_MCMC_upper_limit'] == False:
            expected[i] = ion_table['median_N']
            upper[i] = ion_table['N_right_error']
            lower[i] = ion_table['N_left_error']
        if ion_table['upper_limit'] == True:
            ul[i] = ion_table['2-sigma_UL_N']
        if ion_table['non_MCMC_upper_limit'] == True:
            ul_nomcmc[i] = ion_table['2-sigma_UL_N']


    plt.errorbar(np.arange(1,13), expected, color='green', yerr=(lower, upper), fmt='o')
    plt.scatter(np.arange(1,13), ul, color='red', marker='v')
    plt.scatter(np.arange(1,13), ul_nomcmc, edgecolor='red', color= 'white', marker='v')

    plt.ylabel('logN')
    plt.ylim(11,15)
    plt.show()


def plot_log_prob_histogram(log_probs, ls_log_prob=None, bins=50,
                             save_path=None, return_fig=False, verbose=False,
                             xlim=None):
    """Plot a histogram of flattened log-probability samples with optional LS marker."""
    values = np.ravel(np.asarray(log_probs))
    finite_mask = np.isfinite(values)
    if not np.all(finite_mask):
        if verbose:
            print(f"Discarding {np.size(values) - np.count_nonzero(finite_mask)} non-finite log-prob entries")
        values = values[finite_mask]

    if values.size == 0:
        raise ValueError("No finite log-probability values provided for plotting")

    fig, ax = plt.subplots(figsize=(6, 4))

    if xlim is not None:
        if len(xlim) != 2:
            raise ValueError("xlim must be a (min, max) tuple when provided")
        xmin, xmax = xlim
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
            raise ValueError("xlim must contain finite values with xmin < xmax")
        hist_range = (xmin, xmax)
        ax.set_xlim(xmin, xmax)
        hist_bins = np.linspace(xmin, xmax, bins + 1) if np.isscalar(bins) else bins
        ax.hist(values, bins=hist_bins, range=hist_range, color='steelblue', alpha=0.7, edgecolor='black')
    else:
        ax.hist(values, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('log probability')
    ax.set_ylabel('count')
    ax.set_title('Log-probability distribution')

    if ls_log_prob is not None and np.isfinite(ls_log_prob):
        ax.axvline(ls_log_prob, color='red', linestyle='--', linewidth=1.5,
                   label='Least-squares log prob')
        ax.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        if not return_fig:
            plt.close(fig)
            return None

    if return_fig:
        return fig
