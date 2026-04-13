from header import *  # Importing necessary dependencies
from line_info import *  # Importing additional dependencies from other modules
from helpers import *  # Importing helper functions from a different module

deltav = 60


def _extract_line_value(line_entry, key):
    value = line_entry[key]
    if hasattr(value, 'shape') and value.shape == ():
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            raise ValueError(f"Line entry for '{key}' is empty.")
        return value[0]
    return value


def calc_Wr(spectrum, line, z, v=0*u.km/u.second, dv=[-deltav*u.km/u.second, deltav*u.km/u.second], mask=1.0):
    line = _extract_line_value(line, 'wave')
    velocity_axis = to_velocity(spectrum.spectral_axis, line, z)

    v_cut = np.logical_and((velocity_axis > (dv[0])), velocity_axis < (dv[1]))

    F_cut = ((1 - spectrum.flux) * mask) [v_cut]
    F_err_cut = (spectrum.uncertainty.array * mask) [v_cut]

    rest_spectral_axis = spectrum.spectral_axis / (1 + z)
    wave_cut = rest_spectral_axis[v_cut]
    index = find_nearest_index(spectrum.spectral_axis.value, line*(1 + z))
    dW = (rest_spectral_axis[index] - rest_spectral_axis[index - 1])

    Wr = np.trapz(F_cut, wave_cut)
    #Wr = np.sum(F_cut) * dW
    WrErr = np.sqrt(np.trapz(F_err_cut**2, wave_cut) * dW)
    return Wr, WrErr


def calc_Wr_velocity(velocity, flux, flux_err, line, dv=[-deltav*u.km/u.second, deltav*u.km/u.second], mask=1.0):
    line = _extract_line_value(line, 'wave')
    v_cut = np.logical_and((velocity > (dv[0])), velocity < (dv[1]))
    velocity_cut = velocity[v_cut]
    rest_spectral_axis = to_wavelength(velocity_cut, line, z=0)

    F_cut = ((1 - flux) * mask) [v_cut]
    F_err_cut = (flux_err * mask) [v_cut]
    index = find_nearest_index(rest_spectral_axis.value, line * (1 + 0))
    dW = (rest_spectral_axis[index] - rest_spectral_axis[index - 1])
    Wr = np.trapz(F_cut, rest_spectral_axis)
    WrErr = np.sqrt(np.trapz(F_err_cut**2, rest_spectral_axis) * dW)

    return Wr, WrErr

def calc_median_noise(spectrum, line, z, v=0*u.km/u.second, dv=[-deltav*u.km/u.second, deltav*u.km/u.second], mask=1.0):
    line = _extract_line_value(line, 'wave')
    velocity_axis = to_velocity(spectrum.spectral_axis, line, z)

    v_cut = np.logical_and((velocity_axis > (dv[0])), velocity_axis < (dv[1]))

    F_cut = ((1 - spectrum.flux) * mask) [v_cut]
    F_err_cut = (spectrum.uncertainty.array * mask) [v_cut]

    return np.median(F_err_cut)

def calc_median_flux(spectrum, line, z, v=0*u.km/u.second, dv=[-deltav*u.km/u.second, deltav*u.km/u.second], mask=1.0):
    line = _extract_line_value(line, 'wave')
    velocity_axis = to_velocity(spectrum.spectral_axis, line, z)

    v_cut = np.logical_and((velocity_axis > (dv[0])), velocity_axis < (dv[1]))

    Flux = ((spectrum.flux) * mask) [v_cut]

    return np.median(Flux)

def calc_AOD(spectrum, line, z, v=0*u.km/u.second, dv=[-deltav*u.km/u.second, deltav*u.km/u.second], mask=1.0):

    
    f = _extract_line_value(line, 'f')
    line = _extract_line_value(line, 'wave')
    velocity_axis = to_velocity(spectrum.spectral_axis, line, z)
    v_cut = np.logical_and((velocity_axis > (dv[0])), velocity_axis < (dv[1]))
    F_cut = (spectrum.flux * mask) [v_cut]

    F_err_cut = (spectrum.uncertainty.array * mask) [v_cut]

    tau = np.log(1 / F_cut)
    tau_err = F_err_cut / F_cut

    tau[tau == np.inf] = 0
    tau[np.isnan(tau)] = 0

    tau_err[tau_err == np.inf] = 0
    tau_err[np.isnan(tau_err)] = 0

    # print(tau)
    # print(tau_err)

    rest_spectral_axis = spectrum.spectral_axis / (1 + z)
    wave_cut = spectrum.spectral_axis[v_cut]
    index = find_nearest_index(spectrum.spectral_axis.value, line*(1 + z))
    dW = (rest_spectral_axis[index] - rest_spectral_axis[index - 1])

    AOD = np.trapz(tau, wave_cut)
    AODErr = np.sqrt(np.trapz(tau_err**2, wave_cut) * dW)
    return AOD, AODErr

def calc_pixel_AOD(spectrum, line, z, v=0*u.km/u.second, dv=[-deltav*u.km/u.second, deltav*u.km/u.second], mask=1.0):
        f = _extract_line_value(line, 'f')
        line = _extract_line_value(line, 'wave')
        velocity_axis = to_velocity(spectrum.spectral_axis, line, z)
        v_cut = np.logical_and((velocity_axis > (dv[0])), velocity_axis < (dv[1]))
        F_cut = (spectrum.flux * mask) [v_cut]
    
        F_err_cut = (spectrum.uncertainty.array * mask) [v_cut]
    
        tau = np.log(1 / F_cut)
        tau_err = F_err_cut / F_cut
        tau[tau == np.inf] = 0
        tau[np.isnan(tau)] = 0
        tau_err[tau_err == np.inf] = 0
        rest_spectral_axis = spectrum.spectral_axis / (1 + z)
        wave_cut = spectrum.spectral_axis[v_cut]
        index = find_nearest_index(spectrum.spectral_axis.value, line*(1 + z))
        dW = (rest_spectral_axis[index] - rest_spectral_axis[index - 1])
    
        AOD = tau
        AODErr = tau_err
        return AOD, AODErr, velocity_axis[v_cut]


def calc_pixel_flux(spectrum, line, z, v=0*u.km/u.second, dv=[-deltav*u.km/u.second, deltav*u.km/u.second], mask=1.0):
    line = _extract_line_value(line, 'wave')
    velocity_axis = to_velocity(spectrum.spectral_axis, line, z)

    v_cut = np.logical_and((velocity_axis > (dv[0])), velocity_axis < (dv[1]))

    F = ((spectrum.flux) * mask) [v_cut]
    F_err_cut = (spectrum.uncertainty.array * mask) [v_cut]

    F_err_cut = np.where(F_err_cut == 0, 1e10, F_err_cut)
    return F, F_err_cut, velocity_axis[v_cut]

def AOD_to_N(AOD, AODErr, line):
    f = _extract_line_value(line, 'f')
    wave = _extract_line_value(line, 'wave')
    N = m_e * c**2 / np.pi / e.gauss**2 / f / (wave * u.Angstrom)**2 * AOD
    NErr = m_e * c**2 / np.pi / e.gauss**2 / f / (wave * u.Angstrom)**2 * AODErr

    return N.to(u.cm**(-2)), NErr.to(u.cm**(-2))

def Wr_to_N(Wr, WrErr, line):
    f = _extract_line_value(line, 'f')
    wave = _extract_line_value(line, 'wave')
    N = m_e * c**2 / np.pi / e.gauss**2 / f / (wave * u.Angstrom)**2 * Wr
    NErr = m_e * c**2 / np.pi / e.gauss**2 / f / (wave * u.Angstrom)**2 * WrErr
    
    return N.to(u.cm**(-2)), NErr.to(u.cm**(-2))

def N_to_Wr(N, NErr, line):
    f = _extract_line_value(line, 'f')
    wave = _extract_line_value(line, 'wave')
    Wr = np.pi * e.gauss**2 * f * (wave * u.Angstrom)**2 / m_e / c**2 * N
    WrErr = np.pi * e.gauss**2 * f * (wave * u.Angstrom)**2 / m_e / c**2 * NErr

    return Wr.to(u.Angstrom), WrErr.to(u.Angstrom)


def _normalize_velocity_window(velocity_window):
    """Convert a velocity window specification into astropy Quantities."""
    if velocity_window is None:
        return [-deltav * u.km / u.second, deltav * u.km / u.second]

    if (isinstance(velocity_window, (list, tuple, np.ndarray))
            and len(velocity_window) == 2):
        low, high = velocity_window
        low_q = low if hasattr(low, 'unit') else float(low) * u.km / u.second
        high_q = high if hasattr(high, 'unit') else float(high) * u.km / u.second
        if high_q < low_q:
            low_q, high_q = high_q, low_q
        return [low_q, high_q]

    raise ValueError(
        "velocity_window must be None or a sequence of two values (low, high) in km/s"
    )


def _resolve_velocity_window(velocity_window, line_key):
    if isinstance(velocity_window, dict):
        return _normalize_velocity_window(velocity_window.get(line_key))
    return _normalize_velocity_window(velocity_window)


def _resolve_mask(mask, line_key):
    if mask is None:
        return 1.0
    if isinstance(mask, dict):
        return mask.get(line_key, 1.0)
    return mask


def compute_wr_upper_limit(
        spectrum,
        line_key,
        z,
        method="sigma_error",
        sigma_level=3.0,
        velocity_window=None,
        mask=1.0,
):
    """Compute an equivalent-width based upper limit and convert it to log column density.

    Parameters
    ----------
    spectrum : Spectrum1D
        Normalised spectrum containing the target feature.
    line_key : str or sequence of str
        One or more entries in ``SEARCH_LINES['tempname']`` identifying transitions.
    z : float
        Redshift of the absorber.
    method : {"sigma_error", "abs_plus_sigma"}
        Strategy for turning the measured W_r and uncertainty into an upper limit.
    sigma_level : float, optional
        Multiplicative factor for the uncertainty term. Defaults to 3.0.
    velocity_window : tuple/list or dict, optional
        Integration bounds in km/s. When a dict is supplied, it should map line keys to
        window tuples.
    mask : array-like, scalar, or dict, optional
        Multiplicative mask applied to the flux and error arrays. Dict input should map
        line keys to per-line masks.

    Returns
    -------
    tuple
        (W_r_limit in Angstrom, logN limit in dex, chosen_line_key)
    """
    method_key = (method or "").lower()
    if method_key not in {"sigma_error", "abs_plus_sigma"}:
        raise ValueError(
            "method must be one of {'sigma_error', 'abs_plus_sigma'} (case-insensitive)"
        )

    if isinstance(line_key, (list, tuple, np.ndarray, set)):
        line_keys = [str(key) for key in line_key]
    else:
        line_keys = [str(line_key)]

    results = []
    for key in line_keys:
        line_rows = SEARCH_LINES[SEARCH_LINES['tempname'] == key]
        if len(line_rows) == 0:
            raise ValueError(f"No line information found for tempname '{key}'.")
        line_row = line_rows[0]

        dv_bounds = _resolve_velocity_window(velocity_window, key)
        mask_value = _resolve_mask(mask, key)

        Wr, WrErr = calc_Wr(
            spectrum,
            line_row,
            z,
            dv=dv_bounds,
            mask=mask_value,
        )

        sigma_abs = float(np.abs(sigma_level))
        WrErr_mag = np.abs(WrErr)

        if method_key == "sigma_error":
            Wr_limit = sigma_abs * WrErr_mag
        else:  # abs_plus_sigma
            Wr_limit = np.abs(Wr) + sigma_abs * WrErr_mag

        Wr_limit = np.abs(Wr_limit)

        N_limit, _ = Wr_to_N(Wr_limit, WrErr_mag, line_row)
        logN_limit = float(np.log10(N_limit.to(u.cm**-2).value))
        results.append((Wr_limit.to(u.Angstrom), logN_limit, key))

    if not results:
        raise ValueError("No valid transitions supplied for W_r upper-limit calculation.")

    best_result = min(results, key=lambda item: item[1])
    return best_result


def compute_wr_lower_limit(
        spectrum,
        line_key,
        z,
        method="abs_minus_sigma",
        sigma_level=2.0,
        velocity_window=None,
        mask=1.0,
):
    """Compute an equivalent-width based lower limit and convert it to log column density.

    For saturated lines the measured Wr is large; the 2-sigma lower limit is
    ``Wr - n * sigma(Wr)``.  The result is converted to log N via the linear
    (weak-line) approximation through :func:`Wr_to_N`.

    Parameters
    ----------
    spectrum : Spectrum1D
        Normalised spectrum containing the target feature.
    line_key : str or sequence of str
        One or more entries in ``SEARCH_LINES['tempname']`` identifying transitions.
    z : float
        Redshift of the absorber.
    method : {"abs_minus_sigma"}
        Strategy for turning the measured W_r and uncertainty into a lower limit.
        ``"abs_minus_sigma"`` computes ``|Wr| - n * |WrErr|``.
    sigma_level : float, optional
        Multiplicative factor for the uncertainty term. Defaults to 2.0.
    velocity_window : tuple/list or dict, optional
        Integration bounds in km/s. When a dict is supplied, it should map line keys to
        window tuples.
    mask : array-like, scalar, or dict, optional
        Multiplicative mask applied to the flux and error arrays. Dict input should map
        line keys to per-line masks.

    Returns
    -------
    tuple
        (W_r_limit in Angstrom, logN limit in dex, chosen_line_key)
    """
    method_key = (method or "").lower()
    if method_key not in {"abs_minus_sigma"}:
        raise ValueError(
            "method must be one of {'abs_minus_sigma'} (case-insensitive)"
        )

    if isinstance(line_key, (list, tuple, np.ndarray, set)):
        line_keys = [str(key) for key in line_key]
    else:
        line_keys = [str(line_key)]

    results = []
    for key in line_keys:
        line_rows = SEARCH_LINES[SEARCH_LINES['tempname'] == key]
        if len(line_rows) == 0:
            raise ValueError(f"No line information found for tempname '{key}'.")
        line_row = line_rows[0]

        dv_bounds = _resolve_velocity_window(velocity_window, key)
        mask_value = _resolve_mask(mask, key)

        Wr, WrErr = calc_Wr(
            spectrum,
            line_row,
            z,
            dv=dv_bounds,
            mask=mask_value,
        )

        sigma_abs = float(np.abs(sigma_level))
        WrErr_mag = np.abs(WrErr)
        Wr_mag = np.abs(Wr)

        # Lower limit: Wr - n*sigma
        Wr_limit = Wr_mag - sigma_abs * WrErr_mag
        # Clamp at zero — if Wr - n*sigma < 0 the line isn't meaningfully detected
        if hasattr(Wr_limit, 'value'):
            Wr_limit_val = float(Wr_limit.value)
        else:
            Wr_limit_val = float(Wr_limit)
        if Wr_limit_val <= 0:
            raise ValueError(
                f"Lower limit for '{key}' is non-positive (Wr={Wr_mag}, "
                f"{sigma_abs}*sigma={sigma_abs * WrErr_mag}). "
                "The line may not be saturated or the S/N is too low."
            )

        N_limit, _ = Wr_to_N(Wr_limit, WrErr_mag, line_row)
        logN_limit = float(np.log10(N_limit.to(u.cm**-2).value))
        results.append((Wr_limit.to(u.Angstrom) if hasattr(Wr_limit, 'to') else Wr_limit, logN_limit, key))

    if not results:
        raise ValueError("No valid transitions supplied for W_r lower-limit calculation.")

    # For lower limits, pick the transition giving the *highest* logN (most constraining)
    best_result = max(results, key=lambda item: item[1])
    return best_result
