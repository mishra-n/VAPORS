from header import *  # Importing necessary dependencies
from line_info import *  # Importing additional dependencies from other modules
from helpers import *  # Importing helper functions from a different module
from astropy.modeling import Fittable1DModel, Parameter
from scipy.interpolate import make_interp_spline
# Constants for physical calculations
constant2 = (1 / (m_e * c * u.Hz) * (4 * pi**2 * e.gauss**2)).to(u.cm**2).value
c_kms = c.to(u.km/u.second).value
c_As = c.to(u.Angstrom/u.second).value
prefactor_tau = (np.pi * e.gauss**2 / (m_e * c)).cgs.value
sqrt_pi = np.sqrt(np.pi)


def params_to_fit(N, b, v, z, lambda_0, gamma, f):
    """
    Calculate parameters required for Voigt profile fitting.

    Parameters:
    - N (float): Column density.
    - b (float): Doppler parameter.
    - v (float): Velocity.
    - z (float): Redshift.
    - lambda_0 (float): Rest wavelength.
    - gamma (float): Gamma value.
    - f (float): Oscillator strength.

    Returns:
    - lam (float): Wavelength.
    - amplitude_L (float): Amplitude of the Lorentzian component.
    - fwhm_G (float): Full Width at Half Maximum (Gaussian component).
    """
    amplitude_L = N * constant2 * f / gamma
    lam = (v / c_kms) * lambda_0 + lambda_0
    fwhm_G = b * lam * (2 * np.sqrt(np.log(2))) / c_kms
    fwhm_G = fwhm_G * (1 + z)
    lam = lam * (1 + z)
    return lam, amplitude_L, fwhm_G

# Constants for physical calculations
constant1 = (np.sqrt(2 * k_B / (m_p * c**2)) * c).to(u.km/u.second / u.Kelvin**(0.5)).value
multiple = (1e12 / u.cm**2).value / (1e12 / u.cm**2).to(1 / u.micrometer**2).value

def Absorber1D(line_list, z, N_values=None, T=None, b_other=None, v=None,
               fix_T=False, fix_b=False, fix_v=False,
               per_ion_b=False, b_values=None):
    """
    Factory function to create a MultiVoigt1D class with a flexible number of N parameters.
    If fix_T is True, T is treated as a fixed attribute (self.T) and not a model parameter.
    If fix_b is True, b_other is fixed (self.b_other).
    If fix_v is True, v is fixed (self.v).

    If per_ion_b is True, the T/b_other decomposition is replaced by individual
    total Doppler b parameters for each distinct ion (e.g. b_HI, b_CIV).
    A single shared v is still used.  b_values gives defaults per distinct ion
    (same length as unique ions; defaults to 10 km/s each).
    """
    if per_ion_b and (not fix_T and T is None):
        fix_T = True
    if per_ion_b and (not fix_b and b_other is None):
        fix_b = True

    # defaults
    if T is None:
        T = 1  # Default temperature in Kelvin
    if b_other is None:
        b_other = 1
    if v is None:
        v = 0
    if N_values is None:
        N_values = [1e12] * len(line_list)
    if len(line_list) != len(N_values):
        raise ValueError("The length of line_list must match the length of N_values.")
    if not isinstance(line_list, list):
        raise TypeError("line_list must be a list of spectral lines.")
    if not isinstance(N_values, list):
        raise TypeError("N_values must be a list of column densities corresponding to the lines in line_list.")
    if not all(isinstance(N, (int, float)) for N in N_values):
        raise TypeError("All elements in N_values must be numeric (int or float).")
    if not isinstance(z, (int, float)):
        raise TypeError("z must be a numeric value (int or float).")
    if not isinstance(T, (int, float)):
        raise TypeError("T must be a numeric value (int or float).")
    if not isinstance(b_other, (int, float)):
        raise TypeError("b_other must be a numeric value (int or float).")
    if not isinstance(v, (int, float)):
        raise TypeError("v must be a numeric value (int or float).")
    if not all(isinstance(line, str) for line in line_list):
        raise TypeError("All elements in line_list must be strings representing spectral lines.")

    # Find unique ions
    ion_index = {}
    distinct_ions = []
    for line in line_list:
        line_info = SEARCH_LINES[SEARCH_LINES['tempname'] == line]
        ion = line_info['name'][0]
        if ion not in ion_index:
            ion_index[ion] = len(distinct_ions)
            distinct_ions.append(ion)

    # Default per-ion b values
    if per_ion_b and b_values is None:
        b_values = [10.0] * len(distinct_ions)
    if per_ion_b and len(b_values) != len(distinct_ions):
        raise ValueError(
            f"b_values length ({len(b_values)}) must match the number of "
            f"distinct ions ({len(distinct_ions)}): {distinct_ions}"
        )

    # Dynamically create Parameter attributes
    param_dict = {}
    if not per_ion_b:
        if not fix_T:
            param_dict['T'] = Parameter(default=T)
        if not fix_b:
            param_dict['b_other'] = Parameter(default=b_other)
    if not fix_v:
        param_dict['v'] = Parameter(default=v)
    for i, ion in enumerate(distinct_ions):
        param_dict[f'N_{ion}'] = Parameter(default=N_values[i])
    if per_ion_b:
        for i, ion in enumerate(distinct_ions):
            param_dict[f'b_{ion}'] = Parameter(default=b_values[i])

    class MultiVoigt1D(Fittable1DModel):
        n_inputs = 1
        n_outputs = 1

        # add parameters
        for pname, pval in param_dict.items():
            locals()[pname] = pval
        del pname, pval

        def __init__(self, **kwargs):
            self._per_ion_b = per_ion_b
            self._distinct_ions = list(distinct_ions)
            # if T is fixed, store as attribute
            if fix_T:
                self.T = float(T)
            if fix_b:
                self.b_other = float(kwargs.pop('b_other', b_other))
            if fix_v:
                self.v = float(kwargs.pop('v', v))
            if 'bounds' in kwargs and isinstance(kwargs['bounds'], dict):
                if per_ion_b:
                    kwargs['bounds'].pop('T', None)
                    kwargs['bounds'].pop('b_other', None)
                else:
                    if fix_b and 'b_other' in kwargs['bounds']:
                        kwargs['bounds'].pop('b_other', None)
                if fix_v and 'v' in kwargs['bounds']:
                    kwargs['bounds'].pop('v', None)
            # Pop priors dict before super().__init__ (astropy doesn't know about it)
            priors_in = kwargs.pop('priors', {})
            if isinstance(priors_in, dict):
                if per_ion_b:
                    priors_in.pop('T', None)
                    priors_in.pop('b_other', None)
                else:
                    if fix_T:
                        priors_in.pop('T', None)
                    if fix_b:
                        priors_in.pop('b_other', None)
                if fix_v:
                    priors_in.pop('v', None)
            self._priors = priors_in
            self._line_list = line_list
            self._z = float(z)
            self._f_lam_gamma_mass = {}
            ion_index_local = {}
            distinct_ions_local = []
            for line in line_list:
                info = SEARCH_LINES[SEARCH_LINES['tempname'] == line]
                ion = info['name'][0]
                self._f_lam_gamma_mass[line] = (
                    ion,
                    info['f'][0],
                    info['wave'][0],
                    info['Gamma'][0],
                    info['mass'][0]
                )
                if ion not in ion_index_local:
                    ion_index_local[ion] = len(distinct_ions_local)
                    distinct_ions_local.append(ion)
            self._line_to_ion_index = {
                line: ion_index_local[self._f_lam_gamma_mass[line][0]]
                for line in line_list
            }
            # Pre-convert all unit quantities to plain floats once
            self._line_constants = {}
            for line in line_list:
                ion, f, lam0, gamma, mass = self._f_lam_gamma_mass[line]
                lam0_val = lam0.to_value(u.Angstrom) if isinstance(lam0, u.Quantity) else float(lam0)
                gamma_val = gamma.to_value(u.Hz) if isinstance(gamma, u.Quantity) else float(gamma)
                f_val = f.value if isinstance(f, u.Quantity) else float(f)
                mass_val = mass.to_value(u.u) if hasattr(mass, 'to') else float(mass)
                nu0 = c_As / lam0_val
                self._line_constants[line] = (f_val, lam0_val, gamma_val, mass_val, nu0)
            super().__init__(**kwargs)

        def evaluate(self, x, *params):
            arr_x = np.asarray(x, dtype=float)
            tau_total = np.zeros_like(arr_x, dtype=float)

            n_ions = len(self._distinct_ions)

            if self._per_ion_b:
                # Parameter order: [v], N_ion1..N_ionK, b_ion1..b_ionK
                idx = 0
                if fix_v:
                    v_val = self.v
                else:
                    v_val = float(params[idx]); idx += 1
                N_args = params[idx:idx + n_ions]
                b_args = params[idx + n_ions:idx + 2 * n_ions]
            else:
                idx = 0
                if fix_T:
                    T_val = self.T
                else:
                    T_val = float(params[idx]); idx += 1
                if fix_b:
                    b_other_val = self.b_other
                else:
                    b_other_val = float(params[idx]); idx += 1
                if fix_v:
                    v_val = self.v
                else:
                    v_val = float(params[idx]); idx += 1
                N_args = params[idx:]

            one_plus_z_eff = (1.0 + self._z) * (1.0 + v_val / c_kms)
            wave_rest = arr_x / one_plus_z_eff
            nu_rest = c_As / wave_rest

            for line in self._line_list:
                ion_idx = self._line_to_ion_index[line]
                N_param = float(N_args[ion_idx])

                f_val, lam0_val, gamma_val, mass_val, nu0 = self._line_constants[line]

                if self._per_ion_b:
                    b_tot = float(b_args[ion_idx])
                else:
                    b_thermal = constant1 * np.sqrt(T_val) / np.sqrt(mass_val)
                    b_tot = np.sqrt(b_thermal**2 + b_other_val**2)

                doppler_width = b_tot * nu0 / c_kms
                if doppler_width == 0:
                    continue

                a_param = gamma_val / (4.0 * np.pi * doppler_width)
                nu_rest = c_As / wave_rest
                u_arg = (nu_rest - nu0) / doppler_width
                H = wofz(u_arg + 1j * a_param).real

                N_cm2 = N_param * multiple
                tau_total += N_cm2 * prefactor_tau * f_val / (sqrt_pi * doppler_width) * H

            return tau_total

    return MultiVoigt1D




def Continuum1D(num_knots=5, degree=3, x_domain=None, values=None, z=0.0,
                rest_wavelength=None, anchor_edges=False, knot_coef_bounds=(0.85, 1.15)):
    if not isinstance(num_knots, (int, float)):
        raise TypeError("num_knots must be numeric.")
    if num_knots <= 0:
        raise ValueError("num_knots must be greater than 0.")
    if not isinstance(degree, (int, float)):
        raise TypeError("degree must be numeric.")
    if degree < 1:
        raise ValueError("degree must be at least 1.")
    if rest_wavelength is None:
        raise ValueError("rest_wavelength is required.")
    if not isinstance(rest_wavelength, (int, float)):
        raise TypeError("rest_wavelength must be numeric.")
    if not isinstance(z, (int, float)):
        raise TypeError("z must be numeric.")
    if x_domain is not None:
        if len(x_domain) != 2:
            raise ValueError("x_domain must have exactly two elements.")
        if not all(isinstance(val, (int, float)) for val in x_domain):
            raise TypeError("x_domain values must be numeric.")
    if values is not None:
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError("values must be a sequence of numeric coefficients.")
        if not all(isinstance(val, (int, float)) for val in values):
            raise TypeError("values must be numeric.")
    if not isinstance(anchor_edges, bool):
        raise TypeError("anchor_edges must be a boolean.")
    if knot_coef_bounds is None:
        knot_coef_bounds = (0.85, 1.15)
    if (not isinstance(knot_coef_bounds, (list, tuple, np.ndarray))
            or len(knot_coef_bounds) != 2
            or not all(isinstance(val, (int, float)) for val in knot_coef_bounds)):
        raise TypeError("knot_coef_bounds must be a sequence of two numeric values (min, max).")
    knot_coef_min, knot_coef_max = map(float, knot_coef_bounds)
    if knot_coef_min >= knot_coef_max:
        raise ValueError("knot_coef_bounds must satisfy min < max.")

    n_interior = int(num_knots)
    spline_degree = int(degree)
    velocity_domain = (-500.0, 500.0) if x_domain is None else tuple(float(v) for v in x_domain)

    lam_center = float(rest_wavelength) * (1.0 + float(z))
    domain = tuple(lam_center * (1.0 + np.array(velocity_domain, dtype=float) / c_kms))

    total_knots = n_interior + 2 if anchor_edges else n_interior
    if values is not None:
        if len(values) != total_knots:
            raise ValueError("values length must match the total number of knots (including anchored edges).")
        default_coeffs = np.asarray(values, dtype=float)
    else:
        default_coeffs = 0.999 * np.ones(total_knots, dtype=float)

    default_locations = np.linspace(domain[0], domain[1], total_knots, dtype=float)

    knot_loc_names = [f'knot_loc_{i}' for i in range(total_knots)]
    knot_coef_names = [f'knot_coef_{i}' for i in range(total_knots)]

    param_dict = {}
    for idx, name in enumerate(knot_loc_names):
        if anchor_edges and (idx == 0 or idx == total_knots - 1):
            print(f"Anchoring knot location parameter '{name}' at {default_locations[idx]:.2f}")
            param_dict[name] = Parameter(default=default_locations[idx], fixed=True)
        else:
            print(f"Setting knot location parameter '{name}' with bounds ({domain[0]:.2f}, {domain[1]:.2f}), default {default_locations[idx]:.2f}")
            param_dict[name] = Parameter(default=default_locations[idx], min=domain[0], max=domain[1])

    for idx, name in enumerate(knot_coef_names):
        bounded_default = float(np.clip(default_coeffs[idx], knot_coef_min, knot_coef_max))
        print(
            f"Setting knot coefficient parameter '{name}' with bounds "
            f"({knot_coef_min:.2f}, {knot_coef_max:.2f})"
        )
        param_dict[name] = Parameter(
            default=bounded_default,
            min=knot_coef_min,
            max=knot_coef_max,
        )

    class ContinuumSpline1D(Fittable1DModel):
        n_inputs = 1
        n_outputs = 1
        for pname, param in param_dict.items():
            locals()[pname] = param
        del pname, param

        def __init__(self, **kwargs):
            fixed = dict(kwargs.pop('fixed', {}))
            # Pop priors dict before super().__init__ (astropy doesn't know about it)
            priors_in = kwargs.pop('priors', {})
            self._priors = priors_in if isinstance(priors_in, dict) else {}
            super().__init__(fixed=fixed, **kwargs)
            self._num_knots = n_interior
            self._total_knots = total_knots
            self._degree = spline_degree
            self._domain = domain
            self._velocity_domain = velocity_domain
            self._z = float(z)
            self._rest_wavelength = float(rest_wavelength)
            self._anchor_edges = anchor_edges
            self._knot_location_param_names = knot_loc_names
            self._knot_coeff_param_names = knot_coef_names

            name_to_index = {name: idx for idx, name in enumerate(self.param_names)}
            self._knot_location_indices = [name_to_index[name] for name in self._knot_location_param_names]
            self._knot_coeff_indices = [name_to_index[name] for name in self._knot_coeff_param_names]

        @property
        def knot_locations(self):
            params = np.asarray(self.parameters, dtype=float)
            return params[self._knot_location_indices]

        @property
        def knot_values(self):
            params = np.asarray(self.parameters, dtype=float)
            return params[self._knot_coeff_indices]

        def evaluate(self, x, *params):
            expected = 2 * self._total_knots
            if len(params) != expected:
                raise ValueError(f"Expected {expected} parameters (locations + coefficients), got {len(params)}.")

            arr_x = np.asarray(x, dtype=float)
            scalar_input = arr_x.ndim == 0
            eval_x = arr_x if not scalar_input else arr_x[None]

            params_arr = np.asarray(params, dtype=float)
            knot_locations = np.asarray(params_arr[:self._total_knots], dtype=float).reshape(-1)
            knot_coeffs = np.asarray(params_arr[self._total_knots:], dtype=float).reshape(-1)

            sort_idx = np.argsort(knot_locations, kind='mergesort')
            knot_locations = knot_locations[sort_idx]
            knot_coeffs = knot_coeffs[sort_idx]

            domain_low, domain_high = self._domain
            span = max(domain_high - domain_low, 1.0)
            min_spacing = max(span * 1e-12, 1e-6)

            if knot_locations.size >= 2:
                adjusted_locations = knot_locations.copy()

                if self._anchor_edges:
                    adjusted_locations[0] = domain_low
                    adjusted_locations[-1] = domain_high

                for idx in range(1, adjusted_locations.size):
                    prev_allowed = adjusted_locations[idx - 1] + min_spacing
                    if adjusted_locations[idx] <= prev_allowed:
                        adjusted_locations[idx] = prev_allowed

                if self._anchor_edges and adjusted_locations[-1] != domain_high:
                    adjusted_locations[-1] = domain_high

                for idx in range(adjusted_locations.size - 2, -1, -1):
                    max_allowed = adjusted_locations[idx + 1] - min_spacing
                    if adjusted_locations[idx] > max_allowed:
                        adjusted_locations[idx] = max_allowed

                adjusted_locations = np.clip(adjusted_locations, domain_low, domain_high)

                if self._anchor_edges:
                    adjusted_locations[0] = domain_low
                    adjusted_locations[-1] = domain_high

                knot_locations = adjusted_locations

            unique_locations, unique_indices = np.unique(knot_locations, return_index=True)
            knot_locations = unique_locations
            knot_coeffs = knot_coeffs[unique_indices]

            if (knot_locations.size + 2) < self._degree + 1:
                raise ValueError("Insufficient unique knot locations for the spline degree.")
            if np.any(np.diff(knot_locations) <= 0):
                raise ValueError("Knot locations must be strictly increasing.")

            spline = make_interp_spline(knot_locations, knot_coeffs, k=self._degree, bc_type='clamped')
            interp_vals = spline(eval_x)

            domain_low, domain_high = knot_locations[0], knot_locations[-1]
            mask = (eval_x < domain_low) | (eval_x > domain_high)
            if np.any(mask):
                interp_vals = np.asarray(interp_vals)
                interp_vals[mask] = 1.0

            if scalar_input:
                return float(interp_vals[0])
            return interp_vals

    return ContinuumSpline1D


# class Shift1D(Fittable1DModel):
#     offset = Parameter(default=0.0)

#     def __init__(self, offset=0.0, shift_range=(900, 1000), fixed_offset=False, **kwargs):
#         if fixed_offset:
#             super().__init__(offset=Parameter(value=offset, fixed=True), **kwargs)
#         else:
#             super().__init__(offset=offset, **kwargs)
#         self._shift_range = shift_range

#     def evaluate(self, x, offset):
#         low, high = self._shift_range
#         mask = (x >= low) & (x <= high)
#         shifted = np.copy(x)
#         shifted[mask] += offset
#         return shifted


def model_1E(fix=True):
    """
    Create an Exponential1D model.

    Parameters:
    - fix (bool): Whether to fix model parameters.

    Returns:
    - e1_init (Exponential1D): Exponential1D model.
    """
    e1_init = models.Exponential1D(amplitude=1., tau=-1, fixed={'tau': fix, 'amplitude': fix}, name='exponential')
    return e1_init

@custom_model
def FixedExponential1D(x):
    """
    Create an Exponential1D model.

    Parameters:
    - fix (bool): Whether to fix model parameters.

    Returns:
    - e1_init (Exponential1D): Exponential1D model.
    """
    e1_init = np.exp(-x)
    return e1_init

# def model_1V(params, bounds, info_dict):
#     """
#     Create a Voigt profile model for a specific spectral line with given parameters.

#     Parameters:
#     - params (dict): Dictionary of parameter values, including 'N' (column density), 'T' (temperature),
#                      'b_other' (Doppler parameter), and 'v' (velocity).
#     - bounds (list): Parameter bounds.
#     - info_dict (dict): Information dictionary containing 'z' (redshift), 'line' (spectral line name).

#     Returns:
#     - v_init (custom model): A custom Voigt profile model initialized with the provided parameters.
#     """
#     N_start = params['N']
#     T_start = params['T']
#     bo_start = params['b_other']
#     v_start = params['v']
#     z = info_dict['z']
#     line = info_dict['line']
#     x_offset_bool = info_dict['x_offset_bool']
#     lambda_0 = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['wave'][0]
#     gamma = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['Gamma'][0]
#     f = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['f'][0]
#     mass = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['mass'][0]
#     v_init = PhysicalVoigt1D(N=N_start, T=T_start, b_other=bo_start, v=v_start, z=z,
#                              lambda_0=lambda_0, gamma=gamma, f=f, x_offset=0.0, mass=mass,
#                              bounds=bounds, name=str(line) + '__z=' + str(z))
    
#     v_init.x_offset.fixed = x_offset_bool
#     # v_init.z.fixed = True
#     # v_init.lambda_0.fixed = True
#     # v_init.mass.fixed = True
#     # v_init.gamma.fixed = True
#     # v_init.f.fixed = True
#     return v_init


def generic_exp(v_init):
    """
    Create a combined model consisting of a Voigt profile and an Exponential model.

    Parameters:
    - v_init (custom model): A custom Voigt profile model.

    Returns:
    - ve_init (custom model): A combined Voigt and Exponential model.
    """
    e1_init = model_1E(fix=True)
    ve_init = v_init | e1_init
    return ve_init


def Exponentiate1D(v_init):
    """
    Create a combined model consisting of a Voigt profile and an Exponential model.

    Parameters:
    - v_init (custom model): A custom Voigt profile model.

    Returns:
    - ve_init (custom model): A combined Voigt and Exponential model.
    """
    e1_init = FixedExponential1D(name='exponential')
    ve_init = v_init | e1_init
    return ve_init


def collect_priors(model):
    """
    Collect the priors dict from a (possibly compound) astropy model.

    For simple models with a ``_priors`` attribute, returns a dict mapping
    each parameter name to its prior type (e.g. ``'flat'`` or ``'jeffreys'``).

    For compound models, uses astropy's internal ``_param_map`` to walk the
    leaf models and aggregate their ``_priors`` dicts, keyed by the compound
    model's parameter names (e.g. ``'N_HI_0'``, ``'T_1'``).

    Parameters
    ----------
    model : astropy Model
        A simple or compound astropy model whose leaf models may carry
        ``_priors`` dicts (set by :func:`Absorber1D` or :func:`Continuum1D`).

    Returns
    -------
    priors : dict
        ``{compound_param_name: prior_type}`` for every parameter.
        Parameters without an explicit prior default to ``'flat'``.
    """
    # Simple (non-compound) model
    if not hasattr(model, '_leaflist'):
        raw = getattr(model, '_priors', {})
        return {pname: raw.get(pname, 'flat') for pname in model.param_names}

    # Compound model — walk _param_map
    # Accessing param_names ensures _param_map is lazily initialised
    _ = model.param_names
    priors = {}
    for cpd_name, (leaf_idx, orig_name) in model._param_map.items():
        leaf = model._leaflist[leaf_idx]
        leaf_priors = getattr(leaf, '_priors', {})
        priors[cpd_name] = leaf_priors.get(orig_name, 'flat')
    return priors