from header import *  # Importing necessary dependencies

def find_nearest_index(array, number, tolerance=0.0):
    """
    Find the index of the nearest value in an array to a given number,
    ignoring values within the specified tolerance.

    Parameters:
    - array (numpy.ndarray): The input array.
    - number (float): The number to which we want to find the nearest value.
    - tolerance (float): Minimum absolute difference to consider (default 0.0).

    Returns:
    - nearest_index (int or None): The index of the nearest value in the array, or None if none found.
    """
    absolute_diff = np.abs(array - number)
    mask = absolute_diff > tolerance
    if not np.any(mask):
        return None
    nearest_index = np.argmin(np.where(mask, absolute_diff, np.inf))
    return nearest_index

def find_nearest_index_above(array, number, tolerance=0.0):
    """
    Find the index of the nearest value in an array that is greater than a given number
    by at least the specified tolerance.

    Parameters:
    - array (numpy.ndarray): The input array.
    - number (float): The number to which we want to find the nearest value.
    - tolerance (float): Minimum difference above the number (default 0.0).

    Returns:
    - nearest_index (int or None): The index of the nearest value in the array that is greater than the number.
    """
    indices_above = np.where(array > number + tolerance)[0]
    if len(indices_above) == 0:
        return None
    nearest_index = indices_above[np.argmin(np.abs(array[indices_above] - number))]
    return nearest_index

def find_nearest_index_below(array, number, tolerance=0.0):
    """
    Find the index of the nearest value in an array that is less than a given number
    by at least the specified tolerance.

    Parameters:
    - array (numpy.ndarray): The input array.
    - number (float): The number to which we want to find the nearest value.
    - tolerance (float): Minimum difference below the number (default 0.0).

    Returns:
    - nearest_index (int or None): The index of the nearest value in the array that is less than the number.
    """
    indices_below = np.where(array < number - tolerance)[0]
    if len(indices_below) == 0:
        return None
    nearest_index = indices_below[np.argmin(np.abs(array[indices_below] - number))]
    return nearest_index

def find_nearest_value_above(array, number, tolerance=0.0):
    """
    Find the nearest value in an array that is greater than a given number
    by at least the specified tolerance.

    Parameters:
    - array (numpy.ndarray): The input array.
    - number (float): The number to which we want to find the nearest value.
    - tolerance (float): Minimum difference above the number (default 0.0).

    Returns:
    - nearest_value (float or None): The nearest value in the array that is greater than the number.
    """
    indices_above = np.where(array > number + tolerance)[0]
    if len(indices_above) == 0:
        return None
    nearest_value = array[indices_above[np.argmin(np.abs(array[indices_above] - number))]]
    return nearest_value

def find_nearest_value_below(array, number, tolerance=0.0):
    """
    Find the nearest value in an array that is less than a given number
    by at least the specified tolerance.

    Parameters:
    - array (numpy.ndarray): The input array.
    - number (float): The number to which we want to find the nearest value.
    - tolerance (float): Minimum difference below the number (default 0.0).

    Returns:
    - nearest_value (float or None): The nearest value in the array that is less than the number.
    """
    indices_below = np.where(array < number - tolerance)[0]
    if len(indices_below) == 0:
        return None
    nearest_value = array[indices_below[np.argmin(np.abs(array[indices_below] - number))]]
    return nearest_value

def remove_close_values(arr, min_distance):
    """
    Remove values from an array that are closer to each other than a specified minimum distance.

    Parameters:
    - arr (numpy.ndarray): The input array.
    - min_distance (float): The minimum allowed distance between values.

    Returns:
    - cleaned_arr (numpy.ndarray): The array with close values removed.
    - unremoved_indices (numpy.ndarray): The indices of the values that were not removed.
    """
    sorted_indices = np.argsort(arr)  # Get the indices that would sort the array
    sorted_arr = arr[sorted_indices]  # Sort the array based on the indices
    remove_indices = []

    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] - sorted_arr[i-1] < min_distance:
            remove_indices.append(i-1)  # Mark indices of values that are too close

    cleaned_arr = np.delete(sorted_arr, remove_indices)  # Remove the marked values
    unremoved_indices = np.delete(sorted_indices, remove_indices)  # Get the indices of unremoved values
    return cleaned_arr, unremoved_indices

def find_first_occurrences(arr):
    """
    Find the indices of the first occurrences of elements in an array.

    Parameters:
    - arr (numpy.ndarray): The input array.

    Returns:
    - first_occurrence_indices (numpy.ndarray): An array with the indices of the first occurrences of elements.
    """
    indices_dict = {}
    first_occurrence_indices = np.zeros_like(arr, dtype=int) - 1  # Initialize with -1
    for i, element in enumerate(arr):
        if element not in indices_dict:
            indices_dict[element] = i  # Store the index of the first occurrence of each unique element
            first_occurrence_indices[i] = -1
        else:
            first_occurrence_indices[i] = indices_dict[element]  # Set index for non-first occurrences
    return first_occurrence_indices

def pick_evenly_spaced_values(array, n):
    """
    Pick n evenly spaced values from an array.

    Parameters:
    - array (numpy.ndarray): The input array.
    - n (int): The number of values to pick.

    Returns:
    - indices (numpy.ndarray): The indices of the picked values.
    - picked_values (numpy.ndarray): The picked values from the input array.
    """
    n = min(n, len(array))  # Ensure n is not greater than the length of the array
    indices = np.linspace(0, len(array) - 1, n, dtype=int)  # Generate evenly spaced indices
    return indices, array[indices]  # Return the indices and the corresponding values

def to_velocity(wave, line, z):
    """
    Calculate the velocity based on observed and true wavelengths and redshift.

    Parameters:
    - wave (float): The true wavelength.
    - line (float): The observed wavelength.
    - z (float): The redshift.

    Returns:
    - v (float): The calculated velocity.
    """
    observed = line * u.Angstrom * (1 + z)
    true = wave
    v = c * (true - observed) / true  # Calculate velocity using the speed of light constant
    return v

def to_wavelength(v, line, z):
    """
    Calculate the observed wavelength based on velocity and true wavelength.

    Parameters:
    - v (float): The velocity.
    - line (float): The true wavelength.
    - z (float): The redshift.

    Returns:
    - wave (float): The calculated observed wavelength.
    """
    wave = line * u.Angstrom * (1 + z)
    observed = wave - (v * wave / c)  # Calculate observed wavelength using the speed of light constant
    return observed

def log_uniform(a, b, size):
    """
    Generate an array of size random numbers uniformly distributed in log space between a and b.

    Parameters:
    - a (float): The lower bound of the log-uniform distribution.
    - b (float): The upper bound of the log-uniform distribution.
    - size (int): The number of random values to generate.

    Returns:
    - log_uniform_values (numpy.ndarray): An array of random values in log-uniform distribution.
    """
    return 10**np.random.uniform(np.log10(a), np.log10(b), size=size)  # Generate log-uniform random values

def split_string_by_second_uppercase(s):
    """
    Split a string into two parts at the position of the second uppercase character.
    
    Args:
        s (str): The input string.
    
    Returns:
        tuple: A tuple containing two strings:
            - first_part: The substring before the second uppercase character.
            - second_part: The substring starting from the second uppercase character.
              If there is no second uppercase character, the second part is an empty string.
    """
    first_upper_found = False  # Initialize a flag to track the first uppercase character found
    split_index = -1  # Initialize the split index to -1 (indicating no split)

    for i, char in enumerate(s):
        if char.isupper():
            if first_upper_found:
                split_index = i  # Store the index of the second uppercase character
                break
            else:
                first_upper_found = True  # Mark that the first uppercase character has been found

    if split_index >= 0:
        # If a split index was found, split the string into two parts
        first_part = s[:split_index]
        second_part = s[split_index:]
        return first_part, second_part
    else:
        # If no second uppercase character was found, return the original string and an empty string
        return s, ''

def check_values_in_arrays(A, B):
    # Create a set from array B for faster lookup
    set_B = B#set(B)
    
    # Initialize two empty lists for booleans and indices
  #  bool_result = np.full()
    index_result = []
    
    # Iterate through the elements in A
    for value in set_B:
        for index, value2 in enumerate(A):
            if value == value2:
            #    bool_result.append(True)
                index_result.append(index)
          #  else:
          #      bool_result.append(False)
    
    return index_result

def ReLU(x):
    return x * (x > 0)

def gen_vrange_mask(velocity_spectrum, vranges):
    """
    Given a velocity spectrum (astropy Quantity array in km/s) and a list
    of vrange pairs [(vmin, vmax), ...] (each as astropy Quantity), return
    a boolean mask selecting points inside any of the ranges.

    Parameters:
    - velocity_spectrum: astropy.units.Quantity array (e.g., km/s)
    - vranges: list of (vmin, vmax) pairs, each a Quantity with compatible units

    Returns:
    - mask: numpy boolean array with same length as velocity_spectrum
    """
    mask_temp = np.zeros_like(getattr(velocity_spectrum, 'value', velocity_spectrum), dtype=bool)
    # allow empty vranges
    if vranges is None:
        return mask_temp
    for vpair in vranges:
        if vpair is None:
            continue
        print(vpair)
        vmin, vmax = vpair
        # comparisons work with astropy quantities
        mask_range = np.logical_and(velocity_spectrum > vmin, velocity_spectrum < vmax)
        mask_temp = np.logical_or(mask_temp, mask_range)
    return mask_temp


def safe_sigma(sigma_array, floor_fraction=1e-3, min_floor=1e-8):
    """
    Return a sigma array with a sensible non-zero floor to avoid divide-by-zero
    and extreme z-values when computing Gaussian tail/CDF terms.

    Strategy:
    - Compute the median of positive sigma values (if any).
    - Set floor = max(median * floor_fraction, min_floor).
    - Clip the input sigma_array at that floor.

    Parameters:
    - sigma_array: 1D numpy array of uncertainties (may contain zeros or negatives).
    - floor_fraction: fraction of the median positive sigma to use as floor.
    - min_floor: absolute minimum floor.

    Returns:
    - clipped sigma_array (numpy array)
    """
    s = np.array(sigma_array, copy=True)
    # Force negative uncertainties to zero (match existing behavior elsewhere)
    s[s < 0] = 0.0
    positive = s[s > 0]
    if positive.size > 0:
        med = np.median(positive)
    else:
        med = min_floor
    floor = max(med * floor_fraction, min_floor)
    return np.clip(s, floor, None)


def mask_for_line(spectral_axis, line_wave, z, vranges):
    """
    Convenience wrapper: compute velocity array for a given observed spectral
    axis and transition (line_wave) at redshift z and return the boolean mask
    for the provided vranges.

    Parameters:
    - spectral_axis: array-like of observed wavelengths (astropy.Quantity or plain array compatible with to_velocity)
    - line_wave: transition rest wavelength (float or Quantity, typically in Angstrom)
    - z: redshift (float)
    - vranges: list of (vmin, vmax) pairs (astropy Quantities, e.g., km/s)

    Returns:
    - mask: boolean numpy array
    """
    velocity_spectrum = to_velocity(spectral_axis, line_wave, z).to('km/s')
    return gen_vrange_mask(velocity_spectrum, vranges)


def make_continuum_node_veto(model, mask_dict, param_keyword='knot_loc'):
    """
    Build a callable that flags continuum knot parameters entering forbidden
    velocity windows defined in mask_dict['redshifts'][...]['continuum_node_vrange'].

    Parameters
    ----------
    model : astropy.modeling.Model
        The compound model whose parameter ordering matches the MCMC state.
    mask_dict : dict
        Mask configuration expected to contain optional continuum_node_vrange
        entries with velocity intervals per spectral line.
    param_keyword : str, optional
        Substring used to identify continuum knot-location parameters.

    Returns
    -------
    callable or None
        Function taking a parameter vector and returning True if a violation
        occurs. Returns None when no forbidden intervals are configured or no
        matching parameters are present.
    """

    if mask_dict is None:
        return None

    redshift_entries = mask_dict.get('redshifts', [])
    if not redshift_entries:
        return None

    try:
        from astropy.constants import c as const_c
        from astropy import units as u
    except Exception:
        return None

    try:
        from line_info import SEARCH_LINES
    except Exception:
        SEARCH_LINES = None

    c_kms = const_c.to(u.km / u.s).value

    knot_indices = [i for i, name in enumerate(getattr(model, 'param_names', []))
                    if param_keyword in name]
    if not knot_indices:
        return None

    intervals = []
    for entry in redshift_entries:
        z_entry = entry.get('z')
        cont_cfg = entry.get('continuum_node_vrange')
        if not cont_cfg or z_entry is None:
            continue

        for line_name, ranges in cont_cfg.items():
            rest_wavelength = None
            if SEARCH_LINES is not None:
                rows = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]
                if len(rows) > 0:
                    rest_wavelength = rows['wave'][0]
            if rest_wavelength is None:
                continue

            lambda_center = rest_wavelength * (1.0 + float(z_entry))
            for vr in ranges:
                if not isinstance(vr, (list, tuple)) or len(vr) != 2:
                    continue
                vlow, vhigh = vr
                try:
                    vlow = vlow.to(u.km / u.s).value if hasattr(vlow, 'to') else float(vlow)
                    vhigh = vhigh.to(u.km / u.s).value if hasattr(vhigh, 'to') else float(vhigh)
                except Exception:
                    continue
                lam_low = lambda_center * (1.0 + vlow / c_kms)
                lam_high = lambda_center * (1.0 + vhigh / c_kms)
                lo, hi = (min(lam_low, lam_high), max(lam_low, lam_high))
                intervals.append((lo, hi))

    if not intervals:
        return None

    intervals = np.asarray(intervals, dtype=float)
    print(f"Continuum node veto: {len(intervals)} forbidden intervals configured.")
    print(intervals)
    def _violates(pars):
        params = np.asarray(pars, dtype=float)
        knot_values = params[knot_indices]
        for value in knot_values:
            if not np.isfinite(value):
                continue
            if np.any((value >= intervals[:, 0]) & (value <= intervals[:, 1])):
                return True
        return False

    return _violates
