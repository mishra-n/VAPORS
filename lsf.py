from header import *  # Importing necessary dependencies
from line_info import *  # Importing additional dependencies from other modules
from helpers import *  # Importing helper functions from a different module
def getCOSlsf(wave, lifetimePosition=4, vblue=False, skip_print=True):
    """
    Get the COS Line Spread Function (LSF) for a given wavelength and lifetime position.

    Parameters:
    - wave (float): The wavelength for which to retrieve the LSF.
    - lifetimePosition (int, optional): The COS lifetime position (default is 4).
    - vblue (bool, optional): Flag indicating if it's a very blue G130M wavelength (default is False).
    - skip_print (bool, optional): Flag to skip print statements (default is True).

    Returns:
    - lsf (numpy.ndarray): The Line Spread Function as a numpy array with 'pix' and 'kernel' columns.
    """
    if vblue:
        if not skip_print:
            print('Very blue G130M not yet implemented.')
        sys.exit()

    # Decide whether the G130M or G160M LSF is more appropriate based on wavelength
    if wave < 1400:
        grating = 'G130M'
        default_wave = 1300
    else:
        grating = 'G160M'
        default_wave = 1600

    if lifetimePosition <= 4:
        directory = '/' + os.environ['PYABS'] + '/absorption/COS/lp{}/'.format(lifetimePosition)

        if lifetimePosition == 1:
            if not skip_print:
                print(lifetimePosition)
            filename = '{}{}_lp1.npy'.format(directory, grating)
            lsfTable = np.load(filename)

            # Find the LSF column closest to the desired wavelength
            enumeratedWaves_string = lsfTable.dtype.names[1:]
            enumeratedWaves = np.array(enumeratedWaves_string).astype(float)
            waveDistance = np.abs(enumeratedWaves - wave)
            index = np.argmin(waveDistance) - 1
            lsfColumn = enumeratedWaves_string[index]
            if not skip_print:
                print('Getting LSF for LP1 at {}'.format(enumeratedWaves_string[index]))
            
        elif (lifetimePosition == 2) | (lifetimePosition == 3) | (lifetimePosition == 4):
            filename = '{}fuv_{}_{}_lsf.npy'.format(directory, grating, default_wave)
            lsfTable = np.load(filename)

            # Find the LSF column closest to the desired wavelength
            enumeratedWaves_string = lsfTable.dtype.names[1:]
            enumeratedWaves = np.array(enumeratedWaves_string).astype(float)
            waveDistance = np.abs(enumeratedWaves - wave)
            index = np.argmin(waveDistance) - 1
            lsfColumn = enumeratedWaves_string[index]

        lsf = np.zeros(len(lsfTable), dtype={'names':('pix', 'kernel'),
                                           'formats':(float, float)})
        lsf['pix'] = lsfTable['pix']
        lsf['kernel'] = lsfTable[lsfColumn]

        # Normalize the kernel
        lsf['kernel'] = lsf['kernel'] / np.sum(lsf['kernel'])

        return lsf
   
    else:
        print('Lifetime position does not exist.')
        sys.exit()


data = np.loadtxt('/home/mishran/Documents/VAPORS/FUSE_LSF.csv', delimiter=',')
# Interpolate the kernel using scipy's 1D interpolation

# Define the range for sampling
sample_range = np.arange(-39, 40)

lsf_FUSE = np.zeros(len(sample_range), dtype={'names': ('pix', 'kernel'),
                                    'formats': (float, float)})

# Create an interpolation function
interp_func = interp.interp1d(data[:, 0], data[:, 1], kind='linear', bounds_error=False, fill_value=0)

# Sample the kernel at the specified range
lsf_FUSE['kernel'] = interp_func(sample_range)
lsf_FUSE['pix'] = sample_range


# Load STIS LSF data and calculate pixel scale ratio
STIS_LSF_FILE = '/home/mishran/Documents/VAPORS/other_test_data/of8g59020_x1d.fits'
STIS_DATA_FILE = '/home/mishran/Documents/STIS/J2135-5316_final_abscal_new.fits'

STIS_LSF_CONFIG = {
    'G230M': {
        'file': '/home/mishran/Documents/VAPORS/STIS_G230M_LSF.txt',
        'skiprows': 1,
        'apertures': ['0.1', '0.2', '0.5', '2.0'],
        'default_aperture': '0.5',
    },
    'E230M': {
        'file': '/home/mishran/Documents/VAPORS/STIS_E230M_LSF.txt',
        'skiprows': 2,
        'apertures': ['0.1x0.03', '0.2x0.06', '0.2x0.2', '6x0.2'],
        'default_aperture': '0.2x0.2',
    }
}

# Load the text files with kernels for each grating
STIS_KERNELS = {}
STIS_LSF_PIX = {}
for grating, cfg in STIS_LSF_CONFIG.items():
    stis_txt_data = np.loadtxt(cfg['file'], skiprows=cfg['skiprows'])
    STIS_LSF_PIX[grating] = stis_txt_data[:, 0]
    STIS_KERNELS[grating] = {}
    for i, aperture in enumerate(cfg['apertures']):
        # Create interpolator for each aperture
        # The text file has columns: pixel, then aperture-specific kernels
        STIS_KERNELS[grating][aperture] = interp.interp1d(
            stis_txt_data[:, 0], stis_txt_data[:, i + 1],
            kind='linear', bounds_error=False, fill_value=0
        )

# Calculate pixel scale ratio using all orders
def calculate_stis_scale_ratio():
    try:
        # Read LSF reference file - extract ALL orders
        with fits.open(STIS_LSF_FILE) as hdul:
            data_lsf = hdul[1].data
            all_wave_lsf = data_lsf['WAVELENGTH']  # Shape: (30, 1024)
        
        # Read Data file
        with fits.open(STIS_DATA_FILE) as hdul:
            data_obs = hdul[1].data
            # print(f"STIS Data Columns: {data_obs.columns.names}")
            if 'WAVELENGTH' in data_obs.columns.names:
                wave_obs = data_obs['WAVELENGTH']
            elif 'WAVE' in data_obs.columns.names:
                wave_obs = data_obs['WAVE']
            elif 'wave' in data_obs.columns.names:
                wave_obs = data_obs['wave']
            else:
                # Fallback: try accessing 'WAVE' anyway, as it worked in inspection script
                try:
                    wave_obs = data_obs['WAVE']
                except KeyError:
                    print(f"Warning: Could not find wavelength column in STIS data file. Columns: {data_obs.columns.names}")
                    return None

            if len(wave_obs.shape) > 1:
                wave_obs = wave_obs[0]

        # Combine wavelengths from all orders for comprehensive coverage
        combined_waves = []
        combined_scales = []
        combined_wave_mids = []
        
        for order_idx in range(len(all_wave_lsf)):
            wave_lsf_order = all_wave_lsf[order_idx]
            scale_lsf_order = np.diff(wave_lsf_order)
            wave_mid_lsf_order = (wave_lsf_order[1:] + wave_lsf_order[:-1]) / 2
            
            combined_waves.append(wave_lsf_order)
            combined_scales.append(scale_lsf_order)
            combined_wave_mids.append(wave_mid_lsf_order)
        
        # Create unified LSF wavelength scale (all orders combined and sorted)
        all_wave_mids_lsf = np.concatenate(combined_wave_mids)
        all_scales_lsf = np.concatenate(combined_scales)
        
        # Sort by wavelength to create smooth interpolation function
        sort_idx = np.argsort(all_wave_mids_lsf)
        all_wave_mids_lsf = all_wave_mids_lsf[sort_idx]
        all_scales_lsf = all_scales_lsf[sort_idx]
        
        # Calculate scales for observed spectrum
        scale_obs = np.diff(wave_obs)
        wave_mid_obs = (wave_obs[1:] + wave_obs[:-1]) / 2
        
        # Interpolate LSF scale to Obs wavelengths across full range
        scale_lsf_interp = np.interp(wave_mid_obs, all_wave_mids_lsf, all_scales_lsf, 
                                      left=all_scales_lsf[0], right=all_scales_lsf[-1])
        
        ratio = scale_obs / scale_lsf_interp
        
        # Create interpolation function for ratio with better extrapolation
        ratio_func = interp.interp1d(wave_mid_obs, ratio, kind='linear', bounds_error=False, 
                                     fill_value=(ratio[0], ratio[-1]))
        return ratio_func

    except Exception as e:
        print(f"Error calculating STIS pixel scale ratio: {e}")
        return None

STIS_SCALE_RATIO_FUNC = calculate_stis_scale_ratio()

def getFUSElsf(wave):
    """
    Get the FUSE Line Spread Function (LSF) for a given wavelength.

    Parameters:
    - wave (float): The wavelength for which to retrieve the LSF.

    Returns:
    - lsf (numpy.ndarray): The Line Spread Function as a numpy array with 'pix' and 'kernel' columns.
    """

    return lsf_FUSE

def getSTISlsf(wave, aperture=None, grating='E230M'):
    """
    Get the STIS Line Spread Function (LSF) for a given wavelength, aperture, and grating.

    Parameters:
    - wave (float): The wavelength for which to retrieve the LSF.
    - aperture (float or str, optional): The aperture size (default depends on grating).
    - grating (str, optional): The grating type (default is 'E230M').

    Returns:
    - lsf (numpy.ndarray): The Line Spread Function as a numpy array with 'pix' and 'kernel' columns.
    """
    
    if grating not in STIS_KERNELS:
        print(f"Unsupported STIS grating: {grating}. Supported: {list(STIS_KERNELS.keys())}")
        sys.exit()

    cfg = STIS_LSF_CONFIG[grating]

    if aperture is None:
        aperture_str = cfg['default_aperture']
    elif isinstance(aperture, (int, float)):
        if grating == 'G230M':
            aperture_str = f"{float(aperture):.1f}"
        else:
            # Map common numeric apertures to E230M names
            if np.isclose(aperture, 0.1):
                aperture_str = '0.1x0.03'
            elif np.isclose(aperture, 0.2):
                aperture_str = '0.2x0.2'
            elif np.isclose(aperture, 6.0):
                aperture_str = '6x0.2'
            else:
                aperture_str = cfg['default_aperture']
    else:
        aperture_str = str(aperture)

    if aperture_str not in STIS_KERNELS[grating]:
        print(f"Aperture not supported for {grating}: {aperture_str}. Supported: {cfg['apertures']}")
        sys.exit()
    
    kernel_interp = STIS_KERNELS[grating][aperture_str]

    # Get the pixel scale ratio at this wavelength
    if STIS_SCALE_RATIO_FUNC is not None:
        ratio = STIS_SCALE_RATIO_FUNC(wave)
    else:
        ratio = 1.0 # Fallback if calculation failed

    # The ratio is (Obs Pixel Size) / (LSF Pixel Size)
    # If ratio < 1 (e.g. 0.35), it means Obs pixels are smaller than LSF pixels? 
    # Wait, let's re-verify.
    # Scale = d_lambda / d_pixel.
    # Scale_Obs ~ 0.05 Ang/pix. Scale_LSF ~ 0.15 Ang/pix.
    # Ratio = 0.05 / 0.15 = 0.33.
    # This means 1 Obs pixel covers 0.33 Angstroms (hypothetically), while 1 LSF pixel covers 1 Angstrom.
    # So 1 LSF pixel corresponds to 3 Obs pixels.
    # The LSF is defined in terms of LSF pixels.
    # We want the LSF in terms of Obs pixels.
    # If the LSF has a width of 1 LSF pixel, it should have a width of 3 Obs pixels.
    # So we need to stretch the LSF.
    # If we evaluate the LSF at x_obs (in Obs pixels), the corresponding coordinate in LSF pixels is x_lsf = x_obs * ratio.
    # Example: x_obs = 3. x_lsf = 3 * 0.33 = 1. Correct.
    
    # Define the range of pixels to evaluate
    # We want enough range to cover the LSF.
    # Original range was -34 to 34 (LSF pixels).
    # In Obs pixels, this would be -34/ratio to 34/ratio.
    # If ratio is 0.35, range is approx -100 to 100.
    
    # Let's pick a safe range.
    pix_range = 100 
    lsf_STIS = np.zeros(2 * pix_range + 1, dtype={'names': ('pix', 'kernel'),
                                    'formats': (float, float)})
    
    lsf_STIS['pix'] = np.arange(-pix_range, pix_range + 1)
    
    # Calculate corresponding LSF coordinates
    # NOTE: STIS LSF files use "Rel pixel" coordinates (fractional detector pixels).
    # The coordinates are already in detector-pixel units, just oversampled.
    # Interpolation handles the fractional sampling without additional scaling.
    lsf_coords = lsf_STIS['pix'] * ratio
    
    # Evaluate kernel
    lsf_STIS['kernel'] = kernel_interp(lsf_coords)

    # Normalize the kernel
    lsf_STIS['kernel'] = lsf_STIS['kernel'] / np.sum(lsf_STIS['kernel'])

    return lsf_STIS


# def gen_wave_indices(wave_arr, z_list, line_lists, d_lambda=5):
#     """
#     Generate wavelength indices and values for a given redshift and line list.

#     Parameters:
#     - wave_arr (numpy.ndarray): The wavelength array.
#     - z_list (list): List of redshifts.
#     - line_lists (list): List of line lists for each redshift.
#     - d_lambda (float, optional): Minimum allowed wavelength difference (default is 5 Angstroms).

#     Returns:
#     - index_list (numpy.ndarray): Array of wavelength indices.
#     - wave_list (numpy.ndarray): Array of corresponding wavelengths.
#     """
#     index_list = []
#     wave_list = []
    
#     for z, line_list in zip(z_list, line_lists):
#         for line in line_list:
#             wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['wave'][0] * u.Angstrom
#             wave_list.append(wavelength.to('Angstrom').value * (1 + z))
#             index_list.append(find_nearest_index(wave_arr.to('Angstrom').value, wavelength.to('Angstrom').value * (1 + z)))

#     wave_list = np.array(wave_list) * u.Angstrom
#     index_list = np.array(index_list)
#     wave_list, unremoved_indices = remove_close_values(wave_list, d_lambda * u.Angstrom)
#     index_list = index_list[unremoved_indices]

#     return index_list, wave_list

def gen_wave_indices(wave_arr, mask_dict, d_lambda=10):
    """
    Generate wavelength indices and values for a given redshift and line list.

    Parameters:
    - wave_arr (numpy.ndarray): The wavelength array.
    - z_list (list): List of redshifts.
    - line_lists (list): List of line lists for each redshift.
    - d_lambda (float, optional): Minimum allowed wavelength difference (default is 5 Angstroms).

    Returns:
    - index_list (numpy.ndarray): Array of wavelength indices.
    - wave_list (numpy.ndarray): Array of corresponding wavelengths.
    """
    index_list = []
    wave_list = []
    
    for entry in mask_dict['redshifts']:
        z = entry['z']
        # print(z)
        for line in entry['lines']:
            # print(line)
            wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['wave'][0] * u.Angstrom
            wave_list.append(wavelength.to('Angstrom').value * (1 + z))
            index_list.append(find_nearest_index(wave_arr.to('Angstrom').value, wavelength.to('Angstrom').value * (1 + z)))

    wave_list = np.array(wave_list) * u.Angstrom
    index_list = np.array(index_list)
    # wave_list, unremoved_indices = remove_close_values(wave_list, d_lambda * u.Angstrom)
    # index_list = index_list[unremoved_indices]

    return index_list, wave_list


def compute_eval_indices(wave_arr, wave_list, wave_list_indices, d_lambda=10):
    """
    Compute the pixel indices covered by LSF convolution groups (with padding),
    matching the grouping and padding logic in genLSF.

    Parameters:
    - wave_arr (numpy.ndarray): The wavelength array (unitless, in Angstroms).
    - wave_list (astropy Quantity array): Wavelengths for convolution.
    - wave_list_indices (numpy.ndarray): Indices into wave_arr for each wavelength.
    - d_lambda (float): Range threshold for grouping lines (default 10, matching genLSF).

    Returns:
    - eval_indices (numpy.ndarray): Sorted array of pixel indices covered by all groups.
    """
    # Sort by wavelength (same as genLSF)
    sorted_idx = np.argsort(wave_list)
    wave_list_s = wave_list[sorted_idx]
    wli_s = wave_list_indices[sorted_idx]

    # Group wavelengths if within 2.5*d_lambda (same as genLSF)
    groups = []
    current_group = [0]
    for i in range(1, len(wave_list_s)):
        if (wave_list_s[i] - wave_list_s[i - 1]) <= (2.5 * d_lambda) * u.Angstrom:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    # Resolution and pix_width (same as genLSF)
    resolution = np.median(wave_arr[1:] - wave_arr[:-1])
    pix_width = int(d_lambda / resolution)
    n_total = len(wave_arr)

    # Collect all indices covered by any group region (with padding)
    covered = np.zeros(n_total, dtype=bool)
    for grp in groups:
        min_idx = int(np.min(wli_s[grp]))
        max_idx = int(np.max(wli_s[grp]))
        pad_region = 2 * pix_width
        region_start = max(min_idx - pad_region, 0)
        region_end = min(max_idx + pad_region + 1, n_total)
        covered[region_start:region_end] = True

    return np.where(covered)[0]


def genLSF(flux, wave_list=None, wave_list_indices=None, d_lambda=10, wave_arr=None, lifetimePosition=4):
    """
    Convolve a given spectrum with COS/FUSE LSFs at specified wavelengths, grouping
    close lines so they are convolved as one large section (to avoid double convolution).

    If two consecutive wavelengths in wave_list are within 2.5*d_lambda, they will be
    grouped together and convolved in a single pass.

    Parameters:
    - flux (numpy.ndarray): The input spectrum flux.
    - wave_list (numpy.ndarray, optional): Array of wavelengths for convolution.
    - wave_list_indices (numpy.ndarray, optional): Array of indices corresponding to wave_list.
    - d_lambda (float, optional): Range threshold for grouping lines (default is 10).
    - wave_arr (numpy.ndarray, optional): The wavelength array for the spectrum.
    - lifetimePosition (int, optional): COS lifetime position (default is 4).

    Returns:
    - COSLineSpreadFunction1D (function): A convolution function for the given spectrum.
    """
    if wave_arr is None:
        raise ValueError('wave_arr is required.')

    # Sort lines by wavelength
    sorted_indices = np.argsort(wave_list)
    wave_list = wave_list[sorted_indices]
    wave_list_indices = wave_list_indices[sorted_indices]

    # print(wave_list)

    # Group wavelengths if they are within 2.5*d_lambda
    groups = []
    current_group = [0]
    for i in range(1, len(wave_list)):
        # print(i)
        if (wave_list[i] - wave_list[i-1]) <= (2.5 * d_lambda) * u.Angstrom:
            # print('first')
            current_group.append(i)
        else:
            # print('second')
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)
    # print(groups)
    # print(groups)
    def LineSpreadFunction1D(in_flux):
        new_flux = np.array(in_flux, copy=True)
        resolution = np.median(wave_arr[1:] - wave_arr[:-1])
        pix_width = int(d_lambda / resolution)
        group_centers = np.zeros(len(groups), dtype=int)
        group_wave_centers = np.zeros(len(groups), dtype=float)
        group_width =  np.zeros(len(groups), dtype=int)
        for i, grp in enumerate(groups):
            # Find min/max index for the group
            min_idx = np.min(wave_list_indices[grp])
            max_idx = np.max(wave_list_indices[grp])
            # Pad a little region
            pad_region = 2 * pix_width
            region_start = max(min_idx - pad_region, 0)
            region_end = min(max_idx + pad_region + 1, len(new_flux))
            # print('Region start: {}, end: {}'.format(region_start, region_end))
            # wavelengths for the region start and end
            region_wave_start = wave_arr[region_start]
            region_wave_end = wave_arr[region_end - 1]
            # print('Region wave start: {}, end: {}'.format(region_wave_start, region_wave_end))
            # Build combined kernel for this group
            # Length ~ region_end - region_start, centered in that span
            region_length = region_end - region_start
            group_centers[i] = (region_start + region_end) // 2
            group_wave_centers[i] = (region_wave_start + region_wave_end) / 2
            group_width[i] = region_length


            # Select FUSE or COS LSF
            if group_wave_centers[i] < 1100:
                this_lsf = getFUSElsf(group_wave_centers[i])
            elif group_wave_centers[i] < 2000:
                this_lsf = getCOSlsf(group_wave_centers[i], lifetimePosition=lifetimePosition)
            else:
                this_lsf = getSTISlsf(group_wave_centers[i])

            kernel = this_lsf['kernel']

            #print the wavelength ranges that are being convolved
            # print('Convolving region: {} - {} Angstroms with LSF centered at {} Angstroms'.format(region_wave_start, region_wave_end, group_wave_centers[i]))


            conv_region = convolve_fft(new_flux[region_start:region_end],
                                    kernel, fill_value=1,
                                        normalize_kernel=True)
            new_flux[region_start:region_end] = conv_region

        return new_flux

    return LineSpreadFunction1D

        #         output = getCOSlsf(wave_list[i].to('Angstrom').value, lifetimePosition=lifetimePosition)
        #         kernel = output['kernel']

        #     # Calculate the distances to the next and previous indices
        #     if i > 0:
        #         prev_distance = (index - wave_list_indices[i-1]) // 2
        #     else:
        #         prev_distance = 2*pix_width
        #     if i < len(wave_list_indices) - 1:
        #         next_distance = (wave_list_indices[i+1] - index) // 2
        #     else:
        #         next_distance = 2*pix_width

        #     # Use the minimum of the calculated distances and 2*pix_width
        #     minimum = max(index - min(prev_distance, 2*pix_width), 0)
        #     maximum = min(index + min(next_distance, 2*pix_width), len(flux) - 1)


        #     # Find the maximum of the kernel
        #     max_value = np.max(kernel)
        #     # Find the half maximum
        #     half_max = max_value / 2.0
        #     # Find where the kernel crosses the half maximum
        #     crossings = np.where(np.diff(kernel > half_max))[0]
        #     # Calculate the FWHM
        #     fwhm = crossings[-1] - crossings[0]
        #     pad = 3*fwhm
        #     indexes_nopad = np.arange(minimum, maximum, 1)
        #     padded_min = minimum - pad
        #     padded_max = maximum + pad
        #     if padded_min < 0:
        #         padded_min = 0
        #     if padded_max > len(flux):
        #         padded_max = len(flux)
        #     indexes = np.arange(padded_min, padded_max, 1)

        #     blah = convolve_fft(flux[indexes], kernel, fill_value=1, normalize_kernel=True)
            
        #     left_pad = minimum - padded_min
        #     right_pad = padded_max - maximum

        #     new_flux[indexes_nopad] = blah[left_pad:-right_pad]

        # return new_flux



def LineSpreadFunction1D(spectrum, zs, line_lists):
    """
    Generate a Line Spread Function (LSF) model for a given spectrum and spectral lines.

    Parameters:
    - spectrum (Spectrum1D): Input spectrum.
    - zs (list): List of redshift values.
    - line_lists (list): List of spectral lines for each redshift.

    Returns:
    - lsf (custom model): Line Spread Function (LSF) Astropy model.
    """
    # Generate wave indices and wave lists based on the spectrum and redshifts
    index_list1, wave_list1 = gen_wave_indices(spectrum.spectral_axis, zs, line_lists)

    # Generate a convolution function for the LSF
    convolution_function = genLSF(spectrum.flux, wave_arr=spectrum.spectral_axis.value, wave_list=wave_list1, wave_list_indices=index_list1, d_lambda=10)

    # Create a custom model for the Line Spread Function (LSF)
    LSF = custom_model(convolution_function)

    lsf = LSF()
    return lsf

