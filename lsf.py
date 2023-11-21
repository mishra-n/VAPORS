from header import *  # Importing necessary dependencies
from line_info import *  # Importing additional dependencies from other modules
from helpers import *  # Importing helper functions from a different module

def getCOSlsf(wave, lifetimePosition=4, vblue=False):
    """
    Get the COS Line Spread Function (LSF) for a given wavelength and lifetime position.

    Parameters:
    - wave (float): The wavelength for which to retrieve the LSF.
    - lifetimePosition (int, optional): The COS lifetime position (default is 4).
    - vblue (bool, optional): Flag indicating if it's a very blue G130M wavelength (default is False).

    Returns:
    - lsf (numpy.ndarray): The Line Spread Function as a numpy array with 'pix' and 'kernel' columns.
    """
    if vblue == True:
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
            print(lifetimePosition)
            filename = '{}{}_lp1.npy'.format(directory, grating)
            lsfTable = np.load(filename)

            # Find the LSF column closest to the desired wavelength
            enumeratedWaves_string = lsfTable.dtype.names[1:]
            enumeratedWaves = np.array(enumeratedWaves_string).astype(float)
            waveDistance = np.abs(enumeratedWaves - wave)
            index = np.argmin(waveDistance) - 1
            lsfColumn = enumeratedWaves_string[index]
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

def gen_wave_indices(wave_arr, z_list, line_lists, d_lambda=5):
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
    
    for z, line_list in zip(z_list, line_lists):
        for line in line_list:
            wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['wave'][0] * u.Angstrom

            wave_list.append(wavelength.to('Angstrom').value * (1 + z))
            index_list.append(find_nearest_index(wave_arr.to('Angstrom').value, wavelength.to('Angstrom').value * (1 + z)))

    wave_list = np.array(wave_list) * u.Angstrom
    index_list = np.array(index_list)
    wave_list, unremoved_indices = remove_close_values(wave_list, d_lambda * u.Angstrom)
    index_list = index_list[unremoved_indices]

    return index_list, wave_list

def genLSF(flux, wave_list=None, wave_list_indices=None, d_lambda=5, wave_arr=None):
    """
    Convolve a given spectrum with COS Line Spread Functions (LSFs) at specified wavelengths.

    Parameters:
    - flux (numpy.ndarray): The input spectrum flux.
    - wave_list (numpy.ndarray, optional): Array of wavelengths for convolution.
    - wave_list_indices (numpy.ndarray, optional): Array of indices corresponding to wave_list.
    - d_lambda (float, optional): Minimum allowed wavelength difference for convolution (default is 5 Angstroms).
    - wave_arr (numpy.ndarray, optional): The wavelength array for the spectrum.

    Returns:
    - convolution_function (function): A convolution function for the given spectrum.
    """
    def convolution_function(flux):
        if wave_arr is None:
            raise ValueError('You need to give the Spectrum1D object for the fitting')

        resolution = np.median(wave_arr[1:] - wave_arr[:-1])
        pix_width = int(d_lambda / resolution)
        new_flux = flux
        for i, index in enumerate(wave_list_indices):
            output = getCOSlsf(wave_list[i].to('Angstrom').value)
            kernel = output['kernel']

            # Calculate the distances to the next and previous indices
            if i > 0:
                prev_distance = (index - wave_list_indices[i-1]) // 2
            else:
                prev_distance = 2*pix_width
            if i < len(wave_list_indices) - 1:
                next_distance = (wave_list_indices[i+1] - index) // 2
            else:
                next_distance = 2*pix_width

            # Use the minimum of the calculated distances and 2*pix_width
            minimum = max(index - min(prev_distance, 2*pix_width), 0)
            maximum = min(index + min(next_distance, 2*pix_width), len(flux) - 1)


            # Find the maximum of the kernel
            max_value = np.max(kernel)
            # Find the half maximum
            half_max = max_value / 2.0
            # Find where the kernel crosses the half maximum
            crossings = np.where(np.diff(kernel > half_max))[0]
            # Calculate the FWHM
            fwhm = crossings[-1] - crossings[0]
            pad = 3*fwhm
            indexes_nopad = np.arange(minimum, maximum, 1)
            padded_min = minimum - pad
            padded_max = maximum + pad
            if padded_min < 0:
                padded_min = 0
            if padded_max > len(flux):
                padded_max = len(flux)
            indexes = np.arange(padded_min, padded_max, 1)

            blah = convolve_fft(flux[indexes], kernel, fill_value=1, normalize_kernel=True)
            
            left_pad = minimum - padded_min
            right_pad = padded_max - maximum

            new_flux[indexes_nopad] = blah[left_pad:-right_pad]

            #HELLO THE SOLUTION IS THAT WE CANNNOT OVERWRITE FLUXES EVERY LOOP, DO IT ONCE AT THE END
        return new_flux

    return convolution_function


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

