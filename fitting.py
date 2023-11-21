from header import *
from line_info import *
from helpers import *
def fit(spectrum, model, vranges, lines, zs, maxiter=1000):
    """
    Fit a custom model to a spectrum, considering specific spectral lines and redshifts.

    Parameters:
    - spectrum (Spectrum1D): Spectrum to be fitted.
    - model (custom model): Custom model to fit to the spectrum.
    - lines (list of lists): List of lists containing spectral lines for each redshift.
    - zs (list): List of redshift values corresponding to the lines.

    Returns:
    - ve_fit_output (FittedModel): Fitted model.
    - ve_fit (TRFLSQFitter): Fitting algorithm used.
    - mask (boolean array): Mask indicating the spectral regions used in the fit.
    """
    # Initialize the fitting algorithm with uncertainty calculation
    ve_fit = fitting.TRFLSQFitter(calc_uncertainties=True)
    
    # Initialize the mask to None
    mask = None
    
    # Loop through redshifts and corresponding spectral lines
    for i, z in enumerate(zs):
        this_z_lines = lines[i]
        this_z_vranges = vranges[i]
        # Iterate over each line within the current redshift
        for line in this_z_lines:
            # Calculate the transition wavelength for the current line at the given redshift
            transition = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['wave'][0] #* u.Angstrom

            velocity_spectrum =  to_velocity(spectrum.spectral_axis, transition, z).to('km/s')
            vmin = this_z_vranges[line][0]
            vmax = this_z_vranges[line][1]
            # Create a mask for the spectral region around the current transition wavelength
            mask_temp = np.logical_and(
                velocity_spectrum > vmin,
                velocity_spectrum < vmax
            )
            
            # Combine the masks for different lines
            if mask is None:
                mask = mask_temp
            else:
                mask = np.logical_or(mask, mask_temp)

    
    # Calculate weightings based on the uncertainty, setting weights to 0 outside the mask
    weightings = np.nan_to_num(1 / (spectrum.uncertainty.array), nan=0, posinf=0, neginf=0)
    weightings[~mask] = 0
    
    # Fit the model to the spectrum with weightings and a maximum of 1000 iterations
    ve_fit_output = ve_fit(model, spectrum.spectral_axis.value, spectrum.flux.value, weights=weightings, maxiter=maxiter)
    
    return ve_fit_output, ve_fit, mask
