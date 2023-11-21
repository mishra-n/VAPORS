from header import *  # Importing necessary dependencies
from line_info import *  # Importing line_info module
from helpers import *  # Importing helper functions

def createSpectrum_COS(COS_spectrum):
    """
    Given a MUSE Spectrum, return a specutils Spectrum1D object in the rest frame.

    Parameters:
    - COS_spectrum (dict): A dictionary containing MUSE spectrum information.

    Returns:
    - spectrum2 (Spectrum1D): A specutils Spectrum1D object in the rest frame.
    """
    # Calculate the normalized flux and error by dividing by the continuum
    COS_spectrum['normalized_flux'] = COS_spectrum['flux'] / COS_spectrum['continuum']
    COS_spectrum['normalized_error'] = COS_spectrum['error'] / COS_spectrum['continuum']
    
    # Create a Spectrum1D object with the normalized data
    spectrum2 = Spectrum1D(flux=u.Quantity(COS_spectrum['normalized_flux']),
                           uncertainty=StdDevUncertainty(COS_spectrum['normalized_error']),
                           mask=COS_spectrum['mask'],
                           spectral_axis=u.Quantity(COS_spectrum['wave'], unit=u.Angstrom))
    
    return spectrum2

def convertSpectrum_COS(COS_spectrum):
    """
    Convert a COS spectrum to a specutils Spectrum1D object.

    Parameters:
    - COS_spectrum (Spectrum1D): A COS spectrum object.

    Returns:
    - spectrum2 (Spectrum1D): A specutils Spectrum1D object.
    """
    # Create a Spectrum1D object using the existing COS_spectrum data
    spectrum2 = Spectrum1D(flux=COS_spectrum.flux,
                           uncertainty=COS_spectrum.uncertainty,
                           mask=COS_spectrum.mask,
                           spectral_axis=COS_spectrum.frequency)
    
    return spectrum2
