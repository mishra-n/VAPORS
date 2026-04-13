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

def mergeSpectrums(spectrum1, spectrum2, priority='first'):
    """
    Merge two Spectrum1D objects, optionally prioritizing one in overlapping wavelength regions.

    Parameters:
    - spectrum1 (Spectrum1D)
    - spectrum2 (Spectrum1D)
    - priority (str): 'first' or 'second' to choose which spectrum takes precedence in overlaps

    Returns:
    - merged_spectrum (Spectrum1D)
    """
    w1, f1, e1, m1 = spectrum1.spectral_axis.value, spectrum1.flux.value, spectrum1.uncertainty.array, spectrum1.mask
    w2, f2, e2, m2 = spectrum2.spectral_axis.value, spectrum2.flux.value, spectrum2.uncertainty.array, spectrum2.mask
    
    overlap_min = max(w1.min(), w2.min())
    overlap_max = min(w1.max(), w2.max())
    mask_overlap1 = (w1 >= overlap_min) & (w1 <= overlap_max)
    mask_overlap2 = (w2 >= overlap_min) & (w2 <= overlap_max)
    
    if priority == 'first':
        w2, f2, e2, m2 = w2[~mask_overlap2], f2[~mask_overlap2], e2[~mask_overlap2], m2[~mask_overlap2]
    else:
        w1, f1, e1, m1 = w1[~mask_overlap1], f1[~mask_overlap1], e1[~mask_overlap1], m1[~mask_overlap1]
    
    w_merged = np.concatenate((w1, w2))
    f_merged = np.concatenate((f1, f2))
    e_merged = np.concatenate((e1, e2))
    m_merged = np.concatenate((m1, m2))
    
    sort_idx = np.argsort(w_merged)
    w_merged, f_merged, e_merged, m_merged = w_merged[sort_idx], f_merged[sort_idx], e_merged[sort_idx], m_merged[sort_idx]
    
    merged_spectrum = Spectrum1D(flux=u.Quantity(f_merged),
                                 uncertainty=StdDevUncertainty(e_merged),
                                 mask=m_merged,
                                 spectral_axis=u.Quantity(w_merged, unit=u.Angstrom))
    
    return merged_spectrum


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
