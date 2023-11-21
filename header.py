import numpy as np
from numpy import pi
#import redshifting as redshift
import os.path
import glob

import astropy
from astropy.io import fits
from astropy.io.misc import fnpickle, fnunpickle

import astropy.units as u
from astropy.modeling import models, fitting
from astropy.modeling.fitting import fitter_to_model_params, model_to_fit_params
from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling import custom_model

from astropy.nddata import NDData, StdDevUncertainty
from astropy.visualization import quantity_support
from astropy.convolution import Gaussian1DKernel, convolve, convolve_models, convolve_fft
from astropy import uncertainty as unc
from astropy.stats import gaussian_fwhm_to_sigma

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.interpolate as interp
from scipy.interpolate import RegularGridInterpolator
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.signal import convolve


import matplotlib as mpl
from astropy.table import Table, QTable, vstack
import corner
import mpdaf
from astropy.constants import c, m_e, e, k_B, m_p
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines
from specutils.fitting import fit_generic_continuum
import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling.models import Spline1D
from astropy.modeling import models, Parameter, custom_model

from astropy.modeling.fitting import (SplineInterpolateFitter,
                                      SplineSmoothingFitter,
                                      SplineExactKnotsFitter)

from astropy.cosmology import FlatLambdaCDM
from mendeleev import element
import os
from roman import toRoman, fromRoman
import astropy.io.ascii as ascii
from scipy.optimize import minimize
from scipy.interpolate import interpn

