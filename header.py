# import redshifting as redshift

import glob
import os
import os.path
import re
import time
import multiprocessing

import numpy as np
from numpy import pi

import astropy
import astropy.units as u
from astropy.io import fits
from astropy.io.misc import fnpickle, fnunpickle
import astropy.io.ascii as ascii
from astropy.table import Table, QTable, vstack
from astropy.modeling import models, fitting, custom_model, Parameter
from astropy.modeling.models import PowerLaw1D, Spline1D
from astropy.modeling.fitting import (
    fitter_to_model_params,
    model_to_fit_params,
    SplineInterpolateFitter,
    SplineSmoothingFitter,
    SplineExactKnotsFitter
)
from astropy.nddata import NDData, StdDevUncertainty
from astropy.visualization import quantity_support
from astropy.convolution import Gaussian1DKernel, convolve, convolve_models, convolve_fft
from astropy import uncertainty as unc
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.constants import c, m_e, e, k_B, m_p, G
from astropy.cosmology import FlatLambdaCDM

import scipy.interpolate as interp
from scipy.interpolate import RegularGridInterpolator, interpn
import scipy.stats as stats
from scipy.optimize import curve_fit, minimize
from scipy.signal import convolve
from scipy.special import wofz

import matplotlib as mpl
import matplotlib.pyplot as plt

import corner
import mpdaf
import arviz as az
from mendeleev import element
from roman import toRoman, fromRoman
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines, fit_generic_continuum
