from header import *  # Importing necessary dependencies
from line_info import *  # Importing additional dependencies from other modules
from helpers import *  # Importing helper functions from a different module

deltav = 60


def calc_Wr(spectrum, line, z, v=0*u.km/u.second, dv=[-deltav*u.km/u.second, deltav*u.km/u.second], mask=1.0):
    line = line['wave'][0]
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

def calc_AOD(spectrum, line, z, v=0*u.km/u.second, dv=[-deltav*u.km/u.second, deltav*u.km/u.second], mask=1.0):

    
    f = line['f'][0]
    line = line['wave'][0]
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

    AOD = np.trapz(tau, wave_cut)
    AODErr = np.sqrt(np.trapz(tau_err**2, wave_cut) * dW)
    return AOD, AODErr

def AOD_to_N(AOD, AODErr, line):
    N = m_e * c**2 / np.pi / e.gauss**2 / line['f'][0] / (line['wave'][0] * u.Angstrom)**2 * AOD
    NErr = m_e * c**2 / np.pi / e.gauss**2 / line['f'][0] / (line['wave'][0] * u.Angstrom)**2 * AODErr

    return N.to(u.cm**(-2)), NErr.to(u.cm**(-2))



