from header import *
from line_info import *
from spectral import *
from model_builder import *

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['figure.facecolor'] = 'white'
mpl.use('PDF')
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=10)
plt.rc('axes', linewidth=1.5)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15, direction='in')
plt.rc('ytick', labelsize=15, direction='in')
plt.rc('xtick', top=True)
plt.rc('ytick', right=True)
plt.rc('xtick.minor', visible=True)
plt.rc('ytick.minor', visible=True)
plt.rc('xtick.major',size=10, pad=4)
plt.rc('xtick.minor',size=5, pad=4)
plt.rc('ytick.major',size=10)
plt.rc('ytick.minor',size=5)
plt.rc('legend', fontsize=15)


def plot_absorber(fig, ax, spectrum, line_list, z, vrange=300, n=1, gray_out=None):
    """
    Plot the absorber features in the given spectrum.

    Parameters:
    - fig (matplotlib.figure.Figure): The matplotlib figure for the plot.
    - ax (array of matplotlib.axes.Axes): An array of subplots for individual absorber plots.
    - spectrum (Spectrum1D): The input spectrum.
    - line_list (list): List of spectral lines to plot.
    - z (float): The redshift value.

    Returns:
    - None
    """
    nrows = len(line_list)

    for i, line in enumerate(line_list):
        # Get the wavelength of the spectral line
        line_wave = SEARCH_LINES[SEARCH_LINES['tempname']==line]['wave'][0]
        ion_name = SEARCH_LINES[SEARCH_LINES['tempname']==line]['name'][0]

        # Convert the full velocity axis to km/s
        full_velocity_axis = to_velocity(spectrum.spectral_axis, line_wave, z).to('km/s')

        # Averaging for better visualization
        length = spectrum.shape[0]
        remainder = length % n
        avg_flux = np.average(spectrum.flux[remainder::].reshape(-1, n), axis=1)
        avg_std = np.sqrt(np.sum((spectrum.uncertainty.array[remainder::].reshape(-1, n))**2, axis=1)/n**2)
        #avg_std = np.sqrt(avg_variance)
        avg_velocity_axis = np.average(full_velocity_axis[remainder::].reshape(-1, n), axis=1)

        # Plotting
        if len(line_list) > 1:
            ax[i].text(0.98, 0.12, str(ion_name) + ' [' + str(np.round(line_wave, decimals=2)) + ' $\mathrm{\AA}$]', horizontalalignment='right', verticalalignment='bottom', transform=ax[i].transAxes, fontsize='x-large')
            ax[i].step(avg_velocity_axis.value, avg_flux, alpha=0.5, color='blue', where='mid')
            ax[i].fill_between(avg_velocity_axis.value,
                                (avg_flux-avg_std).value,
                                (avg_flux+avg_std).value,
                                color='blue', alpha=0.15, step='mid') 
            
            ax[i].vlines(0, 0, 1.5, color='black', linestyles='-.', alpha=0.5)
            ax[i].hlines(1, -vrange, vrange, color='slategrey', linestyles='dashed', alpha=0.9)
            # Grey out specific velocity range
            if (gray_out is not None) and (gray_out[i] is not None):
                ax[i].fill_betweenx([0,1.5], [gray_out[i][0]]*2, [gray_out[i][1]]*2, color='lightgrey', alpha=0.5)

        else:
            ax.text(0.98, 0.12, str(ion_name) + ' [' + str(np.round(line_wave, decimals=2)) + ' $\mathrm{\AA}$]', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize='x-large')
            ax.step(avg_velocity_axis.value, avg_flux, alpha=0.5, color='blue', where='mid')
            ax.fill_between(avg_velocity_axis.value,
                            (avg_flux-avg_std).value,
                            (avg_flux+avg_std).value,
                            color='blue', alpha=0.15, step='mid')
            
            ax.vlines(0, 0, 1.5, color='black', linestyles='-.', alpha=0.5)
            ax.hlines(1, -vrange, vrange, color='slategrey', linestyles='dashed', alpha=0.9)
            if (gray_out is not None):
                ax[i].fill_betweenx([0,1.5], [gray_out[i][0]]*2, [gray_out[i][1]]*2, color='lightgrey', alpha=0.5)

        plt.xlim(-vrange,vrange)
        plt.ylim(0, 1.5)
        fig.add_subplot(111, frameon=False)
        plt.grid(False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, length=0, width=0, color='none', grid_alpha=0, which='both')

        plt.ylabel('Normalized Flux')
        plt.xlabel('Velocity [km/s]')

def plotter(spectrum, line_list, fit, z, mask=None, vrange=300, n=1, save=True):
    """
    Plot the absorber features and the fitted model.

    Parameters:
    - spectrum (Spectrum1D): The input spectrum.
    - line_list (list): List of spectral lines to plot.
    - fit (callable): The fitted model.
    - z (float): The redshift value.

    Returns:
    - None
    """
    nrows = len(line_list)
    fig, ax = plt.subplots(figsize=(10, int(2 * nrows)), nrows=nrows, sharex=True)

    # Plot the absorber features
    plot_absorber(fig, ax, spectrum, line_list, z, vrange=vrange, n=n)
    
    # Plot the fitted model
    plot_model(fig, ax, spectrum, line_list, fit, z, mask=mask, vrange=vrange)

    if save==True:
        plt.savefig('absorber_plot.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_model(fig, ax, spectrum, line_list, fit, z, mask=None, vrange=300):
    """
    Plot the fitted model for the absorber features in the given spectrum.

    Parameters:
    - fig (matplotlib.figure.Figure): The matplotlib figure for the plot.
    - ax (array of matplotlib.axes.Axes): An array of subplots for individual absorber plots.
    - spectrum (Spectrum1D): The input spectrum.
    - line_list (list): List of spectral lines to plot.
    - fit (callable): The fitted model.
    - z (float): The redshift value.

    Returns:
    - None
    """
    if len(line_list) > 1:
        for i, line in enumerate(line_list):
            # Get the wavelength of the spectral line
            line_wave = SEARCH_LINES[SEARCH_LINES['tempname']==line]['wave'][0]
            full_velocity_axis = to_velocity(spectrum.spectral_axis, line_wave, z).to('km/s')
            ax[i].plot(full_velocity_axis, fit(spectrum.spectral_axis.value), color='Green')
            index = 0

            # Find the index of the sub-model with None as the name
            for j, sub in enumerate(fit):
                if sub.name is None:
                    index = j
                    break

            for voigt in fit:
                if (voigt.name == 'exponential') or (voigt.name is None):
                    continue

                voigt_line_name, voigt_line_z = voigt.name.split('__z=')
                voigt_line_wave = SEARCH_LINES[SEARCH_LINES['tempname']==voigt_line_name]['wave'][0]
                voigt_line_z = float(voigt_line_z)
                voigt_line_observed_wave = voigt_line_wave * (1 + voigt_line_z)
                voigt_line_observed_velocity = to_velocity(voigt_line_observed_wave * u.Angstrom, line_wave, z).to('km/s') 
                voigt_line_observed_velocity = voigt_line_observed_velocity.value   

                if (voigt_line_observed_velocity < -vrange) or (voigt_line_observed_velocity > vrange):
                    continue            
                # if line not in voigt.name:
                #     continue
                lsf_voigt = generic_exp(voigt) | fit[index]
                if str(z) not in voigt.name:
                    ax[i].plot(full_velocity_axis, lsf_voigt(spectrum.spectral_axis.value), '--', color='red', alpha=0.5)
                else:
                    ax[i].plot(full_velocity_axis, lsf_voigt(spectrum.spectral_axis.value), '--', color='green', alpha=0.5)
                    if mask is not None:
                        ax[i].plot(full_velocity_axis, mask)
            ax[i].set_xlim(-vrange,vrange)
            ax[i].set_ylim(0, 1.5)
    else:
        for i, line in enumerate(line_list):
            # Get the wavelength of the spectral line
            line_wave = SEARCH_LINES[SEARCH_LINES['tempname']==line]['wave'][0]
            full_velocity_axis = to_velocity(spectrum.spectral_axis, line_wave, z).to('km/s')
            ax.plot(full_velocity_axis, fit(spectrum.spectral_axis.value), color='Green')
            index = 0

            # Find the index of the sub-model with None as the name
            for j, sub in enumerate(fit):
                if sub.name is None:
                    index = j
                    break

            for voigt in fit:
                if (voigt.name == 'exponential') or (voigt.name is None):
                    continue

                voigt_line_name, voigt_line_z = voigt.name.split('__z=')
                voigt_line_wave = SEARCH_LINES[SEARCH_LINES['tempname']==voigt_line_name]['wave'][0]
                voigt_line_z = float(voigt_line_z)
                voigt_line_observed_wave = voigt_line_wave * (1 + voigt_line_z)
                voigt_line_observed_velocity = to_velocity(voigt_line_observed_wave * u.Angstrom, line_wave, z).to('km/s') 
                voigt_line_observed_velocity = voigt_line_observed_velocity.value   

                if (voigt_line_observed_velocity < -vrange) or (voigt_line_observed_velocity > vrange):
                    continue   

                lsf_voigt = generic_exp(voigt) | fit[index]

                if str(z) not in voigt.name:
                    ax.plot(full_velocity_axis, lsf_voigt(spectrum.spectral_axis.value), '--', color='red', alpha=0.5)
                else:
                    ax.plot(full_velocity_axis, lsf_voigt(spectrum.spectral_axis.value), '--', color='green', alpha=0.5)
                    if mask is not None:
                        ax.plot(full_velocity_axis, mask)
            ax.set_xlim(-vrange,vrange)
            ax.set_ylim(0, 1.5)

def plotter(spectrum, line_list, fit, z, mask=None, vrange=300, n=1, save=True, title=None, name=None, gray_out=None, fig_return=False):
    """
    Plot the absorber features and the fitted model.

    Parameters:
    - spectrum (Spectrum1D): The input spectrum.
    - line_list (list): List of spectral lines to plot.
    - fit (callable): The fitted model.
    - z (float): The redshift value.

    Returns:
    - None
    """
    nrows = len(line_list)
    fig, ax = plt.subplots(figsize=(8, int(2 * nrows)), nrows=nrows, sharex=True)

    # Plot the absorber features
    plot_absorber(fig, ax, spectrum, line_list, z, vrange=vrange, n=n, gray_out=gray_out)
    
    # Plot the fitted model
    plot_model(fig, ax, spectrum, line_list, fit, z, mask=mask, vrange=vrange)

    if title is not None:
        plt.title(title, fontdict={'fontsize': 20})

    if save==True:
        if name is None:
            plt.savefig('absorber_plot.pdf', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(str(name) + '_absorber_plot.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.show()

    if fig_return==True:
        return fig, ax


def plotViolin(output_Ns, ions, subtable):
    """
    Creates a violin plot for the given data.

    Parameters:
    output_Ns : A 2D array where each row corresponds to an ion and each column corresponds to a measurement.
    ions : A list of ion names corresponding to the rows in output_Ns.
    subtable : A DataFrame containing the median, upper, and lower errors, and upper limit flags for each ion.

    The function plots a violin plot of the measurements for each ion. It also plots the median and error bars from the subtable, and marks any upper limits with a downward-pointing triangle.

    The plot is displayed using plt.show().

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(15,5))
    ax.violinplot(output_Ns.swapaxes(0,1), widths=0.7, showextrema=True, showmedians=True)

    ax.set_xticks(np.arange(1,13))
    ax.set_xticklabels(ions)

    expected = np.full(len(ions), np.nan)
    upper = np.full(len(ions), np.nan)
    lower = np.full(len(ions), np.nan)
    ul = np.full(len(ions), np.nan)
    ul_nomcmc = np.full(len(ions), np.nan)
    
    for i,ion in enumerate(ions):
        ion_table = subtable[subtable['ion'] == ion]
        if len(ion_table) == 0:
            continue
        if ion_table['upper_limit'] == False and ion_table['non_MCMC_upper_limit'] == False:
            expected[i] = ion_table['median_N']
            upper[i] = ion_table['N_right_error']
            lower[i] = ion_table['N_left_error']
        if ion_table['upper_limit'] == True:
            ul[i] = ion_table['2-sigma_UL_N']
        if ion_table['non_MCMC_upper_limit'] == True:
            ul_nomcmc[i] = ion_table['2-sigma_UL_N']


    plt.errorbar(np.arange(1,13), expected, color='green', yerr=(lower, upper), fmt='o')
    plt.scatter(np.arange(1,13), ul, color='red', marker='v')
    plt.scatter(np.arange(1,13), ul_nomcmc, edgecolor='red', color= 'white', marker='v')

    plt.ylabel('logN')
    plt.ylim(11,15)
    plt.show()
