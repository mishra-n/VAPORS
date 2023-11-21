from header import *  # Importing necessary dependencies
from line_info import *  # Importing additional dependencies from other modules
from helpers import *  # Importing helper functions from a different module
from upper_limits import *

constant1 = (np.sqrt(2 * k_B / (m_p * c**2)) * c).to(u.km/u.second / u.Kelvin**(0.5)).value

def genSummaryTable(spectrum, chain, fit, upper_limits, non_mcmc_upper_limits):
    """
    Generates a summary table for a given spectrum, chain, fit, and upper limits.

    Parameters:
    spectrum : The spectrum to analyze.
    chain : The Markov Chain to use for the analysis.
    fit : The fit to use for the analysis.
    upper_limits : The upper limits to use for the analysis.
    non_mcmc_upper_limits : The non-MCMC upper limits to use for the analysis.

    Returns:
    table : An Astropy table containing the analysis results.
    """
    # Define the column names and data types
    colnames = ['ion', 'tempname', 'wavelength', 'median_N', 'N_left_error', 'N_right_error', 'upper_limit', 'non_MCMC_upper_limit', '2-sigma_UL_N', 'v', 'v_righterr' , 'v_lefterr', 'b_other', 'b_righterr', 'b_lefterr','T', 'T_lefterr', 'T_righterr', 'mass', 'z', 'absorber_number']
    dtype = [   'U10', 'U10', 'float64', 'float64',  'float64',      'float64',       'bool',        'bool',                'float64',  'float64', 'float64',    'float64',   'float64', 'float64',    'float64', 'float64','float64','float64','float64','float64', 'int']

    # Create an empty Astropy table
    table = Table(names=colnames, dtype=dtype)
    count=0
    for i, param in enumerate(fit.cov_matrix.param_names):
        if 'N_' not in param:
            continue
        model_num = int(param.split('_')[1])
        
        if np.any(np.array(fit.cov_matrix.param_names) == ('b_other_' + str(int(param.split('_')[1])))):
            count=i

        name = fit[model_num].name

        temp_name, redshift = name.split('__z=')
        z = np.round(float(redshift), decimals=4)
        # if str(z) not in name:
        #     print('does this ever happen')
        #     continue


        ion_name = SEARCH_LINES[SEARCH_LINES['tempname'] == temp_name]['name'][0]
        wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == temp_name]['wave'][0]

        v_val = fit[model_num].v.value
        b_other_val = fit[model_num].b_other.value# * u.km / u.s
        T_val = fit[model_num].T.value# * u.K
        mass_value = fit[model_num].mass.value

        param_posterior = chain[:, i]

        N_median = np.median(param_posterior)
        N_left_error = N_median - np.percentile(param_posterior, 16)
        N_right_error = np.percentile(param_posterior, 84) - N_median

        b_posterior = chain[:, count+2]
        b_median = np.median(b_posterior)
        b_left_error = b_median - np.percentile(b_posterior, 16)
        b_right_error = np.percentile(b_posterior, 84) - b_median

        v_posterior = chain[:, count+3]
        v_median = np.median(v_posterior)
        v_left_error = v_median - np.percentile(v_posterior, 16)
        v_right_error = np.percentile(v_posterior, 84) - v_median

        T_posterior = chain[:, count+1]
        T_median = np.median(T_posterior)
        T_left_error = T_median - np.percentile(T_posterior, 16)
        T_right_error = np.percentile(T_posterior, 84) - T_median

        new_row = (ion_name, temp_name, wavelength, N_median, N_left_error, N_right_error, False, False, np.nan, v_median,v_left_error,v_right_error, b_median,b_left_error,b_right_error, T_median,T_left_error,T_right_error, mass_value, z, 0)
        table.add_row(new_row)

        absorber_number = 1
        unique_values = {}
            
    for i, row in enumerate(table):
        v = row['v']
        
        if v not in unique_values:
            unique_values[v] = absorber_number
            absorber_number += 1
        
        table['absorber_number'][i] = unique_values[v]
    
    for i, absorber_num in enumerate(np.unique(table['absorber_number'])):
        for j, line_name in enumerate(upper_limits):
            z = table[table['absorber_number'] == absorber_num]['z'].mean()

            b_median = table[table['absorber_number'] == absorber_num]['b_other'].mean()
            b_left_error = table[table['absorber_number'] == absorber_num]['b_righterr'].mean()
            b_right_error = table[table['absorber_number'] == absorber_num]['b_lefterr'].mean()

            T_median = table[table['absorber_number'] == absorber_num]['T'].mean()
            T_left_error = table[table['absorber_number'] == absorber_num]['T_lefterr'].mean()
            T_right_error = table[table['absorber_number'] == absorber_num]['T_righterr'].mean()

            v_median = table[table['absorber_number'] == absorber_num]['v'].mean()
            v_left_error = table[table['absorber_number'] == absorber_num]['v_lefterr'].mean()
            v_right_error = table[table['absorber_number'] == absorber_num]['v_righterr'].mean()

            mass = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['mass'][0]
            b_therm = constant1 * np.sqrt(T_median) / np.sqrt(mass)
            b_tot = np.sqrt(b_median**2 + b_therm**2)
            line = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]
            Wr, WrErr = calc_Wr(spectrum, line, z, v=table['v'], dv=[-b_tot*u.km/u.second, b_tot*u.km/u.second], mask=1.0)
            AOD, AODErr = calc_AOD(spectrum, line, z, v=table['v'], dv=[-b_tot*u.km/u.second, b_tot*u.km/u.second], mask=1.0)
            N, NErr = AOD_to_N(AOD, AODErr, line)
            name = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['name'][0]
            wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['wave'][0]

            if (wavelength * (1+z) > 1795) or (wavelength * (1+z) < 1180):
                continue

            if ((table['ion'] == name) * (table['absorber_number'] == absorber_num)).sum() == 0:
                new_row = (name, line_name, wavelength, np.nan, np.nan, np.nan, True, False, np.log10(2*NErr * u.cm**2), v_median,v_left_error,v_right_error, b_median,b_left_error,b_right_error, T_median,T_left_error,T_right_error, mass, z, absorber_num)
                table.add_row(new_row)

            else:
                index = np.where((table['ion'] == name) * (table['absorber_number'] == absorber_num))
                compare = table[table['ion'] == name]['2-sigma_UL_N'].mean()
                if np.log10(2*NErr * u.cm**2) < compare:
                    new_row = (name, line_name, wavelength, np.nan, np.nan, np.nan, True, False, np.log10(2*NErr * u.cm**2), v_median,v_left_error,v_right_error, b_median,b_left_error,b_right_error, T_median,T_left_error,T_right_error, mass, z, absorber_num)
                    table[index] = (new_row)

        for j, line_name in enumerate(non_mcmc_upper_limits):
            b_median = table[table['absorber_number'] == absorber_num]['b_other'].mean()
            b_left_error = table[table['absorber_number'] == absorber_num]['b_righterr'].mean()
            b_right_error = table[table['absorber_number'] == absorber_num]['b_lefterr'].mean()

            T_median = table[table['absorber_number'] == absorber_num]['T'].mean()
            T_left_error = table[table['absorber_number'] == absorber_num]['T_lefterr'].mean()
            T_right_error = table[table['absorber_number'] == absorber_num]['T_lefterr'].mean()

            v_median = table[table['absorber_number'] == absorber_num]['v'].mean()
            v_left_error = table[table['absorber_number'] == absorber_num]['v_lefterr'].mean()
            v_right_error = table[table['absorber_number'] == absorber_num]['v_righterr'].mean()

            mass = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['mass'][0]
            b_therm = constant1 * np.sqrt(T_median) / np.sqrt(mass)
            b_tot = np.sqrt(b_median**2 + b_therm**2)
            line = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]
            Wr, WrErr = calc_Wr(spectrum, line, z, v=table['v'], dv=[-b_tot*u.km/u.second, b_tot*u.km/u.second], mask=1.0)
            AOD, AODErr = calc_AOD(spectrum, line, z, v=table['v'], dv=[-b_tot*u.km/u.second, b_tot*u.km/u.second], mask=1.0)
            N, NErr = AOD_to_N(AOD, AODErr, line)

            name = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['name'][0]
            wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['wave'][0]

            if ((table['ion'] == name) * (table['absorber_number'] == absorber_num)).sum() == 0:
                new_row = (name, line_name, wavelength, np.nan, np.nan, np.nan, False, True, np.log10(2*NErr * u.cm**2), v_median,v_left_error,v_right_error, b_median,b_left_error,b_right_error, T_median,T_left_error,T_right_error, mass, z, absorber_num)
                table.add_row(new_row)

            else:
                index = np.where((table['ion'] == name) * (table['absorber_number'] == absorber_num))
                compare = table[table['ion'] == name]['2-sigma_UL_N'].mean()
                if np.log10(2*NErr * u.cm**2) < compare:
                    new_row = (name, line_name, wavelength, np.nan, np.nan, np.nan, False, True, np.log10(2*NErr * u.cm**2), v_median,v_left_error,v_right_error, b_median,b_left_error,b_right_error, T_median,T_left_error,T_right_error, mass, z, absorber_num)
                    table[index] = (new_row)

    table.sort('absorber_number')

    return table