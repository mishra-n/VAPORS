from header import *  # Importing necessary dependencies
from line_info import *  # Importing additional dependencies from other modules
from helpers import *  # Importing helper functions from a different module
from upper_limits import *

constant1 = (np.sqrt(2 * k_B / (m_p * c**2)) * c).to(u.km/u.second / u.Kelvin**(0.5)).value
multiple = (1e12 / u.cm**2).value / (1e12 / u.cm**2).to(1 / u.micrometer**2).value


# def genSummaryTable(spectrum, chain, fit, upper_limits, non_mcmc_upper_limits):
#     """
#     Generates a summary table for a given spectrum, chain, fit, and upper limits.

#     Parameters:
#     spectrum : The spectrum to analyze.
#     chain : The Markov Chain to use for the analysis.
#     fit : The fit to use for the analysis.
#     upper_limits : The upper limits to use for the analysis.
#     non_mcmc_upper_limits : The non-MCMC upper limits to use for the analysis.

#     Returns:
#     table : An Astropy table containing the analysis results.
#     """
#     # Define the column names and data types
#     colnames = ['ion', 'tempname', 'wavelength', 'median_N', 'N_left_error', 'N_right_error', 'upper_limit', 'non_MCMC_upper_limit', '2-sigma_UL_N', 'v', 'v_righterr' , 'v_lefterr', 'b_other', 'b_righterr', 'b_lefterr','T', 'T_lefterr', 'T_righterr', 'mass', 'z', 'absorber_number']
#     dtype = [   'U10', 'U10', 'float64', 'float64',  'float64',      'float64',       'bool',        'bool',                'float64',  'float64', 'float64',    'float64',   'float64', 'float64',    'float64', 'float64','float64','float64','float64','float64', 'int']

#     # Create an empty Astropy table
#     table = Table(names=colnames, dtype=dtype)
#     count=0
#     for i, param in enumerate(fit.cov_matrix.param_names):
#         if 'N_' not in param:
#             continue
#         model_num = int(param.split('_')[1])
        
#         if np.any(np.array(fit.cov_matrix.param_names) == ('b_other_' + str(int(param.split('_')[1])))):
#             count=i

#         name = fit[model_num].name

#         temp_name, redshift = name.split('__z=')
#         z = np.round(float(redshift), decimals=4)

#         ion_name = SEARCH_LINES[SEARCH_LINES['tempname'] == temp_name]['name'][0]
#         wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == temp_name]['wave'][0]

#         v_val = fit[model_num].v.value
#         b_other_val = fit[model_num].b_other.value# * u.km / u.s
#         T_val = fit[model_num].T.value# * u.K
#         mass_value = fit[model_num].mass.value

#         param_posterior = chain[:, i]

#         N_median = np.median(param_posterior)
#         N_left_error = N_median - np.percentile(param_posterior, 16)
#         N_right_error = np.percentile(param_posterior, 84) - N_median

#         b_posterior = chain[:, count+2]
#         b_median = np.median(b_posterior)
#         b_left_error = b_median - np.percentile(b_posterior, 16)
#         b_right_error = np.percentile(b_posterior, 84) - b_median

#         v_posterior = chain[:, count+3]
#         v_median = np.median(v_posterior)
#         v_left_error = v_median - np.percentile(v_posterior, 16)
#         v_right_error = np.percentile(v_posterior, 84) - v_median

#         T_posterior = chain[:, count+1]
#         T_median = np.median(T_posterior)
#         T_left_error = T_median - np.percentile(T_posterior, 16)
#         T_right_error = np.percentile(T_posterior, 84) - T_median

#         new_row = (ion_name, temp_name, wavelength, N_median, N_left_error, N_right_error, False, False, np.nan, v_median,v_left_error,v_right_error, b_median,b_left_error,b_right_error, T_median,T_left_error,T_right_error, mass_value, z, 0)
#         table.add_row(new_row)

#         absorber_number = 1
#         unique_values = {}
            
#     for i, row in enumerate(table):
#         v = row['v']
        
#         if v not in unique_values:
#             unique_values[v] = absorber_number
#             absorber_number += 1
        
#         table['absorber_number'][i] = unique_values[v]
    
#     for i, absorber_num in enumerate(np.unique(table['absorber_number'])):
#         for j, line_name in enumerate(upper_limits):
#             z = table[table['absorber_number'] == absorber_num]['z'].mean()

#             b_median = table[table['absorber_number'] == absorber_num]['b_other'].mean()
#             b_left_error = table[table['absorber_number'] == absorber_num]['b_righterr'].mean()
#             b_right_error = table[table['absorber_number'] == absorber_num]['b_lefterr'].mean()

#             T_median = table[table['absorber_number'] == absorber_num]['T'].mean()
#             T_left_error = table[table['absorber_number'] == absorber_num]['T_lefterr'].mean()
#             T_right_error = table[table['absorber_number'] == absorber_num]['T_righterr'].mean()

#             v_median = table[table['absorber_number'] == absorber_num]['v'].mean()
#             v_left_error = table[table['absorber_number'] == absorber_num]['v_lefterr'].mean()
#             v_right_error = table[table['absorber_number'] == absorber_num]['v_righterr'].mean()

#             mass = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['mass'][0]
#             b_therm = constant1 * np.sqrt(T_median) / np.sqrt(mass)
#             b_tot = np.sqrt(b_median**2 + b_therm**2)
#             line = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]
#             Wr, WrErr = calc_Wr(spectrum, line, z, v=table['v'], dv=[-b_tot*u.km/u.second, b_tot*u.km/u.second], mask=1.0)
#             AOD, AODErr = calc_AOD(spectrum, line, z, v=table['v'], dv=[-b_tot*u.km/u.second, b_tot*u.km/u.second], mask=1.0)
#             N, NErr = AOD_to_N(AOD, AODErr, line)
#             name = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['name'][0]
#             wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['wave'][0]

#             if (wavelength * (1+z) > 1795) or (wavelength * (1+z) < 1100):
#                 continue

#             if ((table['ion'] == name) * (table['absorber_number'] == absorber_num)).sum() == 0:
#                 new_row = (name, line_name, wavelength, np.nan, np.nan, np.nan, True, False, np.log10(2*NErr * u.cm**2), v_median,v_left_error,v_right_error, b_median,b_left_error,b_right_error, T_median,T_left_error,T_right_error, mass, z, absorber_num)
#                 table.add_row(new_row)

#             else:
#                 index = np.where((table['ion'] == name) * (table['absorber_number'] == absorber_num))
#                 compare = table[table['ion'] == name]['2-sigma_UL_N'].mean()
#                 if np.log10(2*NErr * u.cm**2) < compare:
#                     new_row = (name, line_name, wavelength, np.nan, np.nan, np.nan, True, False, np.log10(2*NErr * u.cm**2), v_median,v_left_error,v_right_error, b_median,b_left_error,b_right_error, T_median,T_left_error,T_right_error, mass, z, absorber_num)
#                     table[index] = (new_row)

#         for j, line_name in enumerate(non_mcmc_upper_limits):
#             b_median = table[table['absorber_number'] == absorber_num]['b_other'].mean()
#             b_left_error = table[table['absorber_number'] == absorber_num]['b_righterr'].mean()
#             b_right_error = table[table['absorber_number'] == absorber_num]['b_lefterr'].mean()

#             T_median = table[table['absorber_number'] == absorber_num]['T'].mean()
#             T_left_error = table[table['absorber_number'] == absorber_num]['T_lefterr'].mean()
#             T_right_error = table[table['absorber_number'] == absorber_num]['T_lefterr'].mean()

#             v_median = table[table['absorber_number'] == absorber_num]['v'].mean()
#             v_left_error = table[table['absorber_number'] == absorber_num]['v_lefterr'].mean()
#             v_right_error = table[table['absorber_number'] == absorber_num]['v_righterr'].mean()

#             mass = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['mass'][0]
#             b_therm = constant1 * np.sqrt(T_median) / np.sqrt(mass)
#             b_tot = np.sqrt(b_median**2 + b_therm**2)
#             line = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]
#             Wr, WrErr = calc_Wr(spectrum, line, z, v=table['v'], dv=[-b_tot*u.km/u.second, b_tot*u.km/u.second], mask=1.0)
#             AOD, AODErr = calc_AOD(spectrum, line, z, v=table['v'], dv=[-b_tot*u.km/u.second, b_tot*u.km/u.second], mask=1.0)
#             N, NErr = AOD_to_N(AOD, AODErr, line)

#             name = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['name'][0]
#             wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['wave'][0]

#             if ((table['ion'] == name) * (table['absorber_number'] == absorber_num)).sum() == 0:
#                 new_row = (name, line_name, wavelength, np.nan, np.nan, np.nan, False, True, np.log10(2*NErr * u.cm**2), v_median,v_left_error,v_right_error, b_median,b_left_error,b_right_error, T_median,T_left_error,T_right_error, mass, z, absorber_num)
#                 table.add_row(new_row)

#             else:
#                 index = np.where((table['ion'] == name) * (table['absorber_number'] == absorber_num))
#                 compare = table[table['ion'] == name]['2-sigma_UL_N'].mean()
#                 if np.log10(2*NErr * u.cm**2) < np.log10(compare):
#                     new_row = (name, line_name, wavelength, np.nan, np.nan, np.nan, False, True, np.log10(2*NErr * u.cm**2), v_median,v_left_error,v_right_error, b_median,b_left_error,b_right_error, T_median,T_left_error,T_right_error, mass, z, absorber_num)
#                     table[index] = (new_row)

#     table.sort('absorber_number')

#     return table

# def genSummaryTableNoChain(spectrum, fit, upper_limits, z_main=None):


#     # Define the column names and data types
#     colnames = ['ion', 'tempname', 'wavelength', 'N', 'Nerr', 'upper_limit', 'v', 'verr', 'b_other', 'berr','T', 'Terr', 'b_tot', 'mass', 'z', 'absorber_number']
#     dtype = [   'U10', 'U10', 'float64','float64','float64',   'bool','float64','float64','float64','float64', 'float64','float64', 'float64','float64', 'float64', 'int']

#     # Create an empty Astropy table
#     table = Table(names=colnames, dtype=dtype)
#     count = 0
#     for i, param in enumerate(fit.cov_matrix.param_names):
#         if 'N_' not in param:
#             continue
#         model_num = int(param.split('_')[1])
#         name = fit[model_num].name

#         temp_name, redshift = name.split('__z=')
#         z = np.round(float(redshift), decimals=4)
#         # if str(z) not in name:
#         #     # print('does this ever happen')
#         #     continue

#         if np.any(np.array(fit.cov_matrix.param_names) == ('b_other_' + str(int(param.split('_')[1])))):
#             count=i

#         ion_name = SEARCH_LINES[SEARCH_LINES['tempname'] == temp_name]['name'][0]
#         wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == temp_name]['wave'][0]

#         N = fit[model_num].N.value
#         b = fit[model_num].b_other.value
#         T = fit[model_num].T.value
#         v = fit[model_num].v.value
#         Nerr = np.sqrt(fit.cov_matrix['N_' + str(model_num), 'N_' + str(model_num)])
#         mass_value = fit[model_num].mass.value

#         b_tot = np.sqrt(b**2 + constant1**2 * T / mass_value)

#         try:
#             berr = np.sqrt(fit.cov_matrix['b_other_' + str(model_num), 'b_other_' + str(model_num)])
#         except ValueError as e:
#             new_row = (ion_name, temp_name, wavelength, N*multiple, Nerr*multiple, False, v, None, b, None, T, None, b_tot, mass_value, z, 0)
#             table.add_row(new_row)
#             continue

#         berr = np.sqrt(fit.cov_matrix['b_other_' + str(model_num), 'b_other_' + str(model_num)])
#         Terr = np.sqrt(fit.cov_matrix['T_' + str(model_num), 'T_' + str(model_num)])
#         verr = np.sqrt(fit.cov_matrix['v_' + str(model_num), 'v_' + str(model_num)])
#         #b_tot_err = np.sqrt((2*b/berr**2)**2 + (constant1*T/mass_value * Terr/T)**2)
#         new_row = (ion_name, temp_name, wavelength, N*multiple, Nerr*multiple, False, v, verr, b, berr, T, Terr, b_tot, mass_value, z, 0)
#         table.add_row(new_row)

#         absorber_number = 1
#         unique_values = {}

#     for i, row in enumerate(table):
#         v = row['v']
        
#         if v not in unique_values:
#             unique_values[v] = absorber_number
#             absorber_number += 1
        
#         table['absorber_number'][i] = unique_values[v]

#     for i, absorber_num in enumerate(np.unique(table['absorber_number'])):
#         z_UL = table[table['absorber_number'] == absorber_num]['z'].mean()
#         if z_main is not None:
#             if (np.abs(z_main - z_UL) > 0.01):
#                 continue
#         for j, line_name in enumerate(upper_limits):
#             # print('upper limit', line_name)
#             if line_name == table[(table['absorber_number'] == absorber_num) * (table['tempname'] == line_name)]['upper_limit'] == False:
#                 # print('skip1')
#                 continue
#             b_tot_UL = table[table['absorber_number'] == absorber_num]['b_tot'].mean()
#             v_UL = table[table['absorber_number'] == absorber_num]['v'].mean()
#             z_UL = table[table['absorber_number'] == absorber_num]['z'].mean()
#             line = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]
#             wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['wave'][0]
#             if (wavelength * (1+z_UL) > 1795) or (wavelength * (1+z_UL) < 1100):
#                 # print('skip2', wavelength * (1+z_UL), z_UL, wavelength)
#                 continue
#             Wr, WrErr = calc_Wr(spectrum, line, z_UL, v=v_UL * u.km/u.second, dv=[-b_tot_UL*u.km/u.second, b_tot_UL*u.km/u.second], mask=1.0)
#             AOD, AODErr = calc_AOD(spectrum, line, z_UL, v=v_UL * u.km/u.second, dv=[-b_tot_UL*u.km/u.second, b_tot_UL*u.km/u.second], mask=1.0)
#             N, NErr = AOD_to_N(AOD, AODErr, line)
#             ## print('upper limit', line_name, N, NErr, v_UL, b_tot_UL)
#             name = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['name'][0]
#             wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['wave'][0]
            
#             if (((table['ion'] == name) * (table['absorber_number'] == absorber_num) * (table['upper_limit'] == True)).sum() == 0):# and ((table[table['upper_limit'] == False]['ion'] == name).sum() == 0):
#                 # print('added')
#                 # print(line_name, np.log10((ReLU(N) + 2*NErr).value), v_UL, b_tot_UL)
#                 new_row = (name, line_name, wavelength, N, NErr, True, v_UL, None, None, None, None, None, b_tot_UL, mass_value, z_UL, absorber_num)
#                 table.add_row(new_row)
#             else:
#                 index = np.where((table['ion'] == name) * (table['absorber_number'] == absorber_num) *(table['upper_limit'] == True))
#                 compare = ReLU(table[index]['N'] + 2*table[index]['Nerr'])
#                 # print('compare')
#                 # print(line_name, np.log10((ReLU(N) + 2*NErr).value), v_UL, b_tot_UL, table[index]['tempname'][0], np.log10(compare[0]),(((ReLU(N) + 2*NErr) * u.cm**2) < compare)[0])
#                 if ((ReLU(N) + 2*NErr) * u.cm**2) < compare:
#                     # print('replace')
#                     # print(line_name, np.log10((ReLU(N) + 2*NErr).value), v_UL, b_tot_UL)
#                     new_row = (name, line_name, wavelength, N, NErr, True, v_UL, None, None, None, None, None, b_tot_UL, mass_value, z_UL, absorber_num)
#                     table[index] = (new_row)

#     return table

def genSummaryTableNoChain2(spectrum, fit, upper_limits, z_main=None):


    # Define the column names and data types
    colnames = ['ion', 'tempname', 'wavelength', 'N', 'Nerr', 'upper_limit', 'v', 'verr', 'b_other', 'berr','T', 'Terr', 'b_tot', 'mass', 'z', 'absorber_number']
    dtype = [   'U10', 'U10', 'float64','float64','float64',   'bool','float64','float64','float64','float64', 'float64','float64', 'float64','float64', 'float64', 'int']

    # Create an empty Astropy table
    table = Table(names=colnames, dtype=dtype)
    count = 0
    absorber_number = 0

    for i, param in enumerate(fit.cov_matrix.param_names):
        if 'N_' not in param:
            continue
        model_num = int(param.split('_')[1])
        print(model_num)
        name = fit[model_num].name

        temp_name, redshift = name.split('__z=')
        z = np.round(float(redshift), decimals=4)
        if np.any(np.array(fit.cov_matrix.param_names) == ('b_other_' + str(int(param.split('_')[1])))):
            count=i

        ion_name = SEARCH_LINES[SEARCH_LINES['tempname'] == temp_name]['name'][0]
        wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == temp_name]['wave'][0]

        N = fit[model_num].N.value
        b = fit[model_num].b_other.value
        T = fit[model_num].T.value
        v = fit[model_num].v.value
        Nerr = np.sqrt(fit.cov_matrix['N_' + str(model_num), 'N_' + str(model_num)])
        mass_value = fit[model_num].mass.value

        b_tot = np.sqrt(b**2 + constant1**2 * T / mass_value)

        try:
            berr = np.sqrt(fit.cov_matrix['b_other_' + str(model_num), 'b_other_' + str(model_num)])
        except ValueError as e:
            new_row = (ion_name, temp_name, wavelength, N*multiple, Nerr*multiple, False, v, None, b, None, T, None, b_tot, mass_value, z, absorber_number)
            table.add_row(new_row)
            continue

        berr = np.sqrt(fit.cov_matrix['b_other_' + str(model_num), 'b_other_' + str(model_num)])
        Terr = np.sqrt(fit.cov_matrix['T_' + str(model_num), 'T_' + str(model_num)])
        verr = np.sqrt(fit.cov_matrix['v_' + str(model_num), 'v_' + str(model_num)])
        #b_tot_err = np.sqrt((2*b/berr**2)**2 + (constant1*T/mass_value * Terr/T)**2)
        new_row = (ion_name, temp_name, wavelength, N*multiple, Nerr*multiple, False, v, verr, b, berr, T, Terr, b_tot, mass_value, z, absorber_number )
        table.add_row(new_row)

    absorber_number = 0
    unique_velocities = np.unique(table['v'])
    for i, row in enumerate(table):
        v = row['v']
        if v in unique_velocities:
            #index where the velocity is the same
            index = np.where(table['v'] == v)
            table['absorber_number'][index] = absorber_number
            absorber_number += 1

    for j, line_name in enumerate(upper_limits):
        line = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]
        wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['wave'][0]
        if (wavelength * (1+z_main) > 1795) or (wavelength * (1+z_main) < 1100):
            # print('skip2', wavelength * (1+z_UL), z_UL, wavelength)
            continue
        Wr, WrErr = calc_Wr(spectrum, line, z_main, v=0 * u.km/u.second, dv=[-45*u.km/u.second, 45*u.km/u.second], mask=1.0)
        AOD, AODErr = calc_AOD(spectrum, line, z_main, v=0 * u.km/u.second, dv=[-45*u.km/u.second, 45*u.km/u.second], mask=1.0)
        N, NErr = Wr_to_N(Wr, WrErr, line)
        ## print('upper limit', line_name, N, NErr, v_UL, b_tot_UL)
        name = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['name'][0]
        wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['wave'][0]

        if table['upper_limit'][table['ion'] == name] == False:
            continue
        
        if (((table['ion'] == name) * (table['absorber_number'] == 99) * (table['upper_limit'] == True)).sum() == 0):# and ((table[table['upper_limit'] == False]['ion'] == name).sum() == 0):
            # print('added')
            # print(line_name, np.log10((ReLU(N) + 2*NErr).value), v_UL, b_tot_UL)
            new_row = (name, line_name, wavelength, N, NErr, True, 0, None, None, None, None, None, 45, mass_value, z_main, 99)
            table.add_row(new_row)
        else:
            index = np.where((table['ion'] == name) * (table['absorber_number'] == 99) *(table['upper_limit'] == True))
            compare = ReLU(table[index]['N'] + 2*table[index]['Nerr'])
            # print('compare')
            # print(line_name, np.log10((ReLU(N) + 2*NErr).value), v_UL, b_tot_UL, table[index]['tempname'][0], np.log10(compare[0]),(((ReLU(N) + 2*NErr) * u.cm**2) < compare)[0])
            if ((ReLU(N) + 2*NErr) * u.cm**2) < compare:
                # print('replace')
                # print(line_name, np.log10((ReLU(N) + 2*NErr).value), v_UL, b_tot_UL)
                new_row = (name, line_name, wavelength, N, NErr, True, 0, None, None, None, None, None, 45, mass_value, z_main, 99)
                table[index] = (new_row)
    
    return table

        
# def genSummaryTableUpperLimitsOnly(spectrum, upper_limits, z):
#     colnames = ['ion', 'tempname', 'wavelength', 'N', 'NErr']
#     dtype = [   'U10', 'U10', 'float64','float64', 'float64']
#     table = Table(names=colnames, dtype=dtype)

#     for i, line_name in enumerate(upper_limits):
#         line = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]
#         wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['wave'][0]
#         # if z_main is not None:
#         #     if (np.abs(z_main - z) > 0.01):
#         #         continue
#         if (wavelength * (1+z) > 1795) or (wavelength * (1+z) < 1100):
#             continue
#         Wr, WrErr = calc_Wr(spectrum, line, z, mask=1.0)
#         AOD, AODErr = calc_AOD(spectrum, line, z, mask=1.0)
#         N, NErr = Wr_to_N(Wr, WrErr, line)
#         name = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['name'][0]
#         wavelength = SEARCH_LINES[SEARCH_LINES['tempname'] == line_name]['wave'][0]
#         new_row = (name, line_name, wavelength, N, NErr)
#         if (table['ion'] == name).sum() == 0:# and ((table[table['upper_limit'] == False]['ion'] == name).sum() == 0):
#             new_row = (name, line_name, wavelength, N, NErr)
#             table.add_row(new_row)
#         else:
#             index = np.where((table['ion'] == name))
#             compare = ReLU(table[index]['N'] + 2*table[index]['NErr'])
#             if ((ReLU(N) + 2*NErr) * u.cm**2) < compare:
#                 new_row = (name, line_name, wavelength, N, NErr)
#                 table[index] = (new_row)

#     return table


def PrintableSummaryTableNoChain(table):
    # Round each value to 2 decimals
    rounded_table = Table(table)
    
    # Take the logarithm of N
    #rounded_table['N'] = np.log10(rounded_table['N'])

    # rounded_table['Nerr_low'] = np.zeros(len(rounded_table))
    # rounded_table['Nerr_high'] = np.zeros(len(rounded_table))
    
    # Calculate the asymmetric error using Nerr
    detections = rounded_table['upper_limit'] == False
    upper_limits = rounded_table['upper_limit'] == True
    # rounded_table['Nerr_low'][detections] = np.log10(table[detections]['N']) - np.log10(table[detections]['N'] - table[detections]['Nerr'])
    # rounded_table['Nerr_high'][detections] = np.log10(table['N'][detections] + table['Nerr'][detections]) - np.log10(table['N'][detections])
    rounded_table['UL'] = np.log10(ReLU(rounded_table['N']) + 2*rounded_table['Nerr'])
    rounded_table['N'] = np.log10(rounded_table['N'])
    # Reorder the columns
    #rounded_table = rounded_table[['ion', 'tempname', 'wavelength', 'N', 'Nerr_low', 'Nerr_high', 'UL', 'upper_limit', 'v', 'verr', 'b_tot', 'mass', 'z', 'absorber_number']]
    rounded_table = rounded_table[['ion', 'tempname', 'wavelength', 'N', 'UL', 'upper_limit', 'v', 'b_tot', 'mass', 'z', 'absorber_number']]

    # rounded_table['b_therm'] = np.sqrt(k_B * rounded_table['T'] * u.Kelvin / (rounded_table['mass'] * m_p)).to('km/s').value
    # rounded_table['b_therm_err'] = np.sqrt(k_B * rounded_table['Terr'] * u.Kelvin / (rounded_table['mass'] * m_p)).to('km/s').value
    # rounded_table['b_tot'] = np.sqrt(rounded_table['b_other']**2 + rounded_table['b_therm']**2)
    # rounded_table['btot_err'] = np.sqrt(rounded_table['berr']**2 + rounded_table['b_therm_err']**2)
    rounded_table.round(decimals=5)
    #rounded_table = rounded_table[['ion', 'wavelength', 'N', 'Nerr_low', 'Nerr_high', 'upper_limit', 'v', 'verr', 'b_tot', 'btot_err', 'z', 'absorber_number']]

    return rounded_table

def PrintableSummaryTableNoChainTotalN(table):
    # Round each value to 2 decimals
    rounded_table = Table(table)

    rounded_table['total_N'] = False

    detections = rounded_table[rounded_table['upper_limit'] == False]

    unique_ions = np.unique(detections['ion'])


    rounded_table['UL'] = np.log10(ReLU(rounded_table['N']) + 2*rounded_table['Nerr'])
    rounded_table['N'] = np.log10(rounded_table['N'])
    rounded_table = rounded_table[['ion', 'tempname', 'wavelength', 'N', 'UL', 'upper_limit', 'v', 'b_tot', 'mass', 'z', 'absorber_number, 'total_N']]
    rounded_table.round(decimals=5)


    for ion in unique_ions:
        ion_subset = rounded_table[rounded_table['ion'] == ion]
        N_tot = (10**ion_subset['N']).sum()
        mass = ion_subset['mass'].mean()
        z = ion_subset['z'].mean()
        absorber_num = -1
        new_row = (ion, '', np.nan, N_tot, np.nan, False, np.nan, np.nan, np.nan, mass, z, absorber_num, True)
        print(new_row)
        rounded_table.add_row(new_row)
    
    #sort rounded table, first detections and non detections, then by ion mass, then absorber number
    rounded_table.sort(['upper_limit', 'mass', 'absorber_number'])

    return rounded_table