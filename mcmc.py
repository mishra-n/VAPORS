from header import *
from helpers import *
from line_info import *

def genSamples(fit, num):
    """
    Generate Monte Carlo samples from a fitted model.

    Parameters:
    - fit (FittedModel): Fitted model.
    - num (int): Number of samples to generate.

    Returns:
    - samples (array): Monte Carlo samples from the fitted model.
    """
    # Initialize an array to store the means of model parameters
    means = np.zeros(shape=len(fit.cov_matrix.param_names))
    
    # Calculate means for each parameter based on the fitted model
    for i, param_name in enumerate(fit.cov_matrix.param_names):
        index = np.where(np.array(fit.param_names, dtype=str) == param_name)[0]
        means[i] = fit.parameters[index]

    # Generate samples from a multivariate normal distribution using means and covariance matrix
    samples = stats.multivariate_normal.rvs(mean=means, cov=fit.cov_matrix.cov_matrix, size=num)
    
    # Loop through parameter names in the covariance matrix
    for i, param_name in enumerate(fit.cov_matrix.param_names):
        # Find indices where samples are outside parameter bounds
        indices_to_replace = np.where(np.logical_or(samples[:, i] < fit.bounds[fit.cov_matrix.param_names[i]][0], samples[:, i] > fit.bounds[fit.cov_matrix.param_names[i]][1]))

        # Handle different parameter types ('b_', 'v_', 'N_', 'T_')
        if ('b_' in fit.cov_matrix.param_names[i]):
            new_params = np.random.uniform(fit.bounds[fit.cov_matrix.param_names[i]][0], fit.bounds[fit.cov_matrix.param_names[i]][1], size=len(indices_to_replace[0]))
        if ('v_' in fit.cov_matrix.param_names[i]):
            new_params = np.random.uniform(fit.bounds[fit.cov_matrix.param_names[i]][0], fit.bounds[fit.cov_matrix.param_names[i]][1], size=len(indices_to_replace[0]))
        if ('N_' in fit.cov_matrix.param_names[i]):
            new_params = log_uniform(fit.bounds[fit.cov_matrix.param_names[i]][0], fit.bounds[fit.cov_matrix.param_names[i]][1], size=len(indices_to_replace[0]))
        if ('T_' in fit.cov_matrix.param_names[i]):
            new_params = log_uniform(fit.bounds[fit.cov_matrix.param_names[i]][0], fit.bounds[fit.cov_matrix.param_names[i]][1], size=len(indices_to_replace[0]))

        # Replace out-of-bounds samples with new random values
        samples[indices_to_replace, i] = new_params

    return samples

def getMCMCParamNames(complete_fit):
    """
    Get parameter names for MCMC fitting.
    
    Parameters:
    - complete_fit (FittedModel): Fitted model.
    
    Returns:
    - mcmc_params (list): List of parameter names for MCMC fitting.
    """
    mcmc_params = []
    for i,voigt1d in enumerate(complete_fit):
        if voigt1d.name is None:
            continue
        if 'z=' not in voigt1d.name:
            continue
        full_name = voigt1d.name
        temp_name, redshift = full_name.split('__z=')
        line = SEARCH_LINES[SEARCH_LINES['tempname'] == temp_name]
        for name, value in zip(voigt1d.param_names, voigt1d.parameters):
            tied = voigt1d.tied[name]
            fixed = voigt1d.fixed[name]
            if (tied == False) and (fixed == False):
                adding_name = line['name'][0] + '_z=' + str(np.round(float(redshift), decimals=4)) + '_' + name
                if (adding_name + '_0') not in mcmc_params:
                    mcmc_params.append(adding_name + '_0')
                else:
                    counter = 1
                    while (adding_name + '_' + str(counter)) in mcmc_params:
                        print(counter)
                        counter += 1
                    mcmc_params.append(adding_name + '_' + str(counter))

                # mcmc_params.append(line['name'][0] + '_z=' + str(np.round(float(redshift), decimals=4)) + '_' + name)

    return np.array(mcmc_params)

# def getMCMC_error_N(name, z, v):


