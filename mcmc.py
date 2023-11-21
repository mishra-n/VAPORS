from header import *
from helpers import *

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
