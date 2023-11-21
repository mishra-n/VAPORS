from header import *  # Importing necessary dependencies
from line_info import *  # Importing additional dependencies from other modules
from helpers import *  # Importing helper functions from a different module

# Constants for physical calculations
constant2 = (1 / (m_e * c * u.Hz) * (4 * pi * e.gauss**2)).to(u.cm**2).value
c_kms = c.to(u.km/u.second).value
c_As = c.to(u.Angstrom/u.second).value

# Function to calculate parameters for Voigt profile fitting
def params_to_fit(N, b, v, z, lambda_0, gamma, f):
    """
    Calculate parameters required for Voigt profile fitting.

    Parameters:
    - N (float): Column density.
    - b (float): Doppler parameter.
    - v (float): Velocity.
    - z (float): Redshift.
    - lambda_0 (float): Rest wavelength.
    - gamma (float): Gamma value.
    - f (float): Oscillator strength.

    Returns:
    - lam (float): Wavelength.
    - amplitude_L (float): Amplitude of the Lorentzian component.
    - fwhm_G (float): Full Width at Half Maximum (Gaussian component).
    """
    amplitude_L = N * constant2 * f / gamma

    lam = (v / c_kms) * lambda_0 + lambda_0
    
    fwhm_G = b * lam * (2 * np.sqrt(np.log(2))) / c_kms
    fwhm_G = fwhm_G * (1 + z)
    lam =  lam * (1 + z)

    return lam, amplitude_L, fwhm_G

# Constants for physical calculations
constant1 = (np.sqrt(2 * k_B / (m_p * c**2)) * c).to(u.km/u.second / u.Kelvin**(0.5)).value
multiple = (1e12 / u.cm**2).value / (1e12 / u.cm**2).to(1 / u.micrometer**2).value

# Custom model for Voigt profile fitting
@custom_model
def PhysicalVoigt1D(x, N=1e2, T = 1e3, b_other = 25, v = 0, z = 1, lambda_0=1215.6701, gamma=626500000.0, mass=1.008, f=0.4164):
    """
    Custom model for Voigt profile fitting.

    Parameters:
    - x (numpy.ndarray): Wavelength values.
    - N (float): Column density.
    - T (float): Temperature.
    - b_other (float): Doppler parameter (other component).
    - v (float): Velocity.
    - z (float): Redshift.
    - lambda_0 (float): Rest wavelength.
    - gamma (float): Gamma value.
    - mass (float): Mass.
    - f (float): Oscillator strength.

    Returns:
    - output (numpy.ndarray): Modeled spectrum.
    """
    # Calculate thermal Doppler parameter
    b_therm = constant1 * np.sqrt(T) / np.sqrt(mass)
    
    # Calculate total Doppler parameter
    b_tot = np.sqrt(b_therm**2 + b_other**2)
    x_0_start = lambda_0
    fwhm_L_start = ((gamma / c_As * (x_0_start)**2) * (1 + z) / (2*np.pi))
    
    lam, amplitude_L, fwhm_G = params_to_fit(N * multiple, b_tot, v, z, lambda_0, gamma, f)
    
    # Initialize Voigt profile model
    v1_init = models.Voigt1D(amplitude_L=amplitude_L, x_0=lam, fwhm_L=fwhm_L_start, fwhm_G=fwhm_G)
    
    # Fix the Lorentzian component's Full Width at Half Maximum (FWHM)
    v1_init.fwhm_L.fixed = True
    
    # Generate the modeled spectrum
    output = v1_init(x)
    
    return output

# Function to create an exponential model
def model_1E(fix=True):
    """
    Create an Exponential1D model.

    Parameters:
    - fix (bool): Whether to fix model parameters.

    Returns:
    - e1_init (Exponential1D): Exponential1D model.
    """
    e1_init = models.Exponential1D(amplitude=1., tau=-1, fixed={'tau': fix, 'amplitude': fix}, name='exponential')
    return e1_init


# Function to create a Voigt profile model for a specific spectral line
def model_1V(params, bounds, info_dict):
    """
    Create a Voigt profile model for a specific spectral line with given parameters.

    Parameters:
    - params (dict): Dictionary of parameter values, including 'N' (column density), 'T' (temperature),
                     'b_other' (Doppler parameter), and 'v' (velocity).
    - bounds (list): Parameter bounds.
    - info_dict (dict): Information dictionary containing 'z' (redshift), 'line' (spectral line name).

    Returns:
    - v_init (custom model): A custom Voigt profile model initialized with the provided parameters.
    """
    N_start = params['N']
    T_start = params['T']
    bo_start = params['b_other']
    v_start = params['v']

    z = info_dict['z']
    line = info_dict['line']
    
    # Retrieve line-specific information from a database (SEARCH_LINES)
    lambda_0 = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['wave'][0]
    gamma = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['Gamma'][0]
    f = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['f'][0]
    mass = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['mass'][0]

    # Initialize a custom Voigt profile model
    v_init = PhysicalVoigt1D(N=N_start, T=T_start, b_other=bo_start, v=v_start, z=z, lambda_0=lambda_0, gamma=gamma, f=f, mass=mass, bounds=bounds, name=str(line) + '__z=' + str(z))
    
    # Fix certain parameters
    v_init.z.fixed = True
    v_init.lambda_0.fixed = True
    v_init.mass.fixed = True
    v_init.gamma.fixed = True
    v_init.f.fixed = True
    
    return v_init

# Function to create a combined Voigt and Exponential model
def generic_exp(v_init):
    """
    Create a combined model consisting of a Voigt profile and an Exponential model.

    Parameters:
    - v_init (custom model): A custom Voigt profile model.

    Returns:
    - ve_init (custom model): A combined Voigt and Exponential model.
    """
    e1_init = model_1E(fix=True)
    ve_init = v_init | e1_init
    return ve_init

# Function to create a multi-component Voigt profile model
def model_1A(params, bounds, info_dict):
    """
    Create a multi-component Voigt profile model.

    Parameters:
    - params (list): List of parameter values for each component.
    - bounds (list): Parameter bounds for each component.
    - info_dict (dict): Information dictionary containing 'line_list' (list of spectral lines) and 'z' (redshift).

    Returns:
    - vn_init (custom model): Multi-component Voigt profile model.
    """
    # Extract spectral lines and redshift from the information dictionary
    lines = info_dict['line_list']
    z = info_dict['z']
    vn_init = None
    
    # Loop through lines and create individual Voigt models for each component
    for i, line in enumerate(lines):
        sub_dict = {}
        sub_dict['z'] = z
        sub_dict['line'] = line
        voigt = model_1V(params[i], bounds, sub_dict)
        
        # Combine individual Voigt models into the multi-component model
        if vn_init is None:
            vn_init = voigt
        else:
            vn_init += voigt
            
    # Determine the name of the first component for tying parameters
    if len(params)==1:
        first_name = vn_init.name
    else:
        first_name = vn_init[0].name
        
    tied_b = True
    tied_T = True
    tied_v = True

    # Define functions to tie parameters for components with the same redshift
    def make_tie_b():
        def tie_b(model):
            for k, voigt in enumerate(model):
                if voigt.name == first_name:
                    return model[k].b_other
        return tie_b

    def make_tie_T():
        def tie_T(model):
            for k, voigt in enumerate(model):
                if voigt.name == first_name:
                    return model[k].T
        return tie_T

    def make_tie_v():
        def tie_v(model):
            for k, voigt in enumerate(model):
                if voigt.name == first_name:
                    return model[k].v
        return tie_v
        
    k = None                  
    if tied_b:
        # Tie Doppler parameters (b) for components with the same redshift
        for i, line in enumerate(lines):
            if i == 1:
                vn_init.b_other_1.tied = make_tie_b()
            if i == 2:
                vn_init.b_other_2.tied = make_tie_b()
            if i == 3:
                vn_init.b_other_3.tied = make_tie_b()
            if i == 4:
                vn_init.b_other_4.tied = make_tie_b()
            if i == 5:
                vn_init.b_other_5.tied = make_tie_b()
            if i == 6:
                vn_init.b_other_6.tied = make_tie_b()
            if i == 7:
                vn_init.b_other_7.tied = make_tie_b()
            if i == 8:
                vn_init.b_other_8.tied = make_tie_b()
            if i == 9:
                vn_init.b_other_9.tied = make_tie_b()
            if i == 10:
                vn_init.b_other_10.tied = make_tie_b()
            if i == 11:
                vn_init.b_other_11.tied = make_tie_b()
            if i == 12:
                vn_init.b_other_12.tied = make_tie_b()
    if tied_T:
        # Tie temperature (T) for components with the same redshift
        for i, line in enumerate(lines):
            if i == 1:
                vn_init.T_1.tied = make_tie_T()
            if i == 2:
                vn_init.T_2.tied = make_tie_T()
            if i == 3:
                vn_init.T_3.tied = make_tie_T()
            if i == 4:
                vn_init.T_4.tied = make_tie_T()
            if i == 5:
                vn_init.T_5.tied = make_tie_T()
            if i == 6:
                vn_init.T_6.tied = make_tie_T()
            if i == 7:
                vn_init.T_7.tied = make_tie_T()
            if i == 8:
                vn_init.T_8.tied = make_tie_T()
            if i == 9:
                vn_init.T_9.tied = make_tie_T()
            if i == 10:
                vn_init.T_10.tied = make_tie_T()
            if i == 11:
                vn_init.T_11.tied = make_tie_b()
            if i == 12:
                vn_init.T_12.tied = make_tie_b()

    if tied_v:
        # Tie velocity (v) for components with the same redshift
        for i, line in enumerate(lines):
            if i == 1:
                vn_init.v_1.tied = make_tie_v()
            if i == 2:
                vn_init.v_2.tied = make_tie_v()
            if i == 3:
                vn_init.v_3.tied = make_tie_v()
            if i == 4:
                vn_init.v_4.tied = make_tie_v()
            if i == 5:
                vn_init.v_5.tied = make_tie_v()
            if i == 6:
                vn_init.v_6.tied = make_tie_v()
            if i == 7:
                vn_init.v_7.tied = make_tie_v()
            if i == 8:
                vn_init.v_8.tied = make_tie_v()
            if i == 9:
                vn_init.v_9.tied = make_tie_v()
            if i == 10:
                vn_init.v_10.tied = make_tie_v()
            if i == 11:
                vn_init.v_11.tied = make_tie_v()
            if i == 12:
                vn_init.v_12.tied = make_tie_v()

    return vn_init


# Function to tie column densities (N) for components with the same redshift
def tieNs(v_init):
    """
    Tie the column densities (N) for components with the same redshift in a custom model.

    Parameters:
    - v_init (custom model): A custom Voigt profile model.

    Returns:
    - None
    """
    # Get the number of submodels in the custom model
    n = v_init.n_submodels

    # Initialize empty arrays to store line names, ions, and redshifts
    lines = np.empty(shape=n, dtype='<U10')  # Line names
    ions = np.empty(shape=n, dtype='<U10')    # Ion names
    zs = np.empty(shape=n)                   # Redshift values

    # Iterate through submodels in the custom model
    if n > 1:
        for i, model in enumerate(v_init):
            # Extract line name and redshift from the submodel's name
            line, z = model.name.split('__z=')
            # Retrieve the ion name based on the line name from a database
            ion = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['name'][0]
            # Store the extracted values in respective arrays
            lines[i] = line
            zs[i] = z
            ions[i] = ion
    else:
        # If there is only one submodel, skip the loop entirely
        line, z = v_init.name.split('__z=')
        ion = SEARCH_LINES[SEARCH_LINES['tempname'] == line]['name'][0]
        lines[0] = line
        zs[0] = z
        ions[0] = ion

    # Find unique redshift values and their corresponding indices
    indexes = np.unique(zs, return_index=True)[1]
    unique_z = [zs[index] for index in sorted(indexes)]

    # Initialize an array to store indices of tied N values
    tie_n_indices = np.empty(shape=0, dtype=int)

    # Initialize a variable to track the number of tied N values before the current redshift
    before_n = 0

    # Loop through unique redshift values
    for z in unique_z:
        # Extract ions corresponding to the current redshift
        subset_ions = ions[zs == z]
        
        # Find the first occurrences of ions in the subset
        subset_first_occurrences = find_first_occurrences(subset_ions)
        
        # Adjust the first occurrences to account for previously tied N values
        for i, number in enumerate(subset_first_occurrences):
            if number == -1:
                continue
            subset_first_occurrences[i] = number + before_n

        # Update the count of tied N values before the current redshift
        before_n += len(subset_first_occurrences)

        # Append the adjusted indices to the tie_n_indices array
        tie_n_indices = np.append(tie_n_indices, subset_first_occurrences)

    # Define a function to tie N values for submodels with the same redshift
    def make_tie_N(i):
        def tie_N(model):
            # Find the line names of the first and second submodels to tie
            first = lines[i]
            second = lines[tie_n_indices[i]]
            # Return the N value of the second submodel
            return model[tie_n_indices[i]].N
        return tie_N

    # Loop through submodels in the custom model
    if n > 1:
        for i, sub_model in enumerate(v_init):
            # If the current submodel is not part of the tied N values, continue to the next submodel
            if tie_n_indices[i] == -1:
                continue
            # Tie the N value of the current submodel to the N value of the corresponding submodel
            v_init[i].N.tied = make_tie_N(i)
    else:
        # If there is only one submodel, skip the code entirely
        pass

    
                   