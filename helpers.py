from header import *  # Importing necessary dependencies

def find_nearest_index(array, number):
    """
    Find the index of the nearest value in an array to a given number.

    Parameters:
    - array (numpy.ndarray): The input array.
    - number (float): The number to which we want to find the nearest value.

    Returns:
    - nearest_index (int): The index of the nearest value in the array.
    """
    absolute_diff = np.abs(array - number)  # Calculate absolute differences
    nearest_index = np.argmin(absolute_diff)  # Find the index of the minimum absolute difference
    return nearest_index

def remove_close_values(arr, min_distance):
    """
    Remove values from an array that are closer to each other than a specified minimum distance.

    Parameters:
    - arr (numpy.ndarray): The input array.
    - min_distance (float): The minimum allowed distance between values.

    Returns:
    - cleaned_arr (numpy.ndarray): The array with close values removed.
    - unremoved_indices (numpy.ndarray): The indices of the values that were not removed.
    """
    sorted_indices = np.argsort(arr)  # Get the indices that would sort the array
    sorted_arr = arr[sorted_indices]  # Sort the array based on the indices
    remove_indices = []

    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] - sorted_arr[i-1] < min_distance:
            remove_indices.append(i-1)  # Mark indices of values that are too close

    cleaned_arr = np.delete(sorted_arr, remove_indices)  # Remove the marked values
    unremoved_indices = np.delete(sorted_indices, remove_indices)  # Get the indices of unremoved values
    return cleaned_arr, unremoved_indices

def find_first_occurrences(arr):
    """
    Find the indices of the first occurrences of elements in an array.

    Parameters:
    - arr (numpy.ndarray): The input array.

    Returns:
    - first_occurrence_indices (numpy.ndarray): An array with the indices of the first occurrences of elements.
    """
    indices_dict = {}
    first_occurrence_indices = np.zeros_like(arr, dtype=int) - 1  # Initialize with -1
    for i, element in enumerate(arr):
        if element not in indices_dict:
            indices_dict[element] = i  # Store the index of the first occurrence of each unique element
            first_occurrence_indices[i] = -1
        else:
            first_occurrence_indices[i] = indices_dict[element]  # Set index for non-first occurrences
    return first_occurrence_indices

def pick_evenly_spaced_values(array, n):
    """
    Pick n evenly spaced values from an array.

    Parameters:
    - array (numpy.ndarray): The input array.
    - n (int): The number of values to pick.

    Returns:
    - indices (numpy.ndarray): The indices of the picked values.
    - picked_values (numpy.ndarray): The picked values from the input array.
    """
    n = min(n, len(array))  # Ensure n is not greater than the length of the array
    indices = np.linspace(0, len(array) - 1, n, dtype=int)  # Generate evenly spaced indices
    return indices, array[indices]  # Return the indices and the corresponding values

def to_velocity(wave, line, z):
    """
    Calculate the velocity based on observed and true wavelengths and redshift.

    Parameters:
    - wave (float): The true wavelength.
    - line (float): The observed wavelength.
    - z (float): The redshift.

    Returns:
    - v (float): The calculated velocity.
    """
    observed = line * u.Angstrom * (1 + z)
    true = wave
    v = c * (true - observed) / true  # Calculate velocity using the speed of light constant
    return v

def log_uniform(a, b, size):
    """
    Generate an array of size random numbers uniformly distributed in log space between a and b.

    Parameters:
    - a (float): The lower bound of the log-uniform distribution.
    - b (float): The upper bound of the log-uniform distribution.
    - size (int): The number of random values to generate.

    Returns:
    - log_uniform_values (numpy.ndarray): An array of random values in log-uniform distribution.
    """
    return 10**np.random.uniform(np.log10(a), np.log10(b), size=size)  # Generate log-uniform random values

def split_string_by_second_uppercase(s):
    """
    Split a string into two parts at the position of the second uppercase character.
    
    Args:
        s (str): The input string.
    
    Returns:
        tuple: A tuple containing two strings:
            - first_part: The substring before the second uppercase character.
            - second_part: The substring starting from the second uppercase character.
              If there is no second uppercase character, the second part is an empty string.
    """
    first_upper_found = False  # Initialize a flag to track the first uppercase character found
    split_index = -1  # Initialize the split index to -1 (indicating no split)

    for i, char in enumerate(s):
        if char.isupper():
            if first_upper_found:
                split_index = i  # Store the index of the second uppercase character
                break
            else:
                first_upper_found = True  # Mark that the first uppercase character has been found

    if split_index >= 0:
        # If a split index was found, split the string into two parts
        first_part = s[:split_index]
        second_part = s[split_index:]
        return first_part, second_part
    else:
        # If no second uppercase character was found, return the original string and an empty string
        return s, ''

def check_values_in_arrays(A, B):
    # Create a set from array B for faster lookup
    set_B = B#set(B)
    
    # Initialize two empty lists for booleans and indices
  #  bool_result = np.full()
    index_result = []
    
    # Iterate through the elements in A
    for value in set_B:
        for index, value2 in enumerate(A):
            if value == value2:
            #    bool_result.append(True)
                index_result.append(index)
          #  else:
          #      bool_result.append(False)
    
    return index_result
