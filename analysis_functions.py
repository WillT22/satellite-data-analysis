import numpy as np
import os
from spacepy import pycdf
import spacepy.omni as omni

#%% Proccess REPT CDF
def process_flux_data(file_paths):
    # Initialize varaibles to be read in
    Epoch = []
    L = []
    Position = []
    FEDU = None
    energy_channels = []
    # Itterate over files in file path
    for file_path in file_paths:
        # Extract filename without path
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")
        # Load the CDF data
        cdf_data = pycdf.CDF(file_path)
        # Read in data
        Epoch.extend(cdf_data["Epoch"][:])
        L.extend(cdf_data["L"][:])
        Position.extend(cdf_data["Position"][:])
        # Get energy channels from first file
        if FEDU is None:
            FEDU = cdf_data["FEDU"][:]
            energy_channels = cdf_data["FEDU_Energy"][:]
        else:
            FEDU = np.vstack((FEDU, cdf_data["FEDU"][:]))
        pitch_angle = cdf_data['FEDU_Alpha'][:]
        pitch_angle = np.where(pitch_angle <= 90, pitch_angle, 180 - pitch_angle)
        cdf_data.close()
    # Convert from km to R_E
    Position = np.array(Position)
    Re = 6378.137 # Earth's Radius
    Position = Position / Re
    # finish reading in data
    return Epoch, L, Position, FEDU, energy_channels, pitch_angle

#%% Function for reading in RBSP ephemeris data
def process_ephem_data(ephem_file_paths):
    # Initialize variables to be read in
    Epoch_ephem = []
    alpha_ephem = []
    Bm_ephem = []
    Lm_ephem = []
    Lstar_ephem = []
    K_ephem = []
    # Itterate over files in file path
    for f, file_path in enumerate(ephem_file_paths):
        # Extract filename without path
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")
        # Load the CDF data
        ephem_data = pycdf.CDF(file_path)
        # Read all but last value for all but last file to prevent duplicated data
        if f < len(ephem_file_paths) - 1:  
            Epoch_ephem.extend(ephem_data['Epoch'][:-1]) 
            Bm_ephem.extend(ephem_data['Bm'][:-1])
            Lm_ephem.extend(ephem_data['Lm_eq'][:-1])
            Lstar_ephem.extend(ephem_data['Lstar'][:-1])
            K_ephem.extend(ephem_data['K'][:-1])
        else:
            Epoch_ephem.extend(ephem_data['Epoch'][:]) 
            alpha_ephem.extend(ephem_data['Alpha'][:])
            Bm_ephem.extend(ephem_data['Bm'][:])
            Lm_ephem.extend(ephem_data['Lm_eq'][:])
            Lstar_ephem.extend(ephem_data['Lstar'][:])
            K_ephem.extend(ephem_data['K'][:])
        ephem_data.close()
    # Convert lists to NumPy arrays
    Epoch_ephem = np.array(Epoch_ephem)
    alpha_ephem = np.array(alpha_ephem)
    Bm_ephem = np.array(Bm_ephem)
    Lm_ephem = np.array(Lm_ephem)
    Lstar_ephem = np.array(Lstar_ephem)
    K_ephem = np.array(K_ephem)
    # Finish rading in and return data
    return Epoch_ephem, alpha_ephem, Lm_ephem, Lstar_ephem, K_ephem

#%% Interpolate Ephemeris data for RBSP times
def interpolate_Ephem(Epoch, Epoch_ephem, Lm_ephem, Lstar_ephem, K_ephem):
    """
    Interpolates ephemeris data (Lm, Lstar, K) to match the time points in Epoch.

    Args:
        Epoch (list of datetime): List of datetime objects representing the desired time points.
        Epoch_ephem (list of datetime): List of datetime objects representing the ephemeris time points.
        Lm_ephem (numpy.ndarray): Array of Lm values corresponding to Epoch_ephem.
        Lstar_ephem (numpy.ndarray): 2D array of Lstar values corresponding to Epoch_ephem.
        K_ephem (numpy.ndarray): 2D array of K values corresponding to Epoch_ephem.

    Returns:
        tuple: A tuple containing the interpolated Lm, Lstar, and K arrays.
    """
    
    # Convert datetime objects to timestamps (floating-point seconds since epoch)
    Epoch_float = [epoch.timestamp() for epoch in Epoch]
    Epoch_ephem_float = [epoch_ephem.timestamp() for epoch_ephem in Epoch_ephem]
    
    # Interpolate Lm values
    Lm_interp = np.interp(Epoch_float, Epoch_ephem_float, Lm_ephem)
    
    # Initialize arrays to store interpolated Lstar and K values
    Lstar_interp = np.zeros((len(Epoch_float), Lstar_ephem.shape[1]))
    K_interp = np.zeros((len(Epoch_float), K_ephem.shape[1]))
    
    # Interpolate Lstar values for each column
    for lstar_col in range(Lstar_ephem.shape[1]):
        Lstar_interp[:,lstar_col] = np.interp(Epoch_float, Epoch_ephem_float, Lstar_ephem[:,lstar_col])
    
    # Interpolate K values for each column
    for k_col in range(K_interp.shape[1]):
        K_interp[:,k_col] = np.interp(Epoch_float, Epoch_ephem_float, K_ephem[:,k_col])
    
    return Lm_interp, Lstar_interp, K_interp

#%% Get Omni data for specified spacetime
def get_Omni(time, position):      
    """
    Retrieves and processes OMNI data for given time points.

    Args:
        time (array-like): Array of datetime objects representing the desired time points.
        position (array-like): Array representing the position (not used in this function, but included per function signature).

    Returns:
        dict: A dictionary containing refined OMNI data, with keys mapped to accepted input labels.
    """

    
    # map keys from file label to accepted input labels
    mag_key_mapping = {
        'Kp_index': 'Kp',
        'Dst_index': 'Dst',
        'PC_N_index': 'dens',  # 'N' maps to 'dens'
        'Plasma_bulk_speed': 'velo',  # 'V' maps to 'velo'
        'Flow_pressure': 'Pdyn',  # 'Pressure' maps to 'Pdyn'  (Using Flow_pressure)
        'By_GSM': 'ByIMF',  # 'BY_GSM' maps to 'ByIMF'
        'Bz_GSM': 'BzIMF',  # 'BZ_GSM' maps to 'BzIMF'
        'AL_index': 'AL',
    }
    
    omnivals_refined = {}
    
    # set unused keys to nan 
    mag_key_unused = ['G1', 'G2', 'G3', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    for key in mag_key_unused:
        omnivals_refined[key] = np.full(len(time), np.nan)
   
    # Get omni values for each time point
    omnivals=omni.get_omni(time, dbase='OMNI2hourly')
    omnivals['Kp_index'] = omnivals['Kp_index']/10
    
    # Map file keys to accepted input labels
    for cdf_key, mag_key in mag_key_mapping.items():
        if cdf_key in omnivals:  # Check if the key exists in the CDF
            omnivals_refined[mag_key] = omnivals[cdf_key][:].copy()
        else:
            print(f"Warning: Key '{cdf_key}' not found in CDF data. Skipping.")
    
    return omnivals_refined

#%% Extend Alpha for more values
def extend_alpha(alpha):
    """
    Extends an array of alpha values by inserting intermediate values between consecutive points.

    Args:
        alpha (list or numpy.ndarray): The input array of alpha values.

    Returns:
        numpy.ndarray: The extended array of alpha values.
    """
    
    # Initialize an empty list to store the extended alpha values.
    alpha_extend = []

    # Iterate through the first 9 elements of the alpha array (excluding the last one).
    for i in range(len(alpha[0:9]) - 1):
        # Append the current alpha value to the extended list.
        alpha_extend.append(alpha[i])

        # Generate intermediate values between the current and next alpha values.
        # np.linspace creates 4 evenly spaced points, and [1:3] selects the middle two points.
        intermediate_values = np.linspace(alpha[i], alpha[i+1], 4)[1:3]

        # Extend the alpha_extend list with the intermediate values.
        alpha_extend.extend(intermediate_values)

    # Append the final alpha value to the extended list.
    alpha_extend.append(alpha[i+1])

    # Convert the extended list to a NumPy array.
    alpha_extend = np.array(alpha_extend)
    
    return alpha_extend

#%% Find times where Lstar is NaN
def find_perigee_times(Lstar, Epoch):
    """
    Finds the start and stop indices and times where Lstar rows are all NaN (usually orbital perigee or missing data).

    Args:
        Lstar (numpy.ndarray): The 2D NumPy array with NaN values.
        Epoch (numpy.ndarray): The 1D NumPy array of time points.

    Returns:
        numpy.ndarray: A 2D NumPy array with start_index, start_time, stop_index, stop_time.
    """

    # Find the indices of rows in Lstar where all elements are NaN.
    # np.all(np.isnan(Lstar), axis=1) creates a boolean array where True indicates rows with all NaNs.
    # np.where(...)[0] returns the indices of the True values.
    Lstar_nan = np.where(np.all(np.isnan(Lstar), axis=1))[0]

    # Initialize an empty list to store the perigee time ranges.
    perigee_times = []

    # Initialize start and stop indices to the first NaN index.
    start_index = Lstar_nan[0]
    stop_index = Lstar_nan[0]

    # Iterate through the NaN indices, starting from the second index.
    for i in range(1, len(Lstar_nan)):
        # Check if the current NaN index is consecutive to the previous stop index.
        if Lstar_nan[i] == stop_index + 1:
            # If consecutive, update the stop index.
            stop_index = Lstar_nan[i]
        else:
            # If not consecutive, it means a perigee time range has ended.
            # Append the start index, start time, stop index, and stop time to the perigee_times list.
            perigee_times.append([start_index, Epoch[start_index], stop_index, Epoch[stop_index]])

            # Start a new perigee time range with the current NaN index.
            start_index = Lstar_nan[i]
            stop_index = Lstar_nan[i]

    # Append the last perigee time range.
    perigee_times.append([start_index, Epoch[start_index], stop_index, Epoch[stop_index]])

    # Convert the perigee_times list to a NumPy array and return it.
    return np.array(perigee_times)

def find_apogee_times(Lstar, Epoch, column_index=8, time_window=100):
    """
    Finds the indices and times of local maxima in a specified column of a 2D Lstar array
    using an if condition, checking within a specified time window.

    Args:
        Lstar (numpy.ndarray): The 2D NumPy array of Lstar values. Each row represents a time point, and each column a parameter.
        Epoch (numpy.ndarray): The 1D NumPy array of datetime objects corresponding to the time points in Lstar.
        column_index (int): The index of the column to use for peak detection. Defaults to 8.
        time_window (int): The number of time steps to check for local maxima. Defaults to 100.

    Returns:
        numpy.ndarray: A 2D NumPy array where each row represents an apogee time, containing:
                         [apogee_index, apogee_time].
    """

    # Initialize an empty list to store the apogee indices and times.
    apogee_times = []

    # Extract the specified column from the Lstar array.
    column_data = Lstar[:, column_index]

    # Replace NaN values in the column data with negative infinity.
    # This ensures that NaN values are not considered as local maxima.
    column_data[np.isnan(column_data)] = -np.inf

    # Iterate through the column data, starting and ending with a buffer of time_window.
    for i in range(time_window, len(column_data) - time_window):
        # Assume the current element is a local maximum initially.
        is_local_max = True

        # Check if the current element is greater than or equal to all other elements within the time window.
        for j in range(i - time_window, i + time_window + 1):
            # Skip comparing the current element with itself.
            if j != i and column_data[j] >= column_data[i]:
                # If any element within the time window is greater than or equal to the current element,
                # it's not a local maximum.
                is_local_max = False
                # Exit the inner loop since we've found it's not a local max.
                break

        # If the current element is a local maximum, append its index and time to the apogee_times list.
        if is_local_max:
            apogee_times.append([i, Epoch[i]])

    # Convert the apogee_times list to a NumPy array and return it.
    return np.array(apogee_times)

#%% Find alpha given K
def find_alpha(K_set, K, alpha):
    """
    Finds the alpha value corresponding to a given K_set by interpolating within a matrix of K values.

    Args:
        K_set (float): The target K value for which to find alpha.
        K (numpy.ndarray): A 2D NumPy array of K values. Each row represents a time point, and each column corresponds to an alpha value.
        alpha (numpy.ndarray): A 1D NumPy array of alpha values corresponding to the columns of K.

    Returns:
        numpy.ndarray: A 1D NumPy array of alpha values, one for each time point in K, corresponding to K_set.
                       NaN is returned for time points where K_set cannot be interpolated.
    """

    # Check if the number of columns in K matches the length of alpha.
    if K.shape[1] != len(alpha):
        raise ValueError("Number of columns in K must match length of alpha.")

    # Initialize an array to store the resulting alpha values, filled with NaN.
    alpha_set = np.full(K.shape[0], np.nan)

    # Iterate through each time point (row) in the K matrix.
    for time_index in range(K.shape[0]):
        # Extract the K values for the current time point.
        row_k = K[time_index, :]

        # Create a mask to identify NaN values in the K row.
        nan_mask = np.isnan(row_k)

        # Create a mask to identify K values that are less than or equal to 1.
        valid_mask = row_k <= 1

        # Combine the masks to exclude NaN values and values greater than 1.
        combined_mask = ~nan_mask & valid_mask

        # Check if there are any valid K values for the current time point.
        if np.any(combined_mask):
            # Extract the valid K values and corresponding alpha values.
            valid_k = row_k[combined_mask]
            valid_alpha = alpha[combined_mask]

            # Sort the valid K values and corresponding alpha values in ascending order of K.
            sort_indices = np.argsort(valid_k)
            valid_k = valid_k[sort_indices]
            valid_alpha = valid_alpha[sort_indices]

            # Check if K_set is within the range of valid K values.
            if np.min(valid_k) <= K_set <= np.max(valid_k):
                # Interpolate the alpha value for K_set using the valid K and alpha values.
                alpha_set[time_index] = np.interp(K_set, valid_k, valid_alpha)

    return alpha_set
        
    
