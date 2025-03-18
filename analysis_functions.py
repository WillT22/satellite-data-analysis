import numpy as np
import os
from spacepy import pycdf
import spacepy.omni as omni
from scipy.optimize import curve_fit

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
    Epoch_float = [epoch.timestamp() for epoch in Epoch]
    Epoch_ephem_float = [epoch_ephem.timestamp() for epoch_ephem in Epoch_ephem]
    Lm_interp = np.interp(Epoch_float, Epoch_ephem_float, Lm_ephem)
    
    Lstar_interp = np.zeros((len(Epoch_float), Lstar_ephem.shape[1]))
    K_interp = np.zeros((len(Epoch_float), K_ephem.shape[1]))
    for lstar_col in range(Lstar_ephem.shape[1]):
        Lstar_interp[:,lstar_col] = np.interp(Epoch_float, Epoch_ephem_float, Lstar_ephem[:,lstar_col])
    for k_col in range(K_interp.shape[1]):
        K_interp[:,k_col] = np.interp(Epoch_float, Epoch_ephem_float, K_ephem[:,k_col])
    return Lm_interp, Lstar_interp, K_interp

#%% Get Omni data for specified spacetime
def get_Omni(time, position):      
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
    mag_key_unused = ['G1', 'G2', 'G3', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    omnivals_refined = {}
    for key in mag_key_unused:
        omnivals_refined[key] = np.full(len(time), np.nan)
    omnivals=omni.get_omni(time, dbase='OMNI2hourly')
    omnivals['Kp_index'] = omnivals['Kp_index']/10
    for cdf_key, mag_key in mag_key_mapping.items():
        if cdf_key in omnivals:  # Check if the key exists in the CDF
            omnivals_refined[mag_key] = omnivals[cdf_key][:].copy()
        else:
            print(f"Warning: Key '{cdf_key}' not found in CDF data. Skipping.")
    return omnivals_refined

#%% Extend Alpha for more values
def extend_alpha(alpha):
    alpha_extend = []
    for i in range(len(alpha[0:9]) - 1):
        alpha_extend.append(alpha[i])
        intermediate_values = np.linspace(alpha[i], alpha[i+1], 4)[1:3] #4 points, get the middle 2
        alpha_extend.extend(intermediate_values)
    alpha_extend.append(alpha[i+1]) #add the final point.
    
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

    Lstar_nan = np.where(np.all(np.isnan(Lstar), axis=1))[0]
    perigee_times = []
    start_index = Lstar_nan[0]
    stop_index = Lstar_nan[0]

    for i in range(1, len(Lstar_nan)):
        if Lstar_nan[i] == stop_index + 1:
            stop_index = Lstar_nan[i]
        else:
            perigee_times.append([start_index, Epoch[start_index], stop_index, Epoch[stop_index]])
            start_index = Lstar_nan[i]
            stop_index = Lstar_nan[i]

    perigee_times.append([start_index, Epoch[start_index], stop_index, Epoch[stop_index]])

    return np.array(perigee_times)

def find_apogee_times(Lstar, Epoch, column_index=8, time_window=100):
    """
    Finds the indices and times of local maxima in a specified column of a 2D Lstar array
    using an if condition, checking within a specified time window.

    Args:
        Lstar (numpy.ndarray): The 2D NumPy array of Lstar values.
        Epoch (numpy.ndarray): The 1D NumPy array of time points.
        column_index (int): The index of the column to use for peak detection.
        time_window (int): The number of time steps to check for local maxima.

    Returns:
        numpy.ndarray: A 2D NumPy array with apogee_index, apogee_time.
    """

    apogee_times = []
    column_data = Lstar[:, column_index]  # Extract the specified column
    column_data[np.isnan(column_data)] = -np.inf  # Replace NaNs with negative infinity

    for i in range(time_window, len(column_data) - time_window):
        is_local_max = True
        for j in range(i - time_window, i + time_window + 1):
            if j != i and column_data[j] >= column_data[i]:
                is_local_max = False
                break

        if is_local_max:
            apogee_times.append([i, Epoch[i]])

    return np.array(apogee_times)

#%% 
def exponential_fit(x, a, b, c):
     """Exponential function: a * exp(b * x) + c"""
     return a * np.exp(b * x) + c
 
def power_law_fit(x, a, b, c):
    """
    Defines a power law function with an offset: y = a * x^b + c

    Args:
        x (float or numpy.ndarray): The independent variable.
        a (float): The scaling factor.
        b (float): The exponent.
        c (float): The offset.

    Returns:
        float or numpy.ndarray: The dependent variable (y).
    """
    return a * x**b + c

def fit_K_v_alpha(K, alpha):
    """
    Fits an exponential function to each row of K (columns 0-8) while handling NaN values.

    Args:
        K (numpy.ndarray): 2D array with shape (num_rows, 17).
        alpha (numpy.ndarray): 1D array representing pitch angles.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              fitted parameters (a, b, c) for each row.
    """

    fitted_params = []
    alpha = alpha[0:int((K.shape[1]+1)/2-1)]
    for row_index in range(K.shape[0]):
        y_values = K[row_index, 0:int((K.shape[1]+1)/2-1)]  # Extract y-values for the current row
        nan_mask = ~np.isnan(y_values)  # Create mask for NaN values
        valid_y_values = y_values[nan_mask]  # Apply mask to y values.
        valid_x_values = np.array(alpha)[nan_mask]  # Apply mask to x values.

        if len(valid_y_values) < 3:  # Ensure there are enough values to fit.
            fitted_params.append({'a': np.nan, 'b': np.nan, 'c': np.nan})
            continue

        try:
            params, covariance = curve_fit(exponential_fit, valid_x_values, valid_y_values, p0=[1, -0.1, 0])
            fitted_params.append({'a': params[0], 'b': params[1], 'c': params[2]})
        except RuntimeError:

            fitted_params.append({'a': np.nan, 'b': np.nan, 'c': np.nan})

    return fitted_params

#%% Find alpha given K
def find_alpha(K_set, K, alpha):
    if K.shape[1] != len(alpha):
        raise ValueError("Number of columns in K must match length of alpha.")

    alpha_set = np.full(K.shape[0], np.nan)  # Initialize with NaN

    for time_index in range(K.shape[0]):
        row_k = K[time_index, :]
        nan_mask = np.isnan(row_k)
        valid_mask = row_k <= 1

        # Combine masks to exclude NaNs and values > 1
        combined_mask = ~nan_mask & valid_mask

        if np.any(combined_mask):  # Check if there are any valid values
            valid_k = row_k[combined_mask]
            valid_alpha = alpha[combined_mask]

            # Ensure valid_k is sorted for interpolation
            sort_indices = np.argsort(valid_k)
            valid_k = valid_k[sort_indices]
            valid_alpha = valid_alpha[sort_indices]

            if np.min(valid_k) <= K_set <= np.max(valid_k): #make sure K_set is in range.
                alpha_set[time_index] = np.interp(K_set, valid_k, valid_alpha)

    return alpha_set
        
    
