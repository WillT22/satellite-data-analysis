import numpy as np
import os
from spacepy import pycdf
import spacepy.omni as omni
import scipy.constants as sc
from scipy.optimize import curve_fit

#%% Proccess REPT CDF
def process_l3_data(file_paths):
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


#%% Calculate energy from set mu and alpha:
electron_E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6) # this is m_0*c^2
def energy_from_mu_alpha(Mu_set, Alpha_set, B_local):
    """
    Calculates energy from Mu_set, Alpha_set, and B_local.

    Args:
        Mu_set (numpy.ndarray or float): A predefined constant value or array (MeV/G).
        Alpha_set (numpy.ndarray): NumPy array of Alpha values (in degrees).
        B_local (numpy.ndarray): NumPy array of local magnetic field values (in nT).

    Returns:
        numpy.ndarray: NumPy array of calculated energy values (MeV).
    """
    
    # Convert Mu_set to a NumPy array if it's a single value
    Mu_set = np.atleast_1d(Mu_set)

    # Convert Alpha_set to radians
    alpha_rad = np.radians(Alpha_set)

    # Calculate sin^2(Alpha)
    sin_squared_alpha = np.sin(alpha_rad)**2

    kinetic_energy = np.zeros((Alpha_set.shape[0], Mu_set.shape[0]))  # Initialize the output array

    for i, mu in enumerate(Mu_set):
        kinetic_energy[:, i] = np.sqrt(2 * electron_E0 * mu * (B_local * 1e-5) / sin_squared_alpha + electron_E0**2) - electron_E0

    return kinetic_energy

#%% Find the averages of fluzes with the same pitch angle
def average_fluxes_by_pitch_angle(FEDU, alpha, energy_channels):
    """
    Averages fluxes for matching pitch angles in FEDU.

    Args:
        FEDU (numpy.ndarray): 3D array of fluxes (time, pitch angle, energy).
        alpha (list or numpy.ndarray): List or array of pitch angle values.
        energy_channels (list or numpy.ndarray): List or array of energy channel values.

    Returns:
        numpy.ndarray: Averaged fluxes array (time, unique pitch angle, energy).
    """

    rounded_alphas = np.round(alpha, 4)
    unique_alphas = np.array(sorted(list(set(rounded_alphas))))
    FEDU_averaged = np.zeros((FEDU.shape[0], len(unique_alphas), len(energy_channels) - 2))

    for time_index in range(FEDU.shape[0]):
        for energy_index in range(len(energy_channels) - 2):
            for pitch_angle_index in range(len(unique_alphas)):
                mirrored_pitch_angle_index = 16 - pitch_angle_index

                if (FEDU[time_index, pitch_angle_index, energy_index] != 0 and
                        FEDU[time_index, mirrored_pitch_angle_index, energy_index] != 0):
                    FEDU_averaged[time_index, pitch_angle_index, energy_index] = np.mean([
                        FEDU[time_index, pitch_angle_index, energy_index],
                        FEDU[time_index, mirrored_pitch_angle_index, energy_index]
                    ])
                elif (FEDU[time_index, pitch_angle_index, energy_index] != 0 and
                      FEDU[time_index, mirrored_pitch_angle_index, energy_index] == 0):
                    FEDU_averaged[time_index, pitch_angle_index, energy_index] = FEDU[time_index, pitch_angle_index, energy_index]
                elif (FEDU[time_index, pitch_angle_index, energy_index] == 0 and
                      FEDU[time_index, mirrored_pitch_angle_index, energy_index] != 0):
                    FEDU_averaged[time_index, pitch_angle_index, energy_index] = FEDU[time_index, mirrored_pitch_angle_index, energy_index]
                else:
                    FEDU_averaged[time_index, pitch_angle_index, energy_index] = 0

    return FEDU_averaged

#%% Interpolated Flux v Pitch Angle using exponential between points
def interpolate_flux_by_alpha(FEDU_averaged, alpha, alpha_set):

    # Round alpha values to avoid precision issues when comparing.
    rounded_alphas = np.round(alpha, 4)
    unique_alphas = np.array(sorted(list(set(rounded_alphas))))
    
    modified_array = FEDU_averaged.copy()
    modified_array[np.where(modified_array ==0)] = 1
    
    # Initialize an array to store the interpolated flux values.
    # Dimensions: (number of time points, symmetric pitch angle bins, set mu values)
    FEDU_interp_alpha = np.zeros((FEDU_averaged.shape[0], FEDU_averaged.shape[2]))

    # Iterate through each time point.
    for time_index in range(FEDU_averaged.shape[0]):
        # Iterate through each pitch angle bin
        for energy_index in range(FEDU_averaged.shape[2]):          
            # Filter non-positive and NaN values
            valid_indices = np.where((modified_array[time_index, :, energy_index] > 0) &
                                     (~np.isnan(modified_array[time_index, :, energy_index])))[0]
            
            # Check if there are enough non-zero data points for interpolation.
            if len(valid_indices) > 1:            
                valid_alphas = unique_alphas[valid_indices]
                insertion_point = np.searchsorted(valid_alphas, alpha_set[time_index])
                if insertion_point < len(valid_alphas):
                    lower_alpha_val = valid_alphas[insertion_point - 1]
                    upper_alpha_val = valid_alphas[insertion_point]
    
                    if upper_alpha_val - lower_alpha_val > 25:
                        FEDU_interp_alpha[time_index, energy_index] = np.nan
                    else:
                        # Perform exponential interpolation
                        log_flux_interp = np.interp(
                            alpha_set[time_index], 
                            valid_alphas,
                            np.log(modified_array[time_index, valid_indices, energy_index]))
                        FEDU_interp_alpha[time_index, energy_index] = np.exp(log_flux_interp)
                else:
                    FEDU_interp_alpha[time_index, energy_index] = np.nan  
            else:
                # Store NaN if there are not enough valid data points for interpolation.
                FEDU_interp_alpha[time_index, energy_index] = np.nan
    
    return FEDU_interp_alpha

#%% Interpolate Flux v Kinetic Energy using exponential between points
def interpolate_flux_by_energy(FEDU_interp_alpha, energy_channels, energy_set):
    
    modified_array = FEDU_interp_alpha.copy()
    
    # Initialize an array to store the interpolated flux values.
    # Dimensions: (number of time points, number of set mu values)
    FEDU_interp_energy = np.zeros((FEDU_interp_alpha.shape[0],energy_set.shape[1]))

    # Iterate thorugh each set mu value
    for mu_set_index in range(energy_set.shape[1]):
        # Iterate through each time point.
        for time_index in range(FEDU_interp_alpha.shape[0]):
            # Filter non-positive and NaN values
            valid_indices = np.where((modified_array[time_index, :] > 0) &
                                     (~np.isnan(modified_array[time_index, :])))[0]
    
            # Check if there are enough non-zero data points for interpolation.
            if len(valid_indices) > 1:            
                valid_energies = energy_channels[valid_indices]
                insertion_point = np.searchsorted(valid_energies, energy_set[time_index, mu_set_index])
                if insertion_point < len(valid_energies):
                    lower_E_val = valid_energies[insertion_point - 1]
                    upper_E_val = valid_energies[insertion_point]
    
                    if upper_E_val - lower_E_val > 5:
                        FEDU_interp_energy[time_index, mu_set_index] = np.nan
                    else:
                        # Perform exponential interpolation
                        log_flux_interp = np.interp(
                            energy_set[time_index, mu_set_index],
                            energy_channels[:-2][valid_indices],
                            np.log(modified_array[time_index, valid_indices]))
                        FEDU_interp_energy[time_index, mu_set_index] = np.exp(log_flux_interp)
                else:
                    FEDU_interp_energy[time_index, mu_set_index] = np.nan
            else:
                # Store NaN if there are not enough valid data points for interpolation.
                FEDU_interp_energy[time_index, mu_set_index] = np.nan

    return FEDU_interp_energy

#%% Calculate PSD from Flux and Energy
electron_E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6) # this is m_0*c^2
def find_psd(FEDU_interp_aE, energy_set):
    psd = np.zeros((FEDU_interp_aE.shape[0], FEDU_interp_aE.shape[1]))
    
    # Iterate thorugh each set mu value
    for mu_set_index in range(FEDU_interp_aE.shape[1]):
        # Iterate through each time point.
        for time_index in range(FEDU_interp_aE.shape[0]):
            if not (np.isnan(FEDU_interp_aE[time_index, mu_set_index]) or FEDU_interp_aE[time_index, mu_set_index] == 0):
                E_rel = energy_set[time_index, mu_set_index]**2 + 2*energy_set[time_index, mu_set_index] * electron_E0
                psd[time_index, mu_set_index] = FEDU_interp_aE[time_index, mu_set_index]/E_rel * 1.66e-10 * 1e-3 * 200.3
            else:
                # Store NaN if calculation is invalid.
                psd[time_index, mu_set_index] = np.nan
    
    return psd