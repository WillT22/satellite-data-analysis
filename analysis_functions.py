import numpy as np
import os
from spacepy import pycdf
import spacepy.omni as omni
import scipy.constants as sc

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
        
    
#%% Calculate energy from set mu and alpha:
electron_E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6)
def energy_from_mu_alpha(Mu_set, Alpha_set, B_local):
    """
    Calculates energy from Mu_set, Alpha_set, and B_local.

    Args:
        Mu_set (numpy.ndarray): A predefined constant value (MeV/G).
        Alpha_set (numpy.ndarray): NumPy array of Alpha values (in degrees).
        B_local (numpy.ndarray): NumPy array of local magnetic field values (in nT).

    Returns:
        numpy.ndarray: NumPy array of calculated energy values (MeV).
    """

    # Convert Alpha_set to radians
    alpha_rad = np.radians(Alpha_set)

    # Calculate sin^2(Alpha)
    sin_squared_alpha = np.sin(alpha_rad)**2

    # Calculate the energy
    kinetic_energy = np.sqrt(2 * electron_E0 * Mu_set * (B_local * 1e-5) / sin_squared_alpha + electron_E0**2) - electron_E0

    return kinetic_energy