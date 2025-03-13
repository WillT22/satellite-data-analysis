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

#%% Find times where Lstar is NaN
def find_apogee_times(Lstar, Epoch):
    """
    Finds the start and stop indices and times where Lstar rows are all NaN (usually orbital apogee or missing data).

    Args:
        Lstar (numpy.ndarray): The 2D NumPy array with NaN values.
        Epoch (numpy.ndarray): The 1D NumPy array of time points.

    Returns:
        numpy.ndarray: A 2D NumPy array with start_index, start_time, stop_index, stop_time.
    """

    Lstar_nan = np.where(np.all(np.isnan(Lstar), axis=1))[0]
    apogee_times = []
    start_index = Lstar_nan[0]
    stop_index = Lstar_nan[0]

    for i in range(1, len(Lstar_nan)):
        if Lstar_nan[i] == stop_index + 1:
            stop_index = Lstar_nan[i]
        else:
            apogee_times.append([start_index, Epoch[start_index], stop_index, Epoch[stop_index]])
            start_index = Lstar_nan[i]
            stop_index = Lstar_nan[i]

    apogee_times.append([start_index, Epoch[start_index], stop_index, Epoch[stop_index]])

    return np.array(apogee_times)

def find_perigee_times(Lstar, Epoch, column_index=8, time_window=100):
    """
    Finds the indices and times of local maxima in a specified column of a 2D Lstar array
    using an if condition, checking within a specified time window.

    Args:
        Lstar (numpy.ndarray): The 2D NumPy array of Lstar values.
        Epoch (numpy.ndarray): The 1D NumPy array of time points.
        column_index (int): The index of the column to use for peak detection.
        time_window (int): The number of time steps to check for local maxima.

    Returns:
        numpy.ndarray: A 2D NumPy array with perigee_index, perigee_time.
    """

    perigee_times = []
    column_data = Lstar[:, column_index]  # Extract the specified column
    column_data[np.isnan(column_data)] = -np.inf  # Replace NaNs with negative infinity

    for i in range(time_window, len(column_data) - time_window):
        is_local_max = True
        for j in range(i - time_window, i + time_window + 1):
            if j != i and column_data[j] >= column_data[i]:
                is_local_max = False
                break

        if is_local_max:
            perigee_times.append([i, Epoch[i]])

    return np.array(perigee_times)