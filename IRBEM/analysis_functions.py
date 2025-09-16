import numpy as np
import os
from spacepy import pycdf
import datetime
import spacepy.omni as omni
import scipy.constants as sc

#%% Proccess REPT CDF
def process_l3_data(file_paths):
    # Initialize varaibles to be read in
    Epoch = []
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
    return Epoch, Position, FEDU, energy_channels, pitch_angle

#%% Time average for 1 minute resolution
def time_average(epoch, position, FEDU):
    
    # Find minutes of whole period
    epoch_minutes = [epoch[0].replace(second=0, microsecond=0)]
    for time_index in range(len(epoch[1:])):
        if epoch[time_index].minute != epoch[time_index-1].minute:
            epoch_minutes.append(epoch[time_index].replace(second=0, microsecond=0))
            
    # Find average position each minute
    average_positions = []
    average_FEDU = []
    for minute_index in range(len(epoch_minutes)):
        minute_start = epoch_minutes[minute_index] - datetime.timedelta(seconds=30)
        minute_end =  epoch_minutes[minute_index] + datetime.timedelta(seconds=30)
        minute_indices = np.where((np.array(epoch) >= minute_start) & (np.array(epoch) < minute_end))[0]
        
        if minute_indices.size > 0:
            average_positions.append(np.mean(position[minute_indices], axis=0))
            average_FEDU.append(np.mean(FEDU[minute_indices], axis=0))
        else:
            average_positions.append(np.array([np.nan, np.nan, np.nan]))
            average_FEDU.append(np.full((FEDU.shape[1], FEDU.shape[2]), np.nan))
    
    return epoch_minutes, np.array(average_positions), np.array(average_FEDU)

#%% Find the averages of fluzes with the same pitch angle
def average_fluxes_by_pitch_angle(FEDU, alpha_unique, energy_channels):
    """
    Averages fluxes for matching pitch angles in FEDU.

    Args:
        FEDU (numpy.ndarray): 3D array of fluxes (time, pitch angle, energy).
        alpha (list or numpy.ndarray): List or array of pitch angle values.
        energy_channels (list or numpy.ndarray): List or array of energy channel values.

    Returns:
        numpy.ndarray: Averaged fluxes array (time, unique pitch angle, energy).
    """
    FEDU_averaged = np.zeros((FEDU.shape[0], len(alpha_unique), len(energy_channels)))

    for time_index in range(FEDU.shape[0]):
        for energy_index in range(len(energy_channels)):
            for pitch_angle_index in range(len(alpha_unique)):
                mirrored_pitch_angle_index = 16 - pitch_angle_index
                
                value1 = FEDU[time_index, pitch_angle_index, energy_index]
                value2 = FEDU[time_index, mirrored_pitch_angle_index, energy_index]

                is_nan1 = np.isnan(value1)
                is_nan2 = np.isnan(value2)

                if not is_nan1 and not is_nan2:
                    FEDU_averaged[time_index, pitch_angle_index, energy_index] = np.mean([value1, value2])
                elif not is_nan1 and is_nan2:
                    FEDU_averaged[time_index, pitch_angle_index, energy_index] = value1
                elif is_nan1 and not is_nan2:
                    FEDU_averaged[time_index, pitch_angle_index, energy_index] = value2
                else:
                    FEDU_averaged[time_index, pitch_angle_index, energy_index] = np.nan

    return FEDU_averaged

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


#%% Interpolated Flux v Pitch Angle using exponential between points
def interpolate_flux_by_alpha(FEDU_averaged, alpha_unique, alpha_set):
    """
    Interpolates flux as a function of pitch angle using an exponential
    interpolation between valid data points.

    Args:
        FEDU_averaged (numpy.ndarray): 3D array of averaged flux values
            (time, pitch angle, energy).
        alpha (numpy.ndarray): 1D array of measured pitch angle values (in degrees).
        alpha_set (numpy.ndarray): 1D array of target pitch angle values (in degrees)
            at which to interpolate for each time point.

    Returns:
        numpy.ndarray: 2D array of interpolated flux values (time, energy)
            at the target pitch angles. Returns NaN if interpolation is not
            possible due to insufficient data or large gaps.
    """

    # Copy array for manipulation
    modified_array = FEDU_averaged.copy()

    # Initialize an array to store the interpolated flux values.
    # Dimensions: (number of time points, energy)
    FEDU_interp_alpha = np.zeros((FEDU_averaged.shape[0], FEDU_averaged.shape[2]))

    # Iterate through each time point.
    for time_index in range(FEDU_averaged.shape[0]):
        # Iterate through each energy bin
        for energy_index in range(FEDU_averaged.shape[2]):
            # Filter NaN and zero values
            valid_indices = np.where((modified_array[time_index, :, energy_index] > 0) &
                         (~np.isnan(modified_array[time_index, :, energy_index])))[0]

            # Check if there are enough valid data points for interpolation.
            if len(valid_indices) > 1:
                valid_alphas = alpha_unique[valid_indices]
                # Find where the target alpha would be inserted in the sorted valid alphas
                insertion_point = np.searchsorted(valid_alphas, alpha_set[time_index])
                # If the target alpha would be interpolated (within the range of valid alphas)
                if 0 < insertion_point < len(valid_alphas):
                    # Perform exponential interpolation
                    log_flux_interp = np.interp(
                        alpha_set[time_index],
                        valid_alphas,
                        np.log(modified_array[time_index, valid_indices, energy_index])
                    )
                    FEDU_interp_alpha[time_index, energy_index] = np.exp(log_flux_interp)
                # If the target alpha would be extrapolated (outside the range of valid alphas), replace with NaN
                else:
                    FEDU_interp_alpha[time_index, energy_index] = np.nan
            # If there are not enough points for interpolation, set results as NaN
            else:
                # Store NaN if there are not enough valid data points for interpolation.
                FEDU_interp_alpha[time_index, energy_index] = np.nan

    return FEDU_interp_alpha

#%% Interpolate Flux v Kinetic Energy using exponential between points
def interpolate_flux_by_energy(FEDU_interp_alpha, energy_channels, energy_set):
    """
    Interpolates flux as a function of kinetic energy using an exponential
    interpolation between valid data points.

    Args:
        FEDU_interp_alpha (numpy.ndarray): 2D array of flux values
            (time, pitch angle).
        energy_channels (numpy.ndarray): 1D array of measured kinetic energy
            channel values (in MeV).
        energy_set (numpy.ndarray): 2D array of target kinetic energy values
            (time, mu_set) at which to interpolate.

    Returns:
        numpy.ndarray: 2D array of interpolated flux values (time, mu_set)
            at the target kinetic energies. Returns NaN if interpolation is not
            possible due to insufficient data or large energy gaps.
    """
    # Create a copy of the input array to avoid modifying the original.
    modified_array = FEDU_interp_alpha.copy()
    # Initialize an array to store the interpolated flux values.
    # Dimensions: (number of time points, number of target mu values)
    FEDU_interp_energy = np.zeros((FEDU_interp_alpha.shape[0], energy_set.shape[1]))

    # Iterate through each target mu value.
    for mu_set_index in range(energy_set.shape[1]):
        # Iterate through each time point.
        for time_index in range(FEDU_interp_alpha.shape[0]):
            # Filter out non-positive and NaN flux values for the current pitch angle.
            valid_indices = np.where((modified_array[time_index, :] > 0) &
                         (~np.isnan(modified_array[time_index, :])))[0]
            
            # Check if there are at least two valid data points for interpolation.
            if len(valid_indices) > 1:
                # Extract the valid energy channel values corresponding to the valid flux points.
                valid_energies = energy_channels[valid_indices]
                # Find the index where the target energy would be inserted
                # to maintain the sorted order of valid energies.
                insertion_point = np.searchsorted(valid_energies, energy_set[time_index, mu_set_index])

                # Check if the target energy falls within the range of valid energies (not extrapolation).
                if 0 < insertion_point < len(valid_energies):
                    # Interpolate the logarithm of the flux.
                    log_flux_interp = np.interp(
                        energy_set[time_index, mu_set_index],
                        energy_channels[valid_indices],
                        np.log(modified_array[time_index, valid_indices])
                    )
                    # Exponentiate the result to get the interpolated flux.
                    FEDU_interp_energy[time_index, mu_set_index] = np.exp(log_flux_interp)
                # If the target energy is outside the range of valid energies, set to NaN.
                else:
                    FEDU_interp_energy[time_index, mu_set_index] = np.nan
            # If there are not enough valid data points, set the result to NaN.
            else:
                # Store NaN if there are fewer than two valid data points.
                FEDU_interp_energy[time_index, mu_set_index] = np.nan

    return FEDU_interp_energy
#%% Calculate PSD from Flux and Energy
# Define the relativistic energy conversion factor for an electron.
# This converts the rest mass energy of an electron (m_0*c^2) from Joules to MeV.
electron_E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6)

def find_psd(FEDU_interp_aE, energy_set):
    """
    Calculates the phase space density (PSD) from the interpolated flux
    and corresponding energy values.

    Args:
        FEDU_interp_aE (numpy.ndarray): 2D array of interpolated flux values
            (time, mu_set).
        energy_set (numpy.ndarray): 2D array of energy values (in MeV)
            corresponding to the flux values in FEDU_interp_aE (time, mu_set).

    Returns:
        numpy.ndarray: 2D array of phase space density (PSD) values
            (time, mu_set). Returns NaN for invalid input flux or energy.
    """
   
    # Initialize an array to store the calculated PSD values with the same
    # shape as the input flux array.
    psd = np.zeros((FEDU_interp_aE.shape[0], FEDU_interp_aE.shape[1]))

    # Iterate through each value of the adiabatic invariant mu (magnetic moment).
    for mu_set_index in range(FEDU_interp_aE.shape[1]):
        # Iterate through each time point.
        for time_index in range(FEDU_interp_aE.shape[0]):
            # Check if the interpolated flux and corresponding energy are valid
            # (not NaN and not zero). PSD calculation is not meaningful for these values.
            if not (np.isnan(FEDU_interp_aE[time_index, mu_set_index])):
                # Calculate the relativistic kinetic energy term (E^2 + 2*E*E0),
                # where E is the kinetic energy and E0 is the rest mass energy.
                E_rel = energy_set[time_index, mu_set_index]**2 + \
                        2 * energy_set[time_index, mu_set_index] * electron_E0
                # Calculate the phase space density (PSD) using the formula:
                # PSD = Flux / (E^2 + 2*E*E0) * conversion_factor
                # The conversion factor includes units adjustments.
                psd[time_index, mu_set_index] = \
                    FEDU_interp_aE[time_index, mu_set_index] / E_rel * 1.66e-10 * 1e-3 * 200.3
            else:
                # If the flux or energy is invalid, store NaN in the PSD array
                # for that specific time and mu value.
                psd[time_index, mu_set_index] = np.nan

    return psd