import numpy as np
import os
import glob
from spacepy import pycdf
from spacepy.time import Ticktock
from spacepy.coordinates import Coords
import datetime as dt
import matplotlib.dates as mdates
import scipy.constants as sc
import pandas as pd

#%% Proccess REPT CDF
def process_l3_data(file_paths):
    # Initialize varaibles to be read in
    sat_data = {}
    sat_data['Epoch'] = []
    sat_data['Position'] = []
    sat_data['MLT'] = []
    sat_data['Energy_Channels'] = []
    sat_data['Pitch_Angles'] = []
    sat_data['FEDU'] = None
    
    # Itterate over files in file path
    for file_path in file_paths:
        # Extract filename without path
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")
        # Load the CDF data
        cdf_data = pycdf.CDF(file_path)
        # Read in data
        sat_data['Epoch'].extend(cdf_data["Epoch"][:])
        sat_data['Position'].extend(cdf_data["Position"][:])
        sat_data['MLT'].extend(cdf_data["MLT"][:])
        # Get energy channels from first file
        if sat_data['FEDU'] is None:
            sat_data['FEDU'] = cdf_data["FEDU"][:]
            sat_data['Energy_Channels'] = cdf_data["FEDU_Energy"][:]
        else:
            sat_data['FEDU'] = np.vstack((sat_data['FEDU'], cdf_data["FEDU"][:]))
        sat_data['Pitch_Angles'] = cdf_data['FEDU_Alpha'][:]
        sat_data['Pitch_Angles'] = np.where(sat_data['Pitch_Angles'] <= 90, sat_data['Pitch_Angles'], 180 - sat_data['Pitch_Angles'])
        cdf_data.close()
    print(f"Converting to GSM...")
    # Convert to Ticktock
    sat_data['Epoch'] = Ticktock(sat_data['Epoch'], dtype='UTC')
    # Convert from km to R_E, and from GEO to GSM
    sat_data['Position'] = np.array(sat_data['Position'])
    Re = 6378.137 # Earth's Radius
    sat_data['Position'] = Coords(sat_data['Position'] / Re, 'GEO', 'car')
    sat_data['Position'].ticks = sat_data['Epoch']
    sat_data['Position'] = sat_data['Position'].convert('GSM','car')

    print("Data Loaded \n") 
    return sat_data

#%% Time average for 1 minute resolution
def time_average(sat_data, sat_name):
    
    epoch = sat_data['Epoch'].UTC
    position = sat_data['Position'].data # Shape: (time, 3)
    FEDU = sat_data['FEDU'] # Shape: (time, measured_pitch_angle, energy)
    # Find minutes of whole period
    epoch_minutes = [epoch[0].replace(second=0, microsecond=0)]
    for time_index in range(len(epoch[1:])):
        if epoch[time_index].minute != epoch[time_index-1].minute:
            epoch_minutes.append(epoch[time_index].replace(second=0, microsecond=0))
            
    # Find average position each minute
    average_epochs = []
    average_positions = []
    average_FEDU = []
    minutes_mask = len(epoch_minutes)*[True]
    for minute_index in range(len(epoch_minutes)):
        minute_start = epoch_minutes[minute_index] - dt.timedelta(seconds=30)
        minute_end =  epoch_minutes[minute_index] + dt.timedelta(seconds=30)
        minute_indices = np.where((np.array(epoch) >= minute_start) & (np.array(epoch) < minute_end))[0]
        
        if minute_indices.size > 0:
            average_epochs.append(epoch_minutes[minute_index])
            average_positions.append(np.mean(position[minute_indices], axis=0))
            average_FEDU.append(np.mean(FEDU[minute_indices], axis=0))
        else:
            minutes_mask[minute_index] = False
    sat_data['Epoch'] = Ticktock(average_epochs, dtype='UTC')
    sat_data['Position'] = Coords(average_positions, 'GSM', 'car')
    sat_data['Position'].ticks = sat_data['Epoch']
    sat_data['FEDU'] = np.array(average_FEDU)
    return sat_data

#%% Extract Magentometer Data and match with nearest time point
def find_mag(sat_data, sat_name):
    mag_folder = '/home/wzt0020/REPT_data/MagData/'

    Epoch = sat_data['Epoch'].UTC
    date = min(Epoch).date()
    end_date = max(Epoch).date()
    
    mag_data = {}
    mag_data['Epoch'] = []
    mag_data['b_satellite'] = []

    
    if max(Epoch).year == min(Epoch).year:
        file_base = os.path.join(mag_folder, sat_name, str(max(Epoch).year))
        while date <= end_date:
            date_str = date.strftime("%Y%m%d")
            filename_pattern = f"*_magnetometer_4sec-gsm_emfisis-l3_{date_str}_v*.cdf"
            filepath_matches = glob.glob(os.path.join(file_base, filename_pattern))
            file_path = filepath_matches[0]
            cdf_data = pycdf.CDF(file_path)
            mag_data['Epoch'].extend(cdf_data['Epoch_centered'][:])
            mag_data['b_satellite'].extend(cdf_data['Magnitude'][:])
            date += dt.timedelta(days=1)
        mag_data['Epoch'] = Ticktock(mag_data['Epoch'], dtype='UTC')
        mag_data['b_satellite'] = np.array(mag_data['b_satellite'])*1e-5 # Convert to Gauss

    epoch_nums = mdates.date2num(Epoch)
    mag_epochs_nums = mdates.date2num(mag_data['Epoch'].UTC)
    idx = np.searchsorted(mag_epochs_nums, epoch_nums)
    idx = np.clip(idx, 1, len(mag_epochs_nums) - 1)
    nearest_time = np.where(np.abs(mag_epochs_nums[idx - 1] - epoch_nums) <= np.abs(mag_epochs_nums[idx] - epoch_nums), idx - 1, idx)
    sat_data['b_satellite'] = mag_data['b_satellite'][nearest_time]
    print('Magnetic Field Data Extracted')
    return sat_data


#%% Find the averages of fluzes with the same pitch angle
def Average_FluxbyPA(sat_data, sat_name):
    """
    Averages fluxes for matching pitch angles in FEDU.

    Args:
        FEDU (numpy.ndarray): 3D array of fluxes (time, pitch angle, energy).
        alpha (list or numpy.ndarray): List or array of pitch angle values.
        energy_channels (list or numpy.ndarray): List or array of energy channel values.

    Returns:
        numpy.ndarray: Averaged fluxes array (time, unique pitch angle, energy).
    """
    FEDU = sat_data['FEDU'] # Shape: (time, measured_pitch_angle, energy)
    Energy_Channels = sat_data['Energy_Channels'] # Shape: (energy,)
    
    # Ensure Pitch_Angles is a NumPy array for consistent indexing
    pitch_angles = np.asarray(sat_data['Pitch_Angles'])

    alpha_unique = np.array(sorted(list(set(np.round(pitch_angles[pitch_angles <= 90], 4)))))

    FEDU_averaged = np.full((FEDU.shape[0], len(alpha_unique), FEDU.shape[2]), np.nan, dtype=np.float64)
    # Shape: (time, unique_pitch_angle, energy)

    for pitch_angle_idx_unique, alpha_val in enumerate(alpha_unique):

        PA_mask = (np.round(pitch_angles, 4) == np.round(alpha_val, 4))
        if np.round(alpha_val, 4) == 90.0:
            mirror_mask = PA_mask # 90 degrees mirrors to itself
        else:
            mirror_mask = (np.round(pitch_angles, 4) == np.round(180 - alpha_val, 4))

        # Combine masks to get all data points for this symmetric pair
        combined_alpha_mask = PA_mask | mirror_mask

        # If there are any data points for this symmetric pair
        if np.sum(combined_alpha_mask) > 0:
            # Extract flux values for the current symmetric pitch angle pair
            # Shape: (time, num_matched_raw_pitch_angles, energy)
            flux_values_for_pair = FEDU[:, combined_alpha_mask, :]
            averaged_flux_slice = np.nanmean(flux_values_for_pair, axis=1)
            FEDU_averaged[:, pitch_angle_idx_unique, :] = averaged_flux_slice
        # else: if no data for this unique alpha, it remains NaN (from initialization)

    sat_data['FEDU_averaged'] = FEDU_averaged
    sat_data['Pitch_Angles'] = alpha_unique # Update Pitch_Angles to be the unique, averaged ones
    print(f'Pitch Angle Fluxes Averaged')
    return sat_data


#%% Interpolated Flux v Pitch Angle using exponential between points
def Interp_Flux(sat_data, alphaofK, energyofMuAlpha):
    FEDU_averaged = sat_data['FEDU_averaged']
    alpha_unique = sat_data['Pitch_Angles']
    alpha_set = alphaofK
    energy_channels = sat_data['Energy_Channels']

    K_set = np.array(list(alphaofK.columns.tolist()), dtype=float)
    K_set = np.atleast_1d(K_set)

    Mu_set = np.array(list(energyofMuAlpha[K_set[0]].columns.tolist()), dtype=float)
    Mu_set = np.atleast_1d(Mu_set)

    FEDU_interp_alpha = {}
    FEDU_interp = {}

    
    for i_K, K in enumerate(K_set):
        FEDU_interp_alpha[K] = np.zeros((len(sat_data['Epoch']), len(energy_channels)))
        FEDU_interp[K] = np.zeros((len(sat_data['Epoch']), len(Mu_set)))
        # Iterate through each time point.
        for time_index in range(len(sat_data['Epoch'])):
        #--- Phase 1: Interpolate over Pitch Angle ---#
            # Iterate through each energy channel
            for energy_index in range(FEDU_averaged.shape[2]):
                # Filter NaN and zero values
                PA_mask = np.where((FEDU_averaged[time_index, :, energy_index] > 0) &
                            (~np.isnan(FEDU_averaged[time_index, :, energy_index])))[0]

                # Check if there are enough valid data points for interpolation.
                if len(PA_mask) > 1:
                    PA_valid = alpha_unique[PA_mask]
                    # Find where the target alpha would be inserted in the sorted valid alphas
                    insertion_point = np.searchsorted(PA_valid, alpha_set[K][time_index])
                    # If the target alpha would be interpolated (within the range of valid alphas)
                    if 0 < insertion_point < len(PA_valid):
                        # Perform exponential interpolation
                        log_flux_interp = np.interp(
                            alpha_set[K][time_index],
                            PA_valid,
                            np.log(FEDU_averaged[time_index, PA_mask, energy_index])
                        )
                        FEDU_interp_alpha[K][time_index, energy_index] = np.exp(log_flux_interp)
                    # If the target alpha would be extrapolated (outside the range of valid alphas), replace with NaN
                    else:
                        FEDU_interp_alpha[K][time_index, energy_index] = np.nan
                # If there are not enough points for interpolation, set results as NaN
                else:
                    # Store NaN if there are not enough valid data points for interpolation.
                    FEDU_interp_alpha[K][time_index, energy_index] = np.nan
            
        #--- Phase 2: Interpolate over Energy ---#
            energy_set = energyofMuAlpha[K].values
            epoch_list = energyofMuAlpha[K].index.tolist()
            for i_Mu, Mu in enumerate(Mu_set):
                    energy_mask = np.where((FEDU_interp_alpha[K][time_index, :] > 0) &
                            (~np.isnan(FEDU_interp_alpha[K][time_index, :])))[0]
                    # Check if there are at least two valid data points for interpolation.
                    if len(energy_mask) > 1:
                        # Extract the valid energy channel values corresponding to the valid flux points.
                        valid_energies = energy_channels[energy_mask]
                        # Find the index where the target energy would be inserted
                        # to maintain the sorted order of valid energies.
                        insertion_point = np.searchsorted(valid_energies, energy_set[time_index, i_Mu])

                        # Check if the target energy falls within the range of valid energies (not extrapolation).
                        if 0 < insertion_point < len(valid_energies):
                            # Interpolate the logarithm of the flux.
                            log_flux_interp = np.interp(
                                energy_set[time_index, i_Mu],
                                valid_energies,
                                np.log(FEDU_interp_alpha[K][time_index, energy_mask])
                            )
                            # Exponentiate the result to get the interpolated flux.
                            FEDU_interp[K][time_index, i_Mu] = np.exp(log_flux_interp)
                        # If the target energy is outside the range of valid energies, set to NaN.
                        else:
                            FEDU_interp[K][time_index, i_Mu] = np.nan
                    # If there are not enough valid data points, set the result to NaN.
                    else:
                        # Store NaN if there are fewer than two valid data points.
                        FEDU_interp[K][time_index, i_Mu] = np.nan
        FEDU_interp[K] = pd.DataFrame(FEDU_interp[K], index=epoch_list, columns=Mu_set)
    return FEDU_interp, FEDU_interp_alpha