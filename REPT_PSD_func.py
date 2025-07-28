import numpy as np
import os
import glob
from spacepy import pycdf
from spacepy.time import Ticktock
from spacepy.coordinates import Coords
import datetime as dt
import matplotlib.dates as mdates
import scipy.constants as sc

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


#%% Extract Magentometer Data and match with nearest time point
def find_mag(sat_data, sat_name):
    mag_folder = '/home/will/REPT_data/MagData/'

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
    print('Magnetic Field Data Extracted \n')
    return sat_data


#%% Find the averages of fluzes with the same pitch angle
def average_fluxes_by_pitch_angle(sat_data, sat_name):
    """
    Averages fluxes for matching pitch angles in FEDU.

    Args:
        FEDU (numpy.ndarray): 3D array of fluxes (time, pitch angle, energy).
        alpha (list or numpy.ndarray): List or array of pitch angle values.
        energy_channels (list or numpy.ndarray): List or array of energy channel values.

    Returns:
        numpy.ndarray: Averaged fluxes array (time, unique pitch angle, energy).
    """
    FEDU = sat_data['FEDU']
    Energy_Channels = sat_data['Energy_Channels']
    alpha_unique = np.array(sorted(list(set(np.round(sat_data['Pitch_Angles'][sat_data['Pitch_Angles']<=90], 4)))))

    FEDU_averaged = np.zeros((FEDU.shape[0], len(alpha_unique), len(Energy_Channels)))

    for time_index in range(FEDU.shape[0]):
        for energy_index in range(len(Energy_Channels)):
            for pitch_angle_index in range(len(alpha_unique)):
                mirrored_pitch_angle_index = 16 - pitch_angle_index
                
                value1 = FEDU[time_index, pitch_angle_index, energy_index]
                value2 = FEDU[time_index, mirrored_pitch_angle_index, energy_index]

                is_nan1 = np.isnan(value1)
                is_nan2 = np.isnan(value2)

                if is_nan1 and is_nan2:
                    FEDU_averaged[time_index, pitch_angle_index, energy_index] = np.nan
                else:
                    FEDU_averaged[time_index, pitch_angle_index, energy_index] = np.nanmean([value1, value2])
    sat_data['FEDU_averaged'] = FEDU_averaged
    sat_data['Pitch_Angles'] = alpha_unique
    return sat_data


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
        kinetic_energy[:, i] = np.sqrt(2 * electron_E0 * mu * B_local / sin_squared_alpha + electron_E0**2) - electron_E0

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