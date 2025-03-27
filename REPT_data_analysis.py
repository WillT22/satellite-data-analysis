#%% Import and Initialize
import numpy as np
import os
import glob
import scipy.constants as sc
# Time conversion & IRBEM
from spacepy.time import Ticktock
from spacepy.coordinates import Coords
import spacepy.irbempy as irbem
# Plotting
from datetime import datetime, timedelta
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors

# Import functions
from analysis_functions import process_l3_data
from analysis_functions import process_ephem_data
from analysis_functions import interpolate_Ephem
from analysis_functions import get_Omni
from analysis_functions import extend_alpha
from analysis_functions import find_alpha
from analysis_functions import energy_from_mu_alpha
from analysis_functions import average_fluxes_by_pitch_angle
from analysis_functions import interpolate_flux_by_energy
from analysis_functions import log_func
from analysis_functions import fit_fluxvalpha_log

# Import the latest version of OMNI data
#from spacepy import toolbox as tb
#tb.update(omni2=True)

# Initialize global variables
textsize = 16
Re = 6378.137 #Earth's Radius
Mu_set = 4000 # MeV/G
K_set = 0.10 # R_E*G^(1/2)

# Conversions
# electron mass in MeV is (m_e [kg] * c^2 [m^2/s^2]) [J] / (sc.eV [J/eV] * 10^6 [eV/MeV])
electron_E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6)
# B_local, B_min, B_mirr are in nT: 1 nT = 10^-5 G

#%% Start main class
if __name__ == '__main__':
#%% Folder containing CDF files
    folder_path_l2 = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/l2/"
    if not os.path.exists(folder_path_l2):
        raise FileNotFoundError(f"Error: Folder path not found: {folder_path_l2}")
    folder_path_l3 = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/l3/"
    if not os.path.exists(folder_path_l3):
        raise FileNotFoundError(f"Error: Folder path not found: {folder_path_l3}")
    
    ephemeris_path = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/ephemeris/"
    if not os.path.exists(ephemeris_path):
        raise FileNotFoundError(f"Error: Ephemeris path not found: {ephemeris_path}")
    
    # Get all CDF file paths in the folder
    file_paths_l2_A = glob.glob(folder_path_l2 + "rbspa*[!r]*.cdf") 
    file_paths_l2_B = glob.glob(folder_path_l2 + "rbspb*[!r]*.cdf") 
    file_paths_l3_A = glob.glob(folder_path_l3 + "rbspa*[!r]*.cdf") 
    file_paths_l3_B = glob.glob(folder_path_l3 + "rbspb*[!r]*.cdf") 
    ephem_file_paths_A = glob.glob(ephemeris_path + "rbsp-a*[!r]*.cdf")
    ephem_file_paths_B = glob.glob(ephemeris_path + "rbsp-b*[!r]*.cdf")
    
    
#%% Function for reading in RBSP flux data
    # Read in data from RBSP CDF files
    print("Processing Flux Data:")
    Epoch_A, L_A, Position_A, FEDU_A, energy_channels_A, alpha_A = process_l3_data(file_paths_l3_A)
    FEDU_A = np.where(FEDU_A == -1e+31, 0, FEDU_A)
    Epoch_B, L_B, Position_B, FEDU_B, energy_channels_B, alpha_B = process_l3_data(file_paths_l3_B)
    FEDU_B = np.where(FEDU_B == -1e+31, 0, FEDU_B)
    
    # Handle cases where only A or B data is present (check which lists are not empty)
    if not Epoch_A and not L_A and not FEDU_A:
        print("No RBSPA data found in the folder.")
    if not Epoch_B and not L_B and not FEDU_B:
        print("No RBSPB data found in the folder.")
        
    # Find the earliest and latest Epoch values
    if Epoch_A and Epoch_B: 
        min_epoch = min(min(Epoch_A), min(Epoch_B))
        max_epoch = max(max(Epoch_A), max(Epoch_B))
    else:
        # Handle cases where either Epoch_A or Epoch_B is empty
        if Epoch_A:
            min_epoch = min(Epoch_A)
            max_epoch = max(Epoch_A)
        elif Epoch_B:
            min_epoch = min(Epoch_B)
            max_epoch = max(Epoch_B)
    
#%% Read ephemeris data and interpolate over satellite times
    # Read in data from RBSP Ephemeris files
    print("Processing Ephemeris Data:")
    Epoch_ephem_A, alpha_ephem_A, Lm_ephem_A, Lstar_ephem_A, K_ephem_A = process_ephem_data(ephem_file_paths_A)
    Epoch_ephem_B, alpha_ephem_B, Lm_ephem_B, Lstar_ephem_B, K_ephem_B = process_ephem_data(ephem_file_paths_B)
   
    # Interpolate Ephemeris data for RBSP times
    print("Interpolating Ephemeris Data")
    Lm_interp_A, Lstar_interp_A, K_interp_A = interpolate_Ephem(Epoch_A, Epoch_ephem_A, Lm_ephem_A, Lstar_ephem_A, K_ephem_A)
    Lm_interp_B, Lstar_interp_B, K_interp_B = interpolate_Ephem(Epoch_B, Epoch_ephem_B, Lm_ephem_B, Lstar_ephem_B, K_ephem_B)
    
#%% Load Computationally Intensive Saved Data
    
    print("Loading Saved Data")
    loaded_data = np.load('vital_data.npz', allow_pickle=True)
    
    # Access the loaded variables
    results_A = loaded_data['results_A'].item()
    results_B = loaded_data['results_B'].item()
    

#%% Obtain Omni Information & prepare for calculating K and L*
    # Set up for IRBEM Calculations
    time_A = Ticktock(Epoch_A, 'UTC')
    time_B = Ticktock(Epoch_B, 'UTC')
    position_A = Coords(Position_A, 'GEO', 'car')
    position_B = Coords(Position_B, 'GEO', 'car')
    extMag = 'T89' # Set magnetic field model
    # Set Omni values from downloaded file
    omnivals_refined_A = get_Omni(Epoch_A, Position_A)
    omnivals_refined_B = get_Omni(Epoch_B, Position_B)
    
#%% Calculate the first adiabatic invariant
    # Calculate the first adiabatic invariant for RBSP-A
    print("Calculating Mu (RBSP-A)")
    # Extend alpha to have a better resolution
    alpha_A_extend = extend_alpha(alpha_A)
    # Calculate the B field from IRBEM
    B_A = irbem.get_Bfield(time_A, position_A, extMag=extMag, omnivals=omnivals_refined_A)
    # Separate dictionary into separate variables
    Blocal_A, Bvec_A = B_A["Blocal"], B_A["Bvec"]
    # Create a 3D grid of energy channel nominal energy, pitch angle, and local magnetic field for calculations in mu
    energy_grid, alpha_grid, blocal_grid = np.meshgrid(energy_channels_A, np.deg2rad(alpha_A_extend), Blocal_A*1e-5, indexing='ij')
    # Calculate first adiabatic invariant: [energy channels, pitch angles, time points]
    # At each time point, mu depends on particle energy and pitch angle
    Mu_A = (energy_grid**2 + 2 * energy_grid * electron_E0) * np.sin(alpha_grid)**2 / (2 * electron_E0 * blocal_grid)
    
    # Calculate the first adiabatic invariant for RBSP-B
    print("Calculating Mu (RBSP-B)")
    # Extend alpha to have a better resolution
    alpha_B_extend = extend_alpha(alpha_B)
    # Calculate the B field from IRBEM
    B_B = irbem.get_Bfield(time_B, position_B, extMag=extMag, omnivals=omnivals_refined_B)
    # Separate dictionary into separate variables
    Blocal_B, Bvec_B = B_B["Blocal"], B_B["Bvec"]
    # Create a 3D grid of energy channel nominal energy, pitch angle, and local magnetic field for calculations in mu
    energy_grid, alpha_grid, blocal_grid = np.meshgrid(energy_channels_B, np.deg2rad(alpha_B_extend), Blocal_B*1e-5, indexing='ij')
    # Calculate first adiabatic invariant: [energy channels, pitch angles, time points]
    # At each time point, mu depends on particle energy and pitch angle
    Mu_B = (energy_grid**2 + 2 * energy_grid * electron_E0) * np.sin(alpha_grid)**2 / (2 * electron_E0 * blocal_grid)

#%% Calculate L*
    # Calculate L* for RBSP-A    
    print("Calculating L* (RBSP-A)")
    # Use IRBEM get_Lstar function  ***COMPUTATIONALLY EXPENSIVE***
    #results_A = irbem.get_Lstar(time_A, position_A, alpha=alpha_A_extend, extMag=extMag, omnivals=omnivals_refined_A)
    # Separate dictionary ino variables
    Bmin_A, Bmirr_A, Lm_A, Lstar_A, MLT_A, Xj_A = results_A["Bmin"], results_A["Bmirr"], results_A["Lm"], results_A["Lstar"], results_A["MLT"], results_A["Xj"]
    Lstar_A[Lstar_A == -np.inf] = np.nan
    
    # Calculate L* for RBSP-B  
    print("Calculating L* (RBSP-B)")
    # Use IRBEM get_Lstar function  ***COMPUTATIONALLY EXPENSIVE***
    #results_B = irbem.get_Lstar(time_B, position_B, alpha=alpha_B_extend, extMag=extMag, omnivals=omnivals_refined_B)
    # Separate dictionary ino variables
    Bmin_B, Bmirr_B, Lm_B, Lstar_B, MLT_B, Xj_B = results_B["Bmin"], results_B["Bmirr"], results_B["Lm"], results_B["Lstar"], results_B["MLT"], results_B["Xj"]
    Lstar_B[Lstar_B == -np.inf] = np.nan
    
#%% Save chosen data
    '''
    print("Saving Data")
    # Create a dictionary to store the variables
    data_to_save = {
        'results_A': results_A,
        'results_B': results_B,
    }
    # Save the dictionary to a .npz file (NumPy zip archive)
    np.savez('vital_data.npz', **data_to_save)
    '''
#%% Calculate K
    # Calculate K from X_j: K = X_j * \sqrt(B_mirr)
    print("Calculating K (RBSP-A)")
    K_A = Xj_A * np.sqrt(Bmirr_A*1e-5) # R_E*G^(1/2)
    # Find alpha at each time point given a set K
    alpha_A_set = find_alpha(K_set, K_A, alpha_A_extend) 

    # Calculate K from X_j: K = X_j * \sqrt(B_mirr)
    print("Calculating K (RBSP-B)")
    K_B = Xj_B * np.sqrt(Bmirr_B*1e-5) # R_E*G^(1/2)
    # Find alpha at each time point given a set K
    alpha_B_set = find_alpha(K_set, K_B, alpha_B_extend) 
    
#%% Find L* for calculated alpha values from set K
    print("Calculating L* for set K values (RBSP-A)")
    Lstar_A_set = np.zeros(len(alpha_A_set))
    for time_index in range(len(alpha_A_set)):    
        Lstar_A_set[time_index] = np.interp(alpha_A_set[time_index], alpha_A_extend, Lstar_A[time_index,:])
    
    print("Calculating L* for set K values (RBSP-B)")
    Lstar_B_set = np.zeros(len(alpha_B_set))
    for time_index in range(len(alpha_B_set)):    
        Lstar_B_set[time_index] = np.interp(alpha_B_set[time_index], alpha_B_extend, Lstar_B[time_index,:])

#%% Find energy for a given mu at each time point
    # Find kinetic energy of particle population with a given Mu and alpha calculated from a given K 
    energy_A_set = energy_from_mu_alpha(Mu_set, alpha_A_set, Blocal_A)
    # Find kinetic energy of particle population with a given Mu and alpha calculated from a given K 
    energy_B_set = energy_from_mu_alpha(Mu_set, alpha_B_set, Blocal_B)
    
#%% Interpolate flux for each energy given a mu and K
    print("Averaging fluxes with the same pitch angle (RBSP-A)")
    FEDU_A_averaged = average_fluxes_by_pitch_angle(FEDU_A, alpha_A, energy_channels_A)
    print("Averaging fluxes with the same pitch angle (RBSP-B)")
    FEDU_B_averaged = average_fluxes_by_pitch_angle(FEDU_B, alpha_B, energy_channels_B)
    
    print("Interpolating Flux over Energy (RBSP-A)")
    FEDU_A_interpE = interpolate_flux_by_energy(FEDU_A_averaged, alpha_A, energy_channels_A, energy_A_set)
    print("Interpolating Flux over Energy (RBSP-B)")
    FEDU_B_interpE = interpolate_flux_by_energy(FEDU_B_averaged, alpha_B, energy_channels_B, energy_B_set)
    
    
    ## COMPUTATIIONALLY EXPENSIVE ##
    '''
    print("Finding Log Fit of Flux over Pitch Angle (RBSP-A)")
    FEDU_A_interpE_logfit = fit_fluxvalpha_log(FEDU_A_averaged, alpha_A, energy_channels_A)
    print("Finding Log Fit of Flux over Pitch Angle (RBSP-B)")
    FEDU_B_interpE_logfit = fit_fluxvalpha_log(FEDU_A_averaged, alpha_B, energy_channels_B)
    '''
    
    '''
    print("Saving Log Fit")
    # Create a dictionary to store the variables
    data_to_save = {
        'FEDU_A_interpE_logfit': FEDU_A_interpE_logfit,
        'FEDU_B_interpE_logfit': FEDU_B_interpE_logfit,
    }
    # Save the dictionary to a .npz file (NumPy zip archive)
    np.savez('log_fit.npz', **data_to_save)
    '''
    
    print("Loading Saved Data")
    loaded_data = np.load('log_fit.npz', allow_pickle=True)
    
    # Access the loaded variables
    FEDU_A_interpE_logfit = loaded_data['FEDU_A_interpE_logfit']
    FEDU_A_interpE_logfit = loaded_data['FEDU_A_interpE_logfit']
    
    #print("Interpolating Flux over Pitch Angle (RBSP-A)")
    #print("Interpolating Flux over Pitch Angle (RBSP-B)")
    
    
#%% Plots
    '''
    #%% How flux and E_K relate for set Mu at given times
    time_index = 20000
    colors = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(energy_channels_A[:-4]), dtype=int)]
    rounded_alphas = np.round(alpha_A, 4)
    unique_alphas = sorted(list(set(rounded_alphas)))
    handles, labels = [], []
    fig, (axA) = plt.subplots(figsize=(16, 4))
    
    for energy_index in range(len(energy_channels_A[:-4])):
        scatter = axA.scatter(unique_alphas, FEDU_A_averaged[time_index, :, energy_index], s=50,
                            color=colors[energy_index], label=f"Energy Channel {energy_index}")
        handles.append(scatter)
        labels.append(f"{energy_channels_A[energy_index]:.1f}") # Changed label here
    
    axA.set_title("RBSP-A", fontsize=textsize)
    axA.set_ylabel("Electron Flux (#/cm²/s/sr/MeV)", fontsize=textsize)
    axA.tick_params(axis='both', labelsize=textsize)
    axA.set_xlabel("Pitch Angle", fontsize=textsize)
    #axA.set_yscale('log')
    
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=textsize, markerscale=2,
                title="Energy (MeV)", title_fontsize=textsize)
    fig.suptitle(f"Time = {Epoch_A[time_index]}", fontsize=textsize + 2)
    plt.subplots_adjust(top=0.85, right=0.9)
    plt.show()
    
    

    time_index = 20000
    colors = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(energy_channels_A[:-4]), dtype=int)]
    rounded_alphas = np.round(alpha_A, 4)
    unique_alphas = np.array(sorted(list(set(rounded_alphas))))
    handles, labels = [], []
    fig, (axA) = plt.subplots(figsize=(16, 4))
    positive_y_indices = np.where(FEDU_A_interpE[time_index, :] > 0)[0]
    for energy_index in range(len(energy_channels_A[:-4])):
        scatter = axA.scatter(unique_alphas[positive_y_indices], FEDU_A_averaged[time_index, positive_y_indices, energy_index], 
                              color=colors[energy_index], label=f"Energy Channel {energy_index}")
        handles.append(scatter)
        labels.append(f"{energy_channels_A[energy_index]:.1f}") # Changed label here
        try:
            alpha_smooth = np.linspace(0, 90, 500)  # Adjust 500 for more/less smoothness
            log_fit = log_func(alpha_smooth, *FEDU_A_interpE_logfit[time_index, energy_index, :])
            # Filter out y-values <= 0 for log fit plot
            positive_log_fit_indices = np.where(log_fit > 0)[0]
            axA.plot(alpha_smooth[positive_log_fit_indices], log_fit[positive_log_fit_indices],
                 color=colors[energy_index], linestyle='--')
        except (TypeError, ValueError) as e:
                print(f"Logarithmic fit failed for time index {time_index}: {e}")
    axA.set_title("RBSP-A", fontsize=textsize)
    axA.set_ylabel("Electron Flux (#/cm²/s/sr/MeV)", fontsize=textsize)
    axA.tick_params(axis='both', labelsize=textsize)
    axA.set_xlabel("Pitch Angle", fontsize=textsize)
    axA.set_yscale('log')
    axA.grid(True)
      
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=textsize, markerscale=2,
                title="Energy (MeV)", title_fontsize=textsize)
    fig.suptitle(f"Time = {Epoch_A[time_index]}", fontsize=textsize + 2)
    plt.subplots_adjust(top=0.85, right=0.85)
    plt.show()
    
    
    
    
    time_index = 20000
    colors = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(energy_channels_A[:-4]), dtype=int)]
    rounded_alphas = np.round(alpha_A, 4)
    unique_alphas = np.array(sorted(list(set(rounded_alphas))))
    fig, (axA) = plt.subplots(figsize=(16, 4))
    
    positive_y_indices = np.where(FEDU_A_interpE[time_index, :] > 0)[0]
    
    scatter = axA.scatter(unique_alphas[positive_y_indices], FEDU_A_interpE[time_index, positive_y_indices], s=50)
    try:
        alpha_smooth = np.linspace(0, 90, 500)  # Adjust 500 for more/less smoothness
        log_fit = log_func(alpha_smooth, *FEDU_A_interpE_logfit[time_index, :])
        # Filter out y-values <= 0 for log fit plot
        positive_log_fit_indices = np.where(log_fit > 0)[0]
        axA.plot(alpha_smooth[positive_log_fit_indices], log_fit[positive_log_fit_indices], color='red', linestyle='--', label='Log Fit')
    except (TypeError, ValueError) as e:
            print(f"Logarithmic fit failed for time index {time_index}: {e}")
    axA.set_title("RBSP-A", fontsize=textsize)
    axA.set_ylabel("Electron Flux (#/cm²/s/sr/MeV)", fontsize=textsize)
    axA.tick_params(axis='both', labelsize=textsize)
    axA.set_xlabel("Pitch Angle", fontsize=textsize)
    axA.set_ylim(max(min(FEDU_A_interpE[time_index, positive_y_indices])-min(FEDU_A_interpE[time_index, positive_y_indices])/2,0), 
                 max(FEDU_A_interpE[time_index, positive_y_indices])+max(FEDU_A_interpE[time_index, positive_y_indices])/4)
    axA.set_yscale('log')
    axA.grid(True)
    
    axA.text(0.05, 0.95, f"Energy: {EnergyofMuAlpha_A[time_index]:.2f} MeV", transform=axA.transAxes, fontsize=textsize, verticalalignment='top')
    
    fig.suptitle(f"Time = {Epoch_A[time_index]}", fontsize=textsize + 2)
    plt.subplots_adjust(top=0.85, right=0.9)
    plt.show()
     
    
    
    # Plot L* v time and add specific time points
    handles, labels = [], []
    fig, (axA, axB) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(16, 6))
    
    # Convert Epoch_A and Epoch_B to NumPy arrays of datetimes
    Epoch_A_np = np.array(Epoch_A)
    Epoch_B_np = np.array(Epoch_B)
    
    # Filter data for Lstar > 2
    mask_A = Lstar_A_set > 2
    Epoch_A_filtered = Epoch_A_np[mask_A]
    Lstar_A_set_filtered = Lstar_A_set[mask_A]
    
    mask_B = Lstar_B_set > 2
    Epoch_B_filtered = Epoch_B_np[mask_B]
    Lstar_B_set_filtered = Lstar_B_set[mask_B]
    
    scatter_A = axA.scatter(Epoch_A_filtered, Lstar_A_set_filtered)
    scatter_B = axB.scatter(Epoch_B_filtered, Lstar_B_set_filtered)
    
    axA.set_title("RBSP-A", fontsize=textsize)
    axA.set_ylabel(r"L* ($R_E$)", fontsize=textsize)
    axA.tick_params(axis='both', labelsize=textsize)
    axA.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    axA.grid(True)
    
    axB.set_ylabel(r"L* ($R_E$)", fontsize=textsize)
    axB.set_title("RBSP-B", fontsize=textsize)
    axB.tick_params(axis='both', labelsize=textsize)
    axB.grid(True)
    axB.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    axB.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=6))
    axB.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    
    # Define time stamps for star markers
    star_time_stamps_A = [500, 5000, 20000]  # Example time stamps
    # Plot star markers
    for time_stamp in star_time_stamps_A:
        axA.scatter(Epoch_A[time_stamp], Lstar_A_set[time_stamp], marker='*', s=500, color='black')
        axB.scatter(Epoch_B[time_stamp], Lstar_B_set[time_stamp], marker='*', s=500, color='black')  # Adjust size and color as needed
    
    fig.suptitle(f"Kinetic Energy of Electrons with K={K_set:.1f}", fontsize=textsize + 2)
    plt.xticks(rotation=45, ha='right', fontsize=textsize)
    plt.subplots_adjust(top=0.88, right=0.95)
    plt.show()
    '''
    
    #%% Plot L* v time with interpolated flux as colorbar
    '''
    handles, labels = [], []
    fig, (axA, axB) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(16, 6))
    
    # Convert Epoch_A and Epoch_B to NumPy arrays of datetimes
    Epoch_A_np = np.array(Epoch_A)
    Epoch_B_np = np.array(Epoch_B)
    
    # Filter data for Lstar > 2
    mask_A = Lstar_A_set > 2
    Epoch_A_filtered = Epoch_A_np[mask_A]
    Lstar_A_set_filtered = Lstar_A_set[mask_A]
    FEDU_A_interpaE_filtered = FEDU_A_interpaE[mask_A]
    
    mask_B = Lstar_B_set > 2
    Epoch_B_filtered = Epoch_B_np[mask_B]
    Lstar_B_set_filtered = Lstar_B_set[mask_B]
    energy_B_set_filtered = energy_B_set[mask_B]
    
    # Linear colorbar set up
    min_val = min(np.nanmin(FEDU_A_interpaE_filtered), np.nanmin(FEDU_A_interpaE_filtered))
    max_val = max(np.nanmax(FEDU_A_interpaE_filtered), np.nanmax(FEDU_A_interpaE_filtered))
    
    scatter_A = axA.scatter(Epoch_A_filtered, Lstar_A_set_filtered, c=FEDU_A_interpaE_filtered, vmin=min_val, vmax=max_val)
    scatter_B = axB.scatter(Epoch_B_filtered, Lstar_B_set_filtered, c=FEDU_A_interpaE_filtered, vmin=min_val, vmax=max_val)
    
    axA.set_title("RBSP-A", fontsize=textsize)
    axA.set_ylabel(r"L* ($R_E$)", fontsize=textsize)
    axA.tick_params(axis='both', labelsize=textsize)
    axA.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    axA.grid(True)
    
    axB.set_ylabel(r"L* ($R_E$)", fontsize=textsize)
    axB.set_title("RBSP-B", fontsize=textsize)
    axB.tick_params(axis='both', labelsize=textsize)
    axB.grid(True)
    axB.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    axB.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=6))
    axB.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    
    cbar_A = fig.colorbar(scatter_A, ax=axA, fraction=0.03, pad=0.01)
    cbar_A.set_label(r"$E_K$ (MeV)", fontsize=textsize)
    cbar_A.ax.tick_params(labelsize=textsize)
    cbar_B = fig.colorbar(scatter_B, ax=axB, fraction=0.03, pad=0.01)
    cbar_B.set_label(r"$E_K$ (MeV)", fontsize=textsize)
    cbar_B.ax.tick_params(labelsize=textsize)
    # Set colorbar tick locations to every 2 units within current limits
    ticks = np.arange(2, 14, 2)
    cbar_A.set_ticks(ticks)
    cbar_B.set_ticks(ticks)
    
    # Define time stamps for star markers
    star_time_stamps_A = [500, 5000, 20000]  # Example time stamps
    # Plot star markers
    for time_stamp in star_time_stamps_A:
        axA.scatter(Epoch_A[time_stamp], Lstar_A_set[time_stamp], marker='*', s=500, color='magenta')
        axB.scatter(Epoch_B[time_stamp], Lstar_B_set[time_stamp], marker='*', s=500, color='magenta')  # Adjust size and color as needed
    
    fig.suptitle(f"Kinetic Energy of Electrons with K={K_set:.1f}", fontsize=textsize + 2)
    plt.xticks(rotation=45, ha='right', fontsize=textsize)
    plt.subplots_adjust(top=0.88, right=0.95)
    plt.show()
    '''
    
    
    #%% Plot L* v time with electron kinetic energy as colorbar
    '''
    handles, labels = [], []
    fig, (axA, axB) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(16, 6))
    
    # Convert Epoch_A and Epoch_B to NumPy arrays of datetimes
    Epoch_A_np = np.array(Epoch_A)
    Epoch_B_np = np.array(Epoch_B)
    
    # Filter data for Lstar > 2
    mask_A = Lstar_A_set > 2
    Epoch_A_filtered = Epoch_A_np[mask_A]
    Lstar_A_set_filtered = Lstar_A_set[mask_A]
    energy_A_set_filtered = energy_A_set[mask_A]
    
    mask_B = Lstar_B_set > 2
    Epoch_B_filtered = Epoch_B_np[mask_B]
    Lstar_B_set_filtered = Lstar_B_set[mask_B]
    energy_B_set_filtered = energy_B_set[mask_B]
    
    # Linear colorbar set up
    min_val = min(np.nanmin(energy_A_set_filtered), np.nanmin(energy_B_set_filtered))
    max_val = max(np.nanmax(energy_A_set_filtered), np.nanmax(energy_B_set_filtered))
    
    scatter_A = axA.scatter(Epoch_A_filtered, Lstar_A_set_filtered, c=energy_A_set_filtered, vmin=min_val, vmax=max_val)
    scatter_B = axB.scatter(Epoch_B_filtered, Lstar_B_set_filtered, c=energy_B_set_filtered, vmin=min_val, vmax=max_val)
    
    axA.set_title("RBSP-A", fontsize=textsize)
    axA.set_ylabel(r"L* ($R_E$)", fontsize=textsize)
    axA.tick_params(axis='both', labelsize=textsize)
    axA.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    axA.grid(True)
    
    axB.set_ylabel(r"L* ($R_E$)", fontsize=textsize)
    axB.set_title("RBSP-B", fontsize=textsize)
    axB.tick_params(axis='both', labelsize=textsize)
    axB.grid(True)
    axB.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    axB.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=6))
    axB.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    
    cbar_A = fig.colorbar(scatter_A, ax=axA, fraction=0.03, pad=0.01)
    cbar_A.set_label(r"$E_K$ (MeV)", fontsize=textsize)
    cbar_A.ax.tick_params(labelsize=textsize)
    cbar_B = fig.colorbar(scatter_B, ax=axB, fraction=0.03, pad=0.01)
    cbar_B.set_label(r"$E_K$ (MeV)", fontsize=textsize)
    cbar_B.ax.tick_params(labelsize=textsize)
    # Set colorbar tick locations to every 2 units within current limits
    ticks = np.arange(2, 14, 2)
    cbar_A.set_ticks(ticks)
    cbar_B.set_ticks(ticks)
    
    # Define time stamps for star markers
    star_time_stamps_A = [500, 5000, 20000]  # Example time stamps
    # Plot star markers
    for time_stamp in star_time_stamps_A:
        axA.scatter(Epoch_A[time_stamp], Lstar_A_set[time_stamp], marker='*', s=500, color='magenta')
        axB.scatter(Epoch_B[time_stamp], Lstar_B_set[time_stamp], marker='*', s=500, color='magenta')  # Adjust size and color as needed
    
    fig.suptitle(f"Kinetic Energy of Electrons with K={K_set:.1f}", fontsize=textsize + 2)
    plt.xticks(rotation=45, ha='right', fontsize=textsize)
    plt.subplots_adjust(top=0.88, right=0.95)
    plt.show()
    '''