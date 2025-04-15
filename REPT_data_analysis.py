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
import matplotlib.colors as colors

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
from analysis_functions import interpolate_flux_by_alpha
from analysis_functions import find_psd

# Import the latest version of OMNI data
#from spacepy import toolbox as tb
#tb.update(omni2=True)

# Initialize global variables
textsize = 16
Re = 6378.137 #Earth's Radius
Mu_set = np.array((4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G
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

#%% Load Computationally Intensive Saved Data
    
    print("Loading Saved Data")
    loaded_data = np.load('vital_data.npz', allow_pickle=True)
    
    # Access the loaded variables
    results_A = loaded_data['results_A'].item()
    results_B = loaded_data['results_B'].item()

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
    
#%% Average fluxed with the same pitch angle
    print("Averaging fluxes with the same pitch angle (RBSP-A)")
    FEDU_A_averaged = average_fluxes_by_pitch_angle(FEDU_A, alpha_A, energy_channels_A)
    print("Averaging fluxes with the same pitch angle (RBSP-B)")
    FEDU_B_averaged = average_fluxes_by_pitch_angle(FEDU_B, alpha_B, energy_channels_B)

#%% Interpolate flux for each pitch angle given a mu and K  
    print("Interpolating Flux over Pitch Angle (RBSP-A)")
    FEDU_A_interpa = interpolate_flux_by_alpha(FEDU_A_averaged, alpha_A, alpha_A_set)
    print("Interpolating Flux over Pitch Angle (RBSP-B)")
    FEDU_B_interpa = interpolate_flux_by_alpha(FEDU_B_averaged, alpha_B, alpha_B_set)
    
#%% Interpolate flux for each energy given a mu and K
    print("Interpolating Flux over Energy (RBSP-A)")
    FEDU_A_interpaE = interpolate_flux_by_energy(FEDU_A_interpa, energy_channels_A, energy_A_set)
    print("Interpolating Flux over Energy (RBSP-B)")
    FEDU_B_interpaE = interpolate_flux_by_energy(FEDU_B_interpa, energy_channels_B, energy_B_set)
    
#%% Calculate PSD from flux and energy for a given mu and K
    print("Calculating PSD (RBSP-A)")    
    psd_A = find_psd(FEDU_A_interpaE, energy_A_set)
    print("Calculating PSD (RBSP-B)") 
    psd_B = find_psd(FEDU_B_interpaE, energy_B_set)

#%% Plot PSD
    fig, ax = plt.subplots(figsize=(16, 4))
    
    mu_select = 6
    
    # Convert Epoch_A and Epoch_B to NumPy arrays of datetimes
    Epoch_A_np = np.array(Epoch_A)
    Epoch_B_np = np.array(Epoch_B)
    
    colorscheme = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))
    cmap = colors.ListedColormap(colorscheme)
    
    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1e-12))
    max_val = np.nanmax(np.log10(1e-9))
    
    psd_A_plot = psd_A.copy()
    psd_A_plot[np.where(psd_A_plot == 0)] = 1e-12
    psd_A_plot[np.isnan(psd_A_plot)] = 1e-12
    psd_B_plot = psd_B.copy()
    psd_B_plot[np.where(psd_B_plot == 0)] = 1e-12
    psd_B_plot[np.isnan(psd_B_plot)] = 1e-12
    
    # Plotting, ignoring NaN values in the color
    scatter_A = ax.scatter(Epoch_A_np, Lstar_A_set,
                     c=np.log10(psd_A_plot[:, mu_select]), cmap=cmap, vmin=min_val, vmax=max_val)
    scatter_B = ax.scatter(Epoch_B_np, Lstar_B_set,
                     c=np.log10(psd_B_plot[:, mu_select]), cmap=cmap, vmin=min_val, vmax=max_val)

 
    ax.set_title("RBSP-A & RBSP-B", fontsize=textsize)
    ax.set_ylabel(r"L* ($R_E$)", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    # Force labels for first and last x-axis tick marks 
    min_epoch = datetime(1970, 1, 1) + timedelta(hours=math.floor((min_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = datetime(1970, 1, 1) + timedelta(hours=math.ceil((max_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    ax.set_ylim(3, 5.5)
    ax.grid(True)
    
    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    tick_locations = np.arange(min_val, max_val + 1)
    cbar.set_ticks(tick_locations)
    cbar.set_label(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)
    
    fig.suptitle(f"Flux of Electrons with K={K_set:.1f} $G^{{1/2}}R_E$, $\\mu$={Mu_set[mu_select]:.0f} $MeV/G$", fontsize=textsize + 2)
    plt.xticks(rotation=45, ha='right', fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)
    
    plt.show()
    
    
#%% Plot PSD lineplots
    fig, ax = plt.subplots(figsize=(6, 4.5))
    color_set = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(Mu_set), dtype=int)]
    color_set[3] = [0, 1, 1, 1]  # Teal
    
    #time_start  = datetime(2017, 4, 23, 18, 45, 0)
    #time_stop   = datetime(2017, 4, 23, 22, 58, 0)
    
    #time_start  = datetime(2017, 4, 24, 17, 7, 0)
    #time_stop   = datetime(2017, 4, 24, 21, 35, 0)
    
    time_start  = datetime(2017, 4, 25, 15, 30, 0)
    time_stop   = datetime(2017, 4, 25, 19, 57, 0)
    
    # Convert Epoch_A and Epoch_B to NumPy arrays of datetimes
    Epoch_A_np = np.array(Epoch_A)
    Epoch_B_np = np.array(Epoch_B)
    
    # Define Lstar delta
    lstar_delta = 0.1
    
    # Generate Lstar interval boundaries within the time range.
    time_range = Epoch_B_np[(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
    lstar_range = Lstar_B_set[(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
    psd_range = psd_B[(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
    lstar_min = np.min(lstar_range[~np.isnan(lstar_range)])
    lstar_max = np.max(lstar_range[~np.isnan(lstar_range)])
    lstar_intervals = np.arange(np.floor(lstar_min / lstar_delta) * lstar_delta, np.ceil(lstar_max / lstar_delta) * lstar_delta + lstar_delta, lstar_delta)
    
    energy_range = energy_B_set[(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
    interpa_range = FEDU_B_interpa[(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
    interpaE_range = FEDU_B_interpaE[(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
    
    # Initialize arrays to store averaged values.
    averaged_lstar = np.zeros(len(lstar_intervals))
    averaged_psd = np.zeros((len(lstar_intervals), psd_B.shape[1]))
    non_nan_count = np.zeros((len(lstar_intervals), psd_B.shape[1]))
    
    for mu_index in range(len(Mu_set)):
        # Iterate through each Lstar interval.
        for i, lstar_val in enumerate(lstar_intervals):
            # Find indices within the current Lstar interval and time range.
            lstar_start = lstar_val - 1/2 * lstar_delta
            lstar_end = lstar_val + 1/2 * lstar_delta
            interval_indices = np.where((Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop) & (Lstar_B_set >= lstar_start) & (Lstar_B_set < lstar_end))[0]           
            
            non_nan_count[i, mu_index] = np.sum(~np.isnan(psd_B[interval_indices, mu_index]))
            # Calculate averages for the current Lstar interval
            if len(interval_indices) > 1 and np.sum(~np.isnan(psd_B[interval_indices, mu_index])) >= 20:
                averaged_psd[i, mu_index] = np.nanmean(psd_B[interval_indices, mu_index])  # average along the first axis, ignoring NaNs.
            else:
                averaged_psd[i, mu_index] = np.nan
            
        # Create a mask to filter out NaN values
        nan_mask = ~np.isnan(averaged_psd[:, mu_index])
        
        # Apply the mask to both averaged_lstar and averaged_psd
        ax.plot(
            lstar_intervals[nan_mask],
            averaged_psd[nan_mask, mu_index],
            color=color_set[mu_index],
            linewidth=2,
            marker='o',
            markersize=4,
            label=f"{Mu_set[mu_index]:.0f}"
            )
    
    ax.set_xlim(3, 5.5)
    ax.set_xlabel(r"L*", fontsize=textsize - 2)
    ax.set_ylim(1e-13, 1e-5)
    ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize - 2)
    plt.yscale('log')
    ax.grid(True)
    
    # Add legend
    ax.legend(
        title=r"$\mu$ (MeV/G)",
        loc='center right',
        bbox_to_anchor=(1.25, 0.5),
        fontsize='small', #adjust legend fontsize
        title_fontsize='medium', #adjust legend title fontsize
        markerscale=0.7,
        handlelength=1
    )

    # Add K text to the plot
    ax.text(0.02, 0.98, r"K = " + f"{K_set:.1f} $G^{{1/2}}R_E$", transform=ax.transAxes, fontsize=textsize-4, verticalalignment='top') #add the text
    
    # Set the plot title to the time interval
    title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title_str)
    
    plt.tight_layout()
    plt.show()


#%% Plots
    '''
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
    
    #%% Plot L* v time with interpolated flux from set mu, K as colorbar
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
    FEDU_B_interpaE_filtered = FEDU_B_interpaE[mask_B]
    
    #Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1))
    max_val = np.nanmax(np.log10(10**6))

    scatter_A = axA.scatter(Epoch_A_filtered, Lstar_A_set_filtered, c=np.log10(FEDU_A_interpaE_filtered), vmin=min_val, vmax=max_val)
    scatter_B = axB.scatter(Epoch_B_filtered, Lstar_B_set_filtered, c=np.log10(FEDU_B_interpaE_filtered), vmin=min_val, vmax=max_val)

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
    
    cbar_A = fig.colorbar(scatter_A, ax=axA, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    cbar_A.set_label(r"Flux ($cm^{-2} s^{-1} sr^{-1} MeV^{-1}$)", fontsize=textsize)
    cbar_A.ax.tick_params(labelsize=textsize)
    cbar_B = fig.colorbar(scatter_B, ax=axB, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    cbar_B.set_label(r"Flux ($cm^{-2} s^{-1} sr^{-1} MeV^{-1}$)", fontsize=textsize)
    cbar_B.ax.tick_params(labelsize=textsize)
    
    fig.suptitle(f"Flux of Electrons with K={K_set:.1f}, $\\mu$={Mu_set:.0f}", fontsize=textsize + 2)
    plt.xticks(rotation=45, ha='right', fontsize=textsize)
    plt.subplots_adjust(top=0.88, right=0.95)
    plt.show()
    '''
    
    '''
    fig, ax = plt.subplots(figsize=(16, 4))
    
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
    FEDU_B_interpaE_filtered = FEDU_B_interpaE[mask_B]
    
    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1))
    max_val = np.nanmax(np.log10(10**6))
    
    mu_select = 0
    
    scatter_A = ax.scatter(Epoch_A_filtered, Lstar_A_set_filtered, c=np.log10(FEDU_A_interpaE_filtered[:,mu_select]), vmin=min_val, vmax=max_val)
    scatter_B = ax.scatter(Epoch_B_filtered, Lstar_B_set_filtered, c=np.log10(FEDU_B_interpaE_filtered[:,mu_select]), vmin=min_val, vmax=max_val)
    
    ax.set_title("RBSP-A & RBSP-B", fontsize=textsize)
    ax.set_ylabel(r"L* ($R_E$)", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    # Force labels for first and last x-axis tick marks 
    min_epoch = datetime(1970, 1, 1) + timedelta(hours=math.floor((min_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = datetime(1970, 1, 1) + timedelta(hours=math.ceil((max_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    ax.grid(True)
    
    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    cbar.set_label(r"Flux ($cm^{-2} s^{-1} sr^{-1} MeV^{-1}$)", fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)
    
    fig.suptitle(f"Flux of Electrons with K={K_set:.1f}, $\\mu$={Mu_set[mu_select]:.0f}", fontsize=textsize + 2)
    plt.xticks(rotation=45, ha='right', fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)
    
    plt.show()
    '''
    
