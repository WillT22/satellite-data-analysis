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
from analysis_functions import process_flux_data
from analysis_functions import process_ephem_data
from analysis_functions import get_Omni
from analysis_functions import find_apogee_times
from analysis_functions import find_perigee_times

# Import the latest version of OMNI data
#from spacepy import toolbox as tb
#tb.update(omni2=True)

# Initialize global variables
textsize = 16
Re = 6378.137 #Earth's Radius
Mu_select = 4000 # MeV/G
K_select = 0.10 # R_E*G^(1/2)

# Conversions
# electron mass in MeV is (m_e [kg] * c^2 [m^2/s^2]) [J] / (sc.eV [J/eV] * 10^6 [eV/MeV])
electron_mass_mev = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6)
# B_local, B_min, B_mirr are in nT: 1 nT = 10^-5 G
# 1 R_E*T^(-1/2) = 0.01 R_E*G^(-1/2)

# Start main class
if __name__ == '__main__':
#%% Folder containing CDF files
    folder_path_l2 = "C:/Users/Will/Box/Multipoint_Box/REPT Data/April 2017 Storm/l2/"
    if not os.path.exists(folder_path_l2):
        raise FileNotFoundError(f"Error: Folder path not found: {folder_path_l2}")
    folder_path_l3 = "C:/Users/Will/Box/Multipoint_Box/REPT Data/April 2017 Storm/l3/"
    if not os.path.exists(folder_path_l3):
        raise FileNotFoundError(f"Error: Folder path not found: {folder_path_l3}")
    
    ephemeris_path = "C:/Users/Will/Box/Multipoint_Box/REPT Data/April 2017 Storm/ephemeris/"
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
    Epoch_A, L_A, Position_A, FEDU_A, energy_channels_A, alpha_A = process_flux_data(file_paths_l3_A)
    Epoch_B, L_B, Position_B, FEDU_B, energy_channels_B, alpha_B = process_flux_data(file_paths_l3_B)
    
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
    
    print("Interpolating Ephemeris Data:")
    Lm_interp_A, Lstar_interp_A, K_interp_A = interpolate_Ephem(Epoch_A, Epoch_ephem_A, Lm_ephem_A, Lstar_ephem_A, K_ephem_A)
    Lm_interp_B, Lstar_interp_B, K_interp_B = interpolate_Ephem(Epoch_B, Epoch_ephem_B, Lm_ephem_B, Lstar_ephem_B, K_ephem_B)
    
    
#%% Obtain Omni Information & prepare for calculating K and L*
    # Set up for IRBEM Calculations
    time_A = Ticktock(Epoch_A, 'UTC')
    time_B = Ticktock(Epoch_B, 'UTC')
    position_A = Coords(Position_A, 'GEO', 'car')
    position_B = Coords(Position_B, 'GEO', 'car')
    extMag = 'T89'
    omnivals_refined_A = get_Omni(Epoch_A, Position_A)
    omnivals_refined_B = get_Omni(Epoch_B, Position_B)
    
    print("Calculating Mu (RBSP-A)")
    B_A = irbem.get_Bfield(time_A, position_A, extMag=extMag, omnivals=omnivals_refined_A)
    Blocal_A, Bvec_A = B_A["Blocal"], B_A["Bvec"]
    energy_grid, alpha_grid, blocal_grid = np.meshgrid(energy_channels_A, np.deg2rad(alpha_A[0:9]), Blocal_A*1e-5, indexing='ij')
    Mu_A = (energy_grid**2 + 2 * energy_grid * electron_mass_mev) * np.sin(alpha_grid)**2 / (2 * electron_mass_mev * blocal_grid)
    #Mu_A = (energy_channels_A^2+2*energy_channels_A*sc.electron_mass*sc.c^2)*np.sin(alpha)/(2*sc.electron_mass*sc.c^2*Blocal_A) 
    
    print("Calculating Mu (RBSP-B)")
    B_B = irbem.get_Bfield(time_B, position_B, extMag=extMag, omnivals=omnivals_refined_B)
    Blocal_B, Bvec_B = B_B["Blocal"], B_B["Bvec"]
    energy_grid, alpha_grid, blocal_grid = np.meshgrid(energy_channels_B, np.deg2rad(alpha_B[0:9]), Blocal_B*1e-5, indexing='ij')
    Mu_B = (energy_grid**2 + 2 * energy_grid * electron_mass_mev) * np.sin(alpha_grid)**2 / (2 * electron_mass_mev * blocal_grid)


    print("Calculating L* (RBSP-A)")
    #results_A = irbem.get_Lstar(time_A, position_A, alpha=alpha_A[0:9], extMag=extMag, omnivals=omnivals_refined_A)
    Bmin_A, Bmirr_A, Lm_A, Lstar_A, MLT_A, Xj_A = results_A["Bmin"], results_A["Bmirr"], results_A["Lm"], results_A["Lstar"], results_A["MLT"], results_A["Xj"]
    if len(Bmin_A.shape) == 1:
        Bmin_A = Bmin_A.reshape(len(Bmin_A), 1)
    Bmirr_A = np.concatenate((Bmirr_A[:,:-1], Bmirr_A[:, ::-1]), axis=1)
    Lm_A    = np.concatenate((Lm_A[:,:-1], Lm_A[:, ::-1]), axis=1)
    Lstar_A = np.concatenate((Lstar_A[:,:-1], Lstar_A[:, ::-1]), axis=1)
    Xj_A    = np.concatenate((Xj_A[:,:-1], Xj_A[:, ::-1]), axis=1)
    
    # Find when Lstar is NaN, typically indicating apogee
    apogee_times_A = find_apogee_times(Lstar_A, Epoch_A)
    # Find local maxima, indicating perigee
    perigee_times_A = find_perigee_times(Lstar_A, Epoch_A)
    
    print("Calculating K (RBSP-A)")
    K_A = Xj_A*100 * np.sqrt(Bmin_A*1e-5) * np.sqrt(Bmirr_A*1e-5) # R_E*G^(1/2)
    
    print("Calculating L* (RBSP-B)")
    #results_B = irbem.get_Lstar(time_B, position_B, alpha=alpha_B[0:9], extMag=extMag, omnivals=omnivals_refined_B)
    Bmin_B, Bmirr_B, Lm_B, Lstar_B, MLT_B, Xj_B = results_B["Bmin"], results_B["Bmirr"], results_B["Lm"], results_B["Lstar"], results_B["MLT"], results_B["Xj"]
    if len(Bmin_B.shape) == 1:
        Bmin_B = Bmin_B.reshape(len(Bmin_B), 1)
    Bmirr_B = np.concatenate((Bmirr_B[:,:-1], Bmirr_B[:, ::-1]), axis=1)
    Lm_B    = np.concatenate((Lm_B[:,:-1], Lm_B[:, ::-1]), axis=1)
    Lstar_B = np.concatenate((Lstar_B[:,:-1], Lstar_B[:, ::-1]), axis=1)
    Lstar_B_nan = np.where(np.all(np.isnan(Lstar_B), axis=1))[0]
    Xj_B    = np.concatenate((Xj_B[:,:-1], Xj_B[:, ::-1]), axis=1)

    # Find when Lstar is NaN, typically indicating apogee
    apogee_times_B = find_apogee_times(Lstar_B, Epoch_B)
    # Find local maxima, indicating perigee
    perigee_times_B = find_perigee_times(Lstar_B, Epoch_B)

    print("Calculating K (RBSP-B)")
    K_B = Xj_B*100 * np.sqrt(Bmin_B*1e-5) * np.sqrt(Bmirr_B*1e-5) # R_E*G^(1/2)

#%% Plots
    '''
    # Plot ephemeris file data and calculated L* data for RBSP A&B
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))  
    ax1.scatter(Epoch_A, Lstar_interp_A[:,9], s=5)
    ax1.scatter(Epoch_B, Lstar_interp_B[:,9], s=5)
    ax2.scatter(Epoch_A, Lstar_A[:,4], s=5)
    ax2.scatter(Epoch_B, Lstar_B[:,4], s=5)
    ax1.set_title("Ephemeris L*")  # Top plot label
    ax2.set_title("L*")           # Bottom plot label
    # Force labels for first and last x-axis tick marks 
    min_epoch = datetime(1970, 1, 1) + timedelta(hours=math.floor((min_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = datetime(1970, 1, 1) + timedelta(hours=math.ceil((max_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax2.set_xlim(min_epoch, max_epoch) 
    ax2.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12) )
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    ax1.set_yticks(np.arange(2, 8, 1))
    ax1.set_ylim(2, 7)
    ax2.set_yticks(np.arange(2, 8, 1))
    ax2.set_ylim(2, 7)
    fig.autofmt_xdate()
    plt.show()
    
    # Plot ephemeris file data against calculated L* data for RBSP A
    fig, ax = plt.subplots()  
    ax.scatter(Lstar_interp_A[:,10], Lstar_A, s=2, color = 'blue')
    ax.plot([0, 6], [0, 6], color='black', linestyle='--')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xlabel('Ephemeris L*')
    ax.set_ylabel('L*')
    ax.set_title('Comparison of Ephemeris L* and L* for RBSP-A')
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()
    
    # Plot ephemeris file data against calculated L* data for RBSP B
    fig, ax = plt.subplots()  
    ax.scatter(Lstar_interp_B[:,10], Lstar_B, s=2, color = 'orange')
    ax.plot([0, 6], [0, 6], color='black', linestyle='--')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xlabel('Ephemeris L*')
    ax.set_ylabel('L*')
    ax.set_title('Comparison of Ephemeris L* and L* for RBSP-B')
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()
    '''
    '''
    # Plot calculated Mu data against L* for RBSP A for each pitch angle
    fig, axes = plt.subplots(9, 1, figsize=(12, 30), sharex=True)
    handles, labels = [], []
    colors = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, 12, dtype=int)]
    for i in range(9):
        ax = axes[i]
        for j in range(12):
            scatter = ax.scatter(Lstar_A[4138:5655, i], Mu_A[j, i, 4138:5655], s=4, color=colors[j])
            if i == 0:
                handles.append(scatter)
                labels.append(energy_channels_A[j])
        ax.set_title(f"Pitch Angle {alpha_A[i]:.2f}",fontsize=textsize+2)
        ax.set_ylabel(r'$\mu$', fontsize=textsize)
        ax.set_yscale('log') # Set y-axis to logarithmic scale
        ax.set_xlim(2, np.nanmax(Lstar_A[4138:5655, :]))
        ax.tick_params(axis='x', labelsize=textsize)
        ax.tick_params(axis='y', labelsize=textsize)
        ax.grid(True)
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=textsize, markerscale=5, title="Energy\nChannels\n(MeV)\n", title_fontsize=textsize) # Adjust legend
    ax.set_xlabel("L* from T89d", fontsize=textsize)
    plt.subplots_adjust(right=0.95, top=0.95)
    fig.suptitle(f"RBSP-A: {Epoch_A[4138]} to {Epoch_A[5655]}", fontsize=textsize + 4) # Add figure title
    plt.show()
    
    # Plot Mu vs time for each energy channel at each pitch angle for RBSP-A
    fig, axes = plt.subplots(9, 1, figsize=(12, 30), sharex=True)
    colors = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, 12, dtype=int)]
    for i in range(9):
        ax = axes[i]
        for j in range(12):
            scatter = ax.scatter(Epoch_A[:], Mu_A[j, i, :], s=4, color=colors[j])
        ax.set_title(f"Pitch Angle {alpha_A[i]:.2f}",fontsize=textsize+2)
        ax.set_ylabel(r'$\mu$', fontsize=textsize)
        ax.set_yscale('log') # Set y-axis to logarithmic scale
        ax.tick_params(axis='x', labelsize=textsize)
        ax.tick_params(axis='y', labelsize=textsize)
        ax.grid(True)
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    # Force labels for first and last x-axis tick marks 
    min_epoch = datetime(1970, 1, 1) + timedelta(hours=math.floor((min_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = datetime(1970, 1, 1) + timedelta(hours=math.ceil((max_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    plt.xticks(rotation=45, ha='right', fontsize=textsize)  # Rotate 45 degrees, align right
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=textsize, markerscale=5, title="Energy\nChannels\n(MeV)\n", title_fontsize=textsize) # Adjust legend
    ax.set_xlabel("Time", fontsize=textsize)
    plt.subplots_adjust(right=0.95, top=0.95)
    fig.suptitle(f"RBSP-A: Mu V Time", fontsize=textsize + 4) # Add figure title
    plt.show()
    
    # Plot ephemeris file data and calculated K data for RBSP A&B
    # setting pitch angle, for all time
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 5))
    ax1.scatter(Epoch_A, K_interp_A[:, 2], s=5)
    ax1.scatter(Epoch_B, K_interp_B[:, 2], s=5)
    ax1.set_title("Ephemeris K", fontsize=textsize)
    ax1.set_ylim(-0.1, 3)
    ax1.set_yticks(np.arange(0, 4, 1))
    ax1.set_ylabel("K", fontsize=textsize)
    ax1.tick_params(axis='both', labelsize=textsize)  # Set tick label size
    ax2.scatter(Epoch_A, K_A[:, 7], s=5, color='tab:blue')
    ax2.scatter(Epoch_A, K_A[:, 9], s=5, color='tab:blue')
    ax2.scatter(Epoch_B, K_B[:, 7], s=5, color='tab:orange')
    ax2.scatter(Epoch_B, K_B[:, 9], s=5, color='tab:orange')
    ax2.set_title("Calculated K", fontsize=textsize)
    ax2.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=6))
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    plt.xticks(rotation=45, ha='right', fontsize=textsize)  # Rotate 45 degrees, align right
    ax2.set_ylim(-0.1, 3)
    ax2.set_yticks(np.arange(0, 4, 1))
    ax2.set_xlabel("Time", fontsize=textsize)
    ax2.set_ylabel("K", fontsize=textsize)
    ax2.tick_params(axis='both', labelsize=textsize)  # Set tick label size
    fig.suptitle("Pitch Angle = 80 degrees", fontsize=textsize)
    plt.show()
    
    # Plot ephemeris file data and calculated K data for RBSP A&B
    # setting time point, for all pitch angles
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 5))
    ax1.scatter(alpha_ephem_A, K_interp_A[500, :])
    ax1.scatter(alpha_ephem_B, K_interp_B[500, :])
    ax1.set_title("Ephemeris K", fontsize=textsize)
    ax1.set_ylabel("K", fontsize=textsize)
    ax1.tick_params(axis='both', labelsize=textsize)
    ax2.scatter(alpha_A, K_A[500, :])
    ax2.scatter(alpha_A, K_B[500, :])
    ax2.set_title("Calculated K", fontsize=textsize)
    ax2.set_xlabel("Pitch Angle (degrees)", fontsize=textsize)
    ax2.set_ylabel("K", fontsize=textsize)
    ax2.tick_params(axis='both', labelsize=textsize)
    fig.suptitle(f"Time = {Epoch_A[500]}", fontsize=textsize)
    plt.show()
    '''
    
    #%% GEO to GSM
    '''
    def geo_to_gsm(epoch, position):
        # assumes UTC time
        epoch_ticks = Ticktock(epoch, 'UTC') 
        geo_coords = coords.Coords(position, 'GEO', 'car', ticks=epoch_ticks)
        
        # Convert to GSM coordinates
        gsm_coords = geo_coords.convert('GSM','car') 
        
        # Extract GSM coordinates as a NumPy array
        gsm_position = gsm_coords.data
        return gsm_position
    print("Converting from GEO to GSM:")
    print("Processing RBPS-A")
    gsm_A = geo_to_gsm(Epoch_A, Position_A)
    print("Processing RBPS-B")
    gsm_B = geo_to_gsm(Epoch_B, Position_B)
    '''