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
from analysis_functions import interpolate_Ephem
from analysis_functions import get_Omni
from analysis_functions import extend_alpha
from analysis_functions import find_alpha
from analysis_functions import energy_from_mu_alpha

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
    
    print("Saving Data")
    # Create a dictionary to store the variables
    data_to_save = {
        'results_A': results_A,
        'results_B': results_B,
    }
    # Save the dictionary to a .npz file (NumPy zip archive)
    np.savez('vital_data.npz', **data_to_save)

#%% Calculate K
    # Calculate K from X_j: K = X_j * \sqrt(B_mirr)
    print("Calculating K (RBSP-A)")
    K_A = Xj_A * np.sqrt(Bmirr_A*1e-5) # R_E*G^(1/2)
    # Find alpha at each time point given a set K
    alpha_set_A = find_alpha(K_set, K_A, alpha_A_extend) 

    # Calculate K from X_j: K = X_j * \sqrt(B_mirr)
    print("Calculating K (RBSP-B)")
    K_B = Xj_B * np.sqrt(Bmirr_B*1e-5) # R_E*G^(1/2)
    # Find alpha at each time point given a set K
    alpha_set_B = find_alpha(K_set, K_B, alpha_B_extend) 
    

#%% Find energy for a given mu at each time point
    # Test to ensure derivation is correct    
    Energy_kin = np.sqrt(2*electron_E0*Mu_A[:,:,0]*(Blocal_A[0]*1e-5)/np.sin(np.radians(alpha_A_extend))**2+electron_E0**2)-electron_E0    
    # Find kinetic energy of particle population with a given Mu and alpha calculated from a given K 
    EnergyofMuAlpha_A = energy_from_mu_alpha(Mu_set, alpha_set_A, Blocal_A)
    # Find kinetic energy of particle population with a given Mu and alpha calculated from a given K 
    EnergyofMuAlpha_B = energy_from_mu_alpha(Mu_set, alpha_set_B, Blocal_B)


#%% Plots
    '''
    # Plot ephemeris file data and calculated L* data for RBSP A&B
    handles, labels = [], []
    fig, (axA, axB) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 5))  
    ephemplot_A = axA.scatter(Epoch_A, Lstar_interp_A[:,9], s=5, label="Ephemeris")
    calcplot_A = axA.scatter(Epoch_A, Lstar_A[:,4], s=5, label="IRBEM")
    ephemplot_B = axB.scatter(Epoch_B, Lstar_interp_B[:,9], s=5, label="Ephemeris")
    calcplot_B = axB.scatter(Epoch_B, Lstar_B[:,4], s=5, label="IRBEM")
    axA.set_title("RBSP-A")  # Top plot label
    axB.set_title("RBSP-B")           # Bottom plot label
    # Force labels for first and last x-axis tick marks 
    min_epoch = datetime(1970, 1, 1) + timedelta(hours=math.floor((min_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = datetime(1970, 1, 1) + timedelta(hours=math.ceil((max_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    axB.set_xlim(min_epoch, max_epoch) 
    axB.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12) )
    axB.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    axB.set_yticks(np.arange(2, 8, 1))
    axB.set_ylim(2, 7)
    fig.autofmt_xdate()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=textsize, markerscale=5)
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
    colors = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, 10, dtype=int)]
    handles, labels = [], []
    for i in range(9):
        ax = axes[i]
        for j in range(len(energy_channels_A)-2):
            scatter = ax.scatter(Epoch_A[:], Mu_A[j, i, :], s=4, color=colors[j])
            if i == 0:  # Only add handles and labels for the first subplot
                handles.append(scatter)
                labels.append(f"{energy_channels_A[j]:.2f}") #create label from energy_channels_A
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
    # 25: interp = 13, calc = 2, 14
    # 45: interp = 9, calc = 4, 12
    # 80: interp = 2, calc = 7, 9
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 5))
    ax1.scatter(Epoch_A, K_interp_A[:, 9], s=5)
    ax1.scatter(Epoch_B, K_interp_B[:, 9], s=5)
    ax1.set_title("Ephemeris K", fontsize=textsize)
    ax1.set_ylim(-0.1, 3)
    ax1.set_yticks(np.arange(0, 4, 1))
    ax1.set_ylabel("K", fontsize=textsize)
    ax1.tick_params(axis='both', labelsize=textsize)  # Set tick label size
    ax2.scatter(Epoch_A, K_A[:, 4], s=5, color='tab:blue')
    ax2.scatter(Epoch_A, K_A[:, 12], s=5, color='tab:blue')
    ax2.scatter(Epoch_B, K_B[:, 4], s=5, color='tab:orange')
    ax2.scatter(Epoch_B, K_B[:, 12], s=5, color='tab:orange')
    ax2.set_title("Calculated K", fontsize=textsize)
    ax2.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=6))
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    plt.xticks(rotation=45, ha='right', fontsize=textsize)  # Rotate 45 degrees, align right
    #ax2.set_ylim(-0.1, 3)
    ax2.set_yticks(np.arange(0, 4, 1))
    ax2.set_xlabel("Time", fontsize=textsize)
    ax2.set_ylabel("K", fontsize=textsize)
    ax2.tick_params(axis='both', labelsize=textsize)  # Set tick label size
    fig.suptitle("Pitch Angle = 45 degrees", fontsize=textsize)
    plt.show()
    
    
    
    # Plot ephemeris file data and calculated K data for RBSP A&B
    # setting time point, for all pitch angles
    smooth_alpha = np.linspace(0, 90, 100)
    time_index = 500
    fig, (axA, axB) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(20, 5))
    scatter_calc_A = axA.scatter(alpha_A_extend, K_A[time_index, :], s=100, label="Calculated")  # Blue circles
    exponential_fit_A = axA.plot(smooth_alpha, exponential_fit(smooth_alpha, **K_alpha_fit_A[time_index]), ls='--', linewidth=2, label="Eponential Fit")[0]
    power_fit_A = axA.plot(smooth_alpha, power_law_fit(smooth_alpha, **K_alpha_fit_power_A[time_index]), ls=':', linewidth=3, color=exponential_fit_A.get_color(), label="Power Law Fit")[0]
    scatter_ephem_A = axA.scatter(alpha_ephem_A, K_interp_A[time_index, :], s=100, label="Ephemeris", marker='^')
    axA.set_title("RBSP-A", fontsize=textsize)
    axA.set_ylabel("K", fontsize=textsize)
    axA.tick_params(axis='both', labelsize=textsize)
    axA.set_ylim([-0.1, 1])
    
    axB.scatter(alpha_B_extend, K_B[time_index, :], s=100, label="Calculated")
    axB.plot(smooth_alpha, exponential_fit(smooth_alpha, **K_alpha_fit_B[time_index]), ls='--', linewidth=2, label="Eponential Fit")[0]
    axB.plot(smooth_alpha, power_law_fit(smooth_alpha, **K_alpha_fit_power_B[time_index]), ls=':', linewidth=3,  color=exponential_fit_A.get_color(), label="Power Law Fit")[0]
    axB.scatter(alpha_ephem_B, K_interp_B[time_index, :], s=100, label="Ephemeris", marker='^')
    axB.set_title("RBSP-B", fontsize=textsize)
    axB.set_xlabel("Pitch Angle (degrees)", fontsize=textsize)
    axB.set_ylabel("K", fontsize=textsize)
    axB.tick_params(axis='both', labelsize=textsize)
    axB.set_ylim([-0.1, 1])
    
    # Create a single legend
    handles = [scatter_calc_A, exponential_fit_A, power_fit_A, scatter_ephem_A] #handles from A
    labels = [h.get_label() for h in handles]
    
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=textsize) #create legend in figure space.

    fig.suptitle(f"Time = {Epoch_A[time_index]}", fontsize=textsize)
    plt.show()
    
    
    
    # Plot linear interpolated alpha for K=0.1, comparing to IRBEM calculations
    fig, (axA, axB) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(15, 7))  
    scatter_alpha_set_A = axA.scatter(Epoch_A, alpha_set_A, s=10, label="Interpolated")
    scatter_alphaofK_A = axA.scatter(Epoch_A, alphaofK_A, s=10, label="IRBEM (equitorial)")
    scatter_alpha_set_B = axB.scatter(Epoch_B, alpha_set_B, s=10, label="Interpolated")
    scatter_alphaofK_B = axB.scatter(Epoch_B, alphaofK_B, s=10, label="IRBEM (equitorial)")
    axA.set_title("RBSP-A", fontsize=textsize)  # Top plot label
    axB.set_title("RBSP-B", fontsize=textsize)          # Bottom plot label
    axA.set_ylabel(r"$\alpha$", fontsize=textsize)
    axB.set_ylabel(r"$\alpha$", fontsize=textsize)
    axA.tick_params(axis='both', labelsize=textsize)  # Set tick label size
    # Force labels for first and last x-axis tick marks 
    min_epoch = datetime(1970, 1, 1) + timedelta(hours=math.floor((min_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = datetime(1970, 1, 1) + timedelta(hours=math.ceil((max_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    axB.set_xlim(min_epoch, max_epoch) 
    axB.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12) )
    axB.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    axB.set_xlabel("Time", fontsize=textsize)
    axB.tick_params(axis='both', labelsize=textsize)  # Set tick label size
    plt.xticks(rotation=45, ha='right', fontsize=textsize)  # Rotate 45 degrees, align right
    # Create a single legend
    handles, labels = axA.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=textsize)
    for handle in legend.legend_handles:
        handle.set_sizes([100.0]) # Adjust the size as needed
    fig.suptitle(f"Compare $\\alpha$ from interpolation to IRBEM for K=0.1", fontsize=textsize + 4) # Add figure title
    plt.show()
    
    
    
    # Plot mu v alpha at given time points for each energy channel
    time_index = 500
    colors = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, 10, dtype=int)]
    handles = []
    labels = []
    fig, (axA) = plt.subplots(figsize=(16, 4))
    for energy_channel in range(len(energy_channels_A)-2):
        scatter = axA.scatter(alpha_A_extend, Mu_A[energy_channel, :, time_index], s=50, color=colors[energy_channel], label=f"Energy Channel {energy_channel + 1}")
        handles.append(scatter)
        labels.append(f"{energy_channels_A[energy_channel]:.2f}") #create label from energy_channels_A
    axA.set_title("RBSP-A", fontsize=textsize)
    axA.set_ylabel(r"$\mu$ (MeV/G)", fontsize=textsize)
    axA.tick_params(axis='both', labelsize=textsize)
    axA.set_xlabel("Pitch Angle (degrees)", fontsize=textsize)
    #axA.set_yscale('log')
    
    # Create a single legend
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=textsize, markerscale=2, title="Energy\nChannels\n(MeV)\n", title_fontsize=textsize) # Adjust legend
    fig.suptitle(f"Time = {Epoch_A[time_index]}", fontsize=textsize+2)
    plt.subplots_adjust(top=0.85, right=0.9) #adjusted top and right.
    plt.show()
    
    
    # Plot mu v energy at given time points for each pitch angle
    time_index = 20000
    colors = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(alpha_A_extend), dtype=int)]
    handles = []
    labels = []
    fig, (axA) = plt.subplots(figsize=(16, 4))
    
    for alpha_index in range(len(alpha_A_extend)):  # Renamed loop variable
        scatter = axA.scatter(energy_channels_A[:-2], Mu_A[:-2, alpha_index, time_index], s=50, color=colors[alpha_index],
                            label=f"Pitch Angle {alpha_index + 1}")
        handles.append(scatter)
        labels.append(f"{alpha_A_extend[alpha_index]:.0f}")  # Create label from alpha_A_extend
    
    axA.set_title("RBSP-A", fontsize=textsize)
    axA.set_ylabel(r"$\mu$ (MeV/G)", fontsize=textsize)
    axA.tick_params(axis='both', labelsize=textsize)
    axA.set_xlabel("Energy (MeV)", fontsize=textsize)
    
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=textsize, markerscale=2,
            title="Pitch Angle\n(Degrees)\n", title_fontsize=textsize)  # Corrected legend title
    fig.suptitle(f"Time = {Epoch_A[time_index]}", fontsize=textsize + 2)
    plt.subplots_adjust(top=0.85, right=0.9)
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