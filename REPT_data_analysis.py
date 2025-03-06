#%% Import and Initialize
from spacepy import pycdf, toolbox as tb
import numpy as np
import os
import glob
# Plotting
from datetime import datetime, timedelta
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
# Time conversion
from spacepy.time import Ticktock
from spacepy.coordinates import Coords
import spacepy.omni as omni
import spacepy.irbempy as irbem
import scipy.constants as sc

# Import the latest version of OMNI data
#tb.update(omni2=True)

# Initialize global variables
textsize = 16
Re = 6378.137 #Earth's Radius

# Start main class
if __name__ == '__main__':
#%% Folder containing CDF files
    folder_path = "C:/Users/Will/Box/Multipoint_Box/REPT Data/April 2017 Storm/l2/"
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Error: Folder path not found: {folder_path}")
    
    ephemeris_path = "C:/Users/Will/Box/Multipoint_Box/REPT Data/April 2017 Storm/ephemeris/"
    if not os.path.exists(ephemeris_path):
        raise FileNotFoundError(f"Error: Ephemeris path not found: {ephemeris_path}")
    
    # Get all CDF file paths in the folder
    cdf_file_paths_A = glob.glob(folder_path + "rbspa*[!r]*.cdf") 
    cdf_file_paths_B = glob.glob(folder_path + "rbspb*[!r]*.cdf") 
    ephem_file_paths_A = glob.glob(ephemeris_path + "rbsp-a*[!r]*.cdf")
    ephem_file_paths_B = glob.glob(ephemeris_path + "rbsp-b*[!r]*.cdf")
    
    
#%% Function for reading in RBSP flux data
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
            cdf_data.close()
        # Convert from km to R_E
        Position = np.array(Position)
        Position = Position / Re
        # finish reading in data
        return Epoch, L, Position, FEDU, energy_channels
    
    # Read in data from RBSP CDF files
    print("Processing Flux Data:")
    Epoch_A, L_A, Position_A, FEDU_A, energy_channels_A = process_flux_data(cdf_file_paths_A)
    Epoch_B, L_B, Position_B, FEDU_B, energy_channels_B = process_flux_data(cdf_file_paths_B)
    
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
    # Function for reading in RBSP ephemeris data
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
    mag_key_unused = ['G1', 'G2', 'G3', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    
    def get_Omni(time, position):      
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
    
    time_A = Ticktock(Epoch_A[0:99], 'UTC')
    time_B = Ticktock(Epoch_B[0:99], 'UTC')
    position_A = Coords(Position_A[0:99,:], 'GEO', 'car')
    position_B = Coords(Position_B[0:99,:], 'GEO', 'car')
    alpha = 40
    alpha = np.atleast_1d(alpha)  # Ensure alpha is an array
    extMag = 'T89'
    omnivals_refined_A = get_Omni(Epoch_A, Position_A)
    omnivals_refined_B = get_Omni(Epoch_B, Position_B)
    
    electron_mass_mev = sc.electron_mass / (1e6 * sc.electron_volt)
    print("Calculating Mu (RBSP-A)")
    B_A = irbem.get_Bfield(time_A, position_A, extMag=extMag, omnivals=omnivals_refined_A)
    Blocal_A, Bvec_A = B_A["Blocal"], B_A["Bvec"]
    energy_grid, alpha_grid, blocal_grid = np.meshgrid(energy_channels_A, np.deg2rad(alpha_ephem_A), Blocal_A*1e-5, indexing='ij')
    Mu_A = (energy_grid**2 + 2 * energy_grid * electron_mass_mev * sc.c**2) * np.sin(alpha_grid)**2 / (2 * electron_mass_mev * sc.c**2 * blocal_grid)
    #Mu_A = (energy_channels_A^2+2*energy_channels_A*sc.electron_mass*sc.c^2)*np.sin(alpha)/(2*sc.electron_mass*sc.c^2*Blocal_A) 
    
    print("Calculating Mu (RBSP-B)")
    B_B = irbem.get_Bfield(time_B, position_B, extMag=extMag, omnivals=omnivals_refined_B)
    Blocal_B, Bvec_B = B_B["Blocal"], B_B["Bvec"]
    energy_grid, alpha_grid, blocal_grid = np.meshgrid(energy_channels_B, np.deg2rad(alpha_ephem_B), Blocal_B*1e-5, indexing='ij')
    Mu_B = (energy_grid**2 + 2 * energy_grid * electron_mass_mev * sc.c**2) * np.sin(alpha_grid)**2 / (2 * electron_mass_mev * sc.c**2 * blocal_grid)

    
    print("Calculating L* (RBSP-A)")
    results_A = irbem.get_Lstar(time_A, position_A, alpha=alpha, extMag=extMag, omnivals=omnivals_refined_A)
    Bmin_A, Bmirr_A, Lm_A, Lstar_A, MLT_A, Xj_A = results_A["Bmin"], results_A["Bmirr"], results_A["Lm"], results_A["Lstar"], results_A["MLT"], results_A["Xj"]
    #results = irbem.get_Lstar(Ticktock(Epoch_A[0:99], 'UTC'), Coords(Position_A[0:99,:], 'GEO', 'car'), alpha=40, extMag='T89', omnivals=omnivals_refined)
    print("Calculating L* (RBSP-B)")
    results_B = irbem.get_Lstar(time_A, position_A, alpha=alpha, extMag=extMag, omnivals=omnivals_refined_B)
    Bmin_B, Bmirr_B, Lm_B, Lstar_B, MLT_B, Xj_B = results_B["Bmin"], results_B["Bmirr"], results_B["Lm"], results_B["Lstar"], results_B["MLT"], results_B["Xj"]

    '''
    # Plot ephemeris file data and calculated L* data for RBSP A&B
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))  
    ax1.scatter(Epoch_A, Lstar_interp_A[:,10], s=5)
    ax1.scatter(Epoch_B, Lstar_interp_B[:,10], s=5)
    ax2.scatter(Epoch_A, Lstar_A, s=5)
    ax2.scatter(Epoch_B, Lstar_B, s=5)
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
    
    #%%Plot RBSP Flux Data with ephemeris Lm_ephem
    '''
    print("Plotting Data:")
    # Create a custom colormap based on 'nipy_spectral' to match with IDL rainbow
    cmap = plt.get_cmap('nipy_spectral') 
    new_cmap = cmap(np.linspace(0, 0.875, 256))  # Use only the first 87.5% of the colormap
    
    # Create a new colormap object
    custom_cmap = colors.ListedColormap(new_cmap)
    
    # Create the figure with subplots
    fig, axes = plt.subplots(len(energy_channels_A), 1, figsize=(20, 40), sharex=True)
    
    # Loop through each energy channel
    for i, ax in enumerate(axes.flat):
      # Create the scatter plot on the current subplot
      # divide by 1000 for keV to compare to Zhao 2018
      if FEDU_A is not None:
          subplot_A = ax.scatter(Epoch_A, Lm_ephem_A_interp, c=FEDU_A[:, i]/1000, cmap=custom_cmap, norm=colors.LogNorm())
          # Set colorbar limits to 5 orders of magnitude
          vmin_A, vmax_A = subplot_A.get_clim() 
      if FEDU_B is not None:
          subplot_B = ax.scatter(Epoch_B, Lm_ephem_B_interp, c=FEDU_B[:, i]/1000, cmap=custom_cmap, norm=colors.LogNorm())
          # Set colorbar limits to 5 orders of magnitude
          vmin_B, vmax_B = subplot_B.get_clim() 
    
      # Add labels and title
      ax.set_ylabel('L', fontsize=textsize)
      ax.set_title(f'RBSP REPT {energy_channels_A[i]:.2f} MeV Electron Spin-Averaged Flux', fontsize=textsize)
      # Force labels for first and last x-axis tick marks 
      min_epoch = datetime(1970, 1, 1) + timedelta(hours=math.floor((min_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
      max_epoch = datetime(1970, 1, 1) + timedelta(hours=math.ceil((max_epoch - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
      ax.set_xlim(min_epoch, max_epoch) 
      # Set time labels every 12 hours
      ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12) )
      ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H')) 
      ax.tick_params(axis='both', which='major', labelsize=textsize)
      ax.set_yticks(np.arange(2, 8, 1))  # Set ticks from 2 to 7 with interval 1
      ax.set_ylim(2, 7)
      ax.grid(True)
      
      cbar = plt.colorbar(subplot_A, ax=ax, shrink=0.9, pad=0.01)  # Adjust shrink as needed
      vmax = 10**math.ceil(math.log10(max(vmax_A,vmax_B)))
      vmin = vmax/10**4
      subplot_A.set_clim(vmin, vmax) 
      subplot_B.set_clim(vmin, vmax) 
      cbar.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), num=5))
      # Flux is in (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)
      cbar.set_label(label = 'Flux', fontsize=textsize)
      cbar.ax.tick_params(labelsize=textsize)
    
    # Add x-axis label for last plot
    ax.set_xlabel('UTC', fontsize=textsize)
    fig.suptitle('April 21-26, 2017 RBSP REPT Data', fontsize=textsize+4, y=0.9)
    
    # Remove extra subplots if there aren't enough energy channels
    if len(energy_channels_A) < len(axes.flat):
      for ax in axes.flat[len(energy_channels_A):]:
        fig.delaxes(ax)
    
    # Show the plot
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