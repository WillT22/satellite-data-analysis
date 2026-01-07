#%% Importing relevant libraries
import os
import glob
import sys
import datetime as dt
import time
# Add the current script directory to the system path to import local modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_script_dir)
import numpy as np
import scipy.constants as sc
import matplotlib.dates as mdates
from matplotlib import colors
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import importlib
# --- Import Custom Libraries for Data Processing ---
# GPS_PSD_func contains physics functions for adiabatic invariants (L, L*, Alpha, Mu)
import GPS_PSD_func
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import (QinDenton_period, load_data, data_period, AlphaOfK, 
                          find_Loss_Cone, find_local90PA, EnergyofMuAlpha, find_psd, 
                          find_McIlwain_L, find_Lstar)

# REPT_PSD_func contains data handling functions specific to the REPT instrument
import REPT_PSD_func
importlib.reload(REPT_PSD_func)
from REPT_PSD_func import (process_l3_data, time_average, find_mag, Average_FluxbyPA, Interp_Flux)

#%% Global Variables
textsize = 16
Re = 6378.137 #Earth's Radius

# Adiabatic Invariant Targets
# Mu (First Invariant): Related to particle energy and magnetic field strength
Mu_set = np.array((2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G
# K (Second Invariant): Related to the particle's bounce path length
K_set = np.array((0.1, 1, 2)) # R_E*G^(1/2)

# Execution Mode: 'save' calculates new data and saves to .npz; 'load' reads existing .npz files
mode = 'save' 

## Current available storms: 'april2017storm', 'aug2018storm', 'oct2012storm', 'latefeb2019storm', 'may2019storm', 'sep2019storm'
storm_name = 'latefeb2019storm' 

# Plotting options
plot_flux = True
plot_flux_all = True
plot_energies = False
plot_psd = True
plot_radial = False
plot_radial_Lstar = False
PAD_calculate = False

# Data paths initialization
REPT_data_root = '/home/wzt0020/sat_data_analysis/REPT_data/'
input_folder = os.path.join(REPT_data_root, storm_name)
base_save_folder = os.path.join(REPT_data_root, storm_name)

# External Magnetic Field Model
extMag = 'TS04' # 'T89c', 'TS04', NOT 'TS07'

# Define storm time periods based on name
if storm_name == 'april2017storm':
    start_date  = dt.datetime(2017, 4, 21, 00, 00, 0)
    stop_date   = dt.datetime(2017, 4, 26, 00, 00, 0)

elif storm_name == 'aug2018storm':
    start_date = dt.datetime(2018, 8, 25, 0, 0, 0)
    stop_date = dt.datetime(2018, 8, 28, 0, 0, 0)

elif storm_name == 'oct2012storm':
    start_date  = dt.datetime(2012, 10, 7, 00, 00, 0)
    stop_date   = dt.datetime(2012, 10, 11, 00, 00, 0)

elif storm_name == 'latefeb2019storm':
    start_date  = dt.datetime(2019, 2, 27, 00, 00, 0)
    stop_date   = dt.datetime(2019, 3, 4, 00, 00, 0)

elif storm_name == 'may2019storm':
    start_date  = dt.datetime(2019, 5, 10, 00, 00, 0)
    stop_date   = dt.datetime(2019, 5, 17, 00, 00, 0)

elif storm_name == 'sep2019storm':
    start_date  = dt.datetime(2019, 8, 31, 00, 00, 0)
    stop_date   = dt.datetime(2019, 9, 5, 00, 00, 0)

# Conversions
# electron mass in MeV is (m_e [kg] * c^2 [m^2/s^2]) [J] / (sc.eV [J/eV] * 10^6 [eV/MeV])
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6) # this is m_0*c^2
# b_satellite and b_equator are in Gauss: 1 G = 10^5 nT

# Initialize runtime timer
start_time = time.perf_counter()

# Load Solar Wind / Geomagnetic Indices (Qin-Denton dataset)
# Required for magnetic field models (TS04/T89)
QD_storm_data = QinDenton_period(start_date, stop_date)

#%% Main Execution Block
if __name__ == '__main__':
    
### 1. Load Data ###
    raw_save_path = os.path.join(base_save_folder, 'raw_rept.npz')
    if mode == 'save':
        # PROCESS RAW CDF FILES
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Error: Folder path not found: {input_folder}")
        
        # Get all CDF file paths in the folder
        file_paths_l3_A = glob.glob(input_folder + "/rbspa*[!r]*.cdf") 
        file_paths_l3_B = glob.glob(input_folder + "/rbspb*[!r]*.cdf")
        
        REPT_data_raw = {}
        if len(file_paths_l3_A) != 0:
            REPT_data_raw['rbspa'] = process_l3_data(file_paths_l3_A)
        if len(file_paths_l3_B) != 0:
            REPT_data_raw['rbspb'] = process_l3_data(file_paths_l3_B)
    
        # Save Data for later recall:
        print("Saving Raw REPT Data...")
        np.savez(raw_save_path, **REPT_data_raw)
        print("Data Saved \n")
    
    elif mode == 'load':
        # LOAD PRE-PROCESSED NUMPY FILES
        raw_data_load = np.load(raw_save_path, allow_pickle=True)
        REPT_data_raw = load_data(raw_data_load)
        raw_data_load.close()
        del raw_data_load

    ### 2. Time Filtering & Averaging ###
    REPT_data = {}
    for satellite, sat_data in REPT_data_raw.items():
        print(f'Restricting Time Period for satellite {satellite}...')
        # Filter data to the exact storm interval defined in Globals
        REPT_data[satellite] = data_period(sat_data, start_date, stop_date)
    del REPT_data_raw

    # Average flux data into 1-minute bins to reduce noise
    for satellite, sat_data in REPT_data.items():
        print(f"Time Averaging Fluxes for satellite {satellite}...")
        REPT_data[satellite] = time_average(sat_data, satellite)

    ### 3. Magnetic Field Data ###
    # Extract B-field vectors (used for pitch angle calculation and field modeling)
    for satellite, sat_data in REPT_data.items():
        print(f"Extracting Magnetic Field Data for satellite {satellite}...")
        REPT_data[satellite] = find_mag(sat_data, satellite)

    # Average fluxes assuming gyrotropy (symmetry: flux at alpha = flux at 180-alpha)
    for satellite, sat_data in REPT_data.items():
        print(f"Averaging Fluxes with the same PA for satellite {satellite}...")
        REPT_data[satellite] = Average_FluxbyPA(sat_data, satellite)

    ### 4. Calculate Adiabatic Invariants (Mu) ###
    # Calculate Mu for every energy/pitch angle combination based on local B-field
    for satellite, sat_data in REPT_data.items():
        print(f"Calculating Mu from nominal energies and pitch angles for satellite {satellite}...")
        energy_grid, alpha_grid, blocal_grid = np.meshgrid(sat_data['Energy_Channels'], np.deg2rad(sat_data['Pitch_Angles']), sat_data['b_satellite'], indexing='ij')
        # Formula: Mu = E_perp / B = (E^2 + 2*E*E0) * sin^2(alpha) / (2*E0*B)
        REPT_data[satellite]['Mu_calc'] = (energy_grid**2 + 2 * energy_grid * E0) * np.sin(alpha_grid)**2 / (2 * E0 * blocal_grid)
    del energy_grid, alpha_grid, blocal_grid

    ### 5. Calculate Pitch Angle (Alpha) for Constant K ###
    # We need to find what local pitch angle corresponds to our fixed K values
    alphaofK_filename = f"alphaofK_{extMag}.npz"
    alphaofK_save_path = os.path.join(base_save_folder, alphaofK_filename)
    
    if mode == 'save':
        alphaofK = {}
        for satellite, sat_data in REPT_data.items():
            print(f"Calculating Pitch Angle for satellite {satellite}...")
            # Calculates alpha for the target K values using the external field model
            alphaofK[satellite] = AlphaOfK(sat_data, K_set, extMag)

        # Save Data for later recall:
        print("Saving AlphaofK Data...")
        np.savez(alphaofK_save_path, **alphaofK)
        print("Data Saved \n")

        ### 6. Calculate Loss Cone and Equatorial B ###
        for satellite, sat_data in REPT_data.items():
            print(f"Calculating Equatorial B-field for satellite {satellite}...")
            # Finds B_min (equator), B_footpoint (atmosphere), and the loss cone angle
            REPT_data[satellite]['b_min'], REPT_data[satellite]['P_min'], REPT_data[satellite]['b_footpoint'], REPT_data[satellite]['loss_cone'] = find_Loss_Cone(sat_data, extMag=extMag)
    
        ### 7. Determine Local 90-Degree Pitch Angle ###
        # Finds the equatorial pitch angle that maps to 90 degrees locally
        for satellite, sat_data in REPT_data.items():
            print(f"Finding Local 90 degree pitch angle for {satellite}...")
            sat_data['local90PA'] = find_local90PA(sat_data)

    elif mode == 'load':
        # Load previously calculated Alpha(K) data
        alphaofK_load = np.load(alphaofK_save_path, allow_pickle=True)
        alphaofK = load_data(alphaofK_load)
        for satellite, sat_data in REPT_data.items():
            epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
            alphaofK[satellite] = pd.DataFrame(alphaofK[satellite], index=epoch_str, columns=np.atleast_1d(K_set))
        alphaofK_load.close()
        del alphaofK_load

    # --- Load main processed data if in load mode ---
    save_path = os.path.join(base_save_folder, f'rept_data_{extMag}_new2.npz')
    if mode == 'load':
        complete_load = np.load(save_path, allow_pickle=True)
        REPT_data = load_data(complete_load)
        complete_load.close()
        del complete_load
 
    ### 8. Find Energy for Constant Mu and Alpha ###
    # Determines the energy corresponding to constant Mu at the calculated pitch angles
    energyofmualpha = {}
    energyofmualpha_filename = f"energyofmualpha_{extMag}.npz"
    energyofmualpha_save_path = os.path.join(base_save_folder, energyofmualpha_filename)
    
    for satellite, sat_data in REPT_data.items():
        print(f"Calculating Energy of Mu and Alpha for satellite {satellite}")
        energyofmualpha[satellite] = EnergyofMuAlpha(sat_data, Mu_set, alphaofK[satellite])

    if mode == 'save':
        print("Saving REPT Data (Energy Calculations)...")
        np.savez(energyofmualpha_save_path, **energyofmualpha)
        print("Data Saved \n")

    ### 9. Interpolate Flux ###
    # Interpolate measured flux to the specific Energy/Pitch Angle required for constant Mu/K
    flux = {}
    flux_alpha = {}
    for satellite, sat_data in REPT_data.items():
        print(f"Interpolating flux for satellite {satellite}")
        flux[satellite], flux_alpha[satellite] = Interp_Flux(sat_data, alphaofK[satellite], energyofmualpha[satellite])

### 10. Calculate Phase Space Density (PSD) ###
    # Convert differential flux to PSD: PSD = Flux / p^2
    for satellite, sat_data in REPT_data.items():
        print(f"Calculating PSD for satellite {satellite}")
        REPT_data[satellite]['PSD'] = find_psd(flux[satellite], energyofmualpha[satellite])

    ### 11. Calculate L-Shell Parameters ###
    if mode == 'save':
        # Calculate McIlwain L (dipole-like shell parameter)
        for satellite, sat_data in REPT_data.items():
            print(f"Calculating L for satellite {satellite}...")
            REPT_data[satellite] = find_McIlwain_L(sat_data, alphaofK[satellite], extMag=extMag)

        print("Saving REPT Data (with L)...")
        np.savez(save_path, **REPT_data)
        print("Data Saved \n")
        
    # Calculate L* (Roederer L / Drift Shell) using LANLGeoMag
        for satellite, sat_data in REPT_data.items():
            print(f"Calculating L* for satellite {satellite}...")
            REPT_data[satellite] = find_Lstar(sat_data, alphaofK[satellite], extMag=extMag)

        print("Saving REPT Data (with L*)...")
        np.savez(save_path, **REPT_data)
        print("Data Saved \n")

    ### 12. Calculate Pitch Angle Distribution (PAD) Model ###
    # Optional step to fit Zhao et al. (2018) PAD models to the data
    if PAD_calculate == True:
        PAD_filename = f"REPT_PAD_model_{extMag}.npz"
        PAD_scale_filename = f"REPT_PAD_scale_{extMag}.npz"
        PAD_save_path = os.path.join(base_save_folder, PAD_filename)
        PAD_scale_save_path = os.path.join(base_save_folder, PAD_scale_filename)
        if mode == 'save':
            from Zhao_2018_PAD_Model import (import_Zhao_coeffs, find_Zhao_PAD_coeffs, create_PAD, PAD_Scale_Factor)
            
            Zhao_coeffs = import_Zhao_coeffs()

            Zhao_coeffs_REPT = {}
            REPT_PAD_Model = {}
            REPT_PAD_scale = {}
            REPT_PAD_Int = {}

            for satellite, sat_data in REPT_data.items():
                print(f"Calculating PAD model for satellite {satellite}...")
                # Fit coefficients to the current data
                Zhao_coeffs_REPT[satellite] = find_Zhao_PAD_coeffs(sat_data, QD_storm_data, energyofmualpha[satellite], extMag)
                # Create the full PAD model
                REPT_PAD_Model[satellite] = create_PAD(sat_data, Zhao_coeffs_REPT[satellite], alphaofK[satellite])
                # Calculate scale factors to normalize model to measured flux
                REPT_PAD_scale[satellite], REPT_PAD_Int[satellite] = PAD_Scale_Factor(sat_data, Zhao_coeffs_REPT[satellite], alphaofK[satellite])
            print(f"PAD model calculation completed.")

            # Save Data for later recall:
            print("\nSaving REPT PAD Model Data...")
            np.savez(PAD_save_path, **REPT_PAD_Model )
            print("Data Saved \n")

            print("\nSaving REPT PAD Scale Data...")
            np.savez(PAD_scale_save_path, **REPT_PAD_scale )
            print("Data Saved \n")

        if mode == 'load':
            PAD_model_load = np.load(PAD_save_path, allow_pickle=True)
            REPT_PAD_Model = load_data(PAD_model_load)
            PAD_model_load.close()
            del PAD_model_load
            
            PAD_scale_load = np.load(PAD_scale_save_path, allow_pickle=True)
            REPT_PAD_scale = load_data(PAD_scale_load)
            PAD_scale_load.close()
            del PAD_scale_load


    ### Execution time tracking ###
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    def format_runtime(elapsed_time):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        return f"Script runtime: {hours}h {minutes}m {seconds:.2f}s"

    print(format_runtime(elapsed_time))

#%% PLOTTING SECTIONS
#%% --- Plot 1: Flux for a Single Energy Channel ---
if plot_flux==True:
    # Select target energy (closest match)
    energy = 2.1 # MeV
    energy_channels = REPT_data[list(REPT_data.keys())[0]]['Energy_Channels']
    i_energy = np.argmin(np.abs(energy_channels - energy))
    energy = energy_channels[i_energy] # use exact energy from REPT channels

    # Set time range for the plot
    time_start, time_stop = start_date, stop_date

    time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0)
    time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0)

    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1e3))
    max_val = np.nanmax(np.log10(1e4))

    if extMag == 'T89c': plot_extMag = 'T89'
    else: plot_extMag = extMag

    fig, ax = plt.subplots(figsize=(16, 4))
    for satellite, sat_data in REPT_data.items():
        # Mask data by time
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        
        # Get flux slice for the selected energy
        flux_slice = sat_data['FEDU_averaged'][:,:,i_energy]
        flux_temp_mask = np.where(flux_slice >= 0, flux_slice, np.nan)
        # Average over pitch angles (axis 1)
        flux_plot = np.nanmean(flux_temp_mask, axis=1)/2
        
        # Filter valid fluxes
        flux_mask = (flux_plot > 0) & (flux_plot != np.nan)
        combined_mask = flux_mask & sat_iepoch_mask

        # Scatter plot: Time vs L, colored by Flux
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], sat_data[f'L_LGM_{plot_extMag}IGRF'][combined_mask],
                            c=np.log10(flux_plot[combined_mask]), vmin=min_val, vmax=max_val)

    # Plot formatting
    ax.set_title(f"RBSP A&B REPT, {energy} MeV Electron Differential Flux", fontsize=textsize + 2)
    ax.set_ylabel(r"McIlwain L", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    
    ax.set_xlim(time_start, time_stop)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    ax.set_ylim(3, 6.5)
    ax.grid(True)

    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    cbar.set_label(label = r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)

    plt.show()

#%% --- Plot 2: Flux for ALL Energy Channels (Stacked Subplots) ---
if plot_flux_all==True:
    # Filter for lower energy channels (< 4 MeV)
    energy_channels = REPT_data[list(REPT_data.keys())[0]]['Energy_Channels']
    energy_channels = energy_channels[energy_channels<4]

    # Set time range for the plot
    time_start, time_stop = start_date, stop_date

    # time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0)
    # time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0)
    
    if extMag == 'T89c': plot_extMag = 'T89'
    else: plot_extMag = extMag

    # Create subplots for each energy channel
    fig, axes = plt.subplots(len(energy_channels),1,figsize=(20, 10),sharex=True,sharey=False)
    
    # Loop through each energy channel to create a subplot
    for i_energy, energy in enumerate(energy_channels):
        ax = axes[i_energy]
        # Inner Loop: Plot data from all available satellites (RBSP A & B)
        for satellite, sat_data in REPT_data.items():
            # Time Mask within specified range
            sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
            
            # Extract Flux Data for current energy channel
            # Slice the 3D array: [Time, PitchAngle, Energy] -> [Time, PitchAngle]
            flux_slice = sat_data['FEDU_averaged'][:,:,i_energy]
            # Replace negative values (fill/error codes) with NaN to exclude them from calculations.
            flux_temp_mask = np.where(flux_slice >= 0, flux_slice, np.nan)
            # Dividing by 2 accounts for averaging over two hemispheres or symmetric PAD assumption
            flux_plot = np.nanmean(flux_temp_mask, axis=1)/2

            # Create final mask for valid positive flux values
            flux_mask = (flux_plot > 0) & (flux_plot != np.nan)
            combined_mask = flux_mask & sat_iepoch_mask
            
            # Dynamically set max colorbar value based on data range
            vmax = np.ceil(max(np.log10(flux_plot[combined_mask])))

            # Plot L-shell vs Time, colored by log10(Flux)
            scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], sat_data[f'L_LGM_{plot_extMag}IGRF'][combined_mask],
                                c=np.log10(flux_plot[combined_mask]), vmin=0, vmax=vmax)

        # Subplot formatting
        ax.set_title(f"{energy:.2f} MeV", fontsize=textsize+2)
        ax.set_xlim(time_start, time_stop)
        ax.set_ylim(3, 6.5)
        ax.tick_params(axis='both', labelsize=textsize, pad=5)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.grid(True)

        # Add colorbar for each subplot
        cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, 
                            format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
        # Adjust colorbar ticks based on data range (every 1 or 2 orders of magnitude)
        if vmax > 5:
            cbar.locator = matplotlib.ticker.MultipleLocator(2)
        else:
            cbar.locator = matplotlib.ticker.MultipleLocator(1)
        cbar.ax.tick_params(labelsize=textsize)

    # Format the bottom-most x-axis (Time)
    if i_energy >= len(energy_channels)-1:
        ax.set_xlabel('Time (UTC)', fontsize=textsize,labelpad=2)
        ax.set_xlim(time_start, time_stop)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        ax.tick_params(axis='x', labelsize=textsize, pad=10)

    # Add shared y-axis label and colorbar label
    fig.text(0.96, 0.5, r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', 
         fontsize=textsize, rotation='vertical', va='center')

    # Final layout adjustments
    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(right=0.95, hspace=0.2)
    fig.suptitle(f'RBSP REPT Differential Flux {time_start.strftime('%Y-%m-%d %H')} to {time_stop.strftime('%Y-%m-%d %H')}', fontsize=textsize + 4, y=0.94)
    plt.show()

#%% --- Plot 3: Energy vs L* (Colored by Time) ---
if plot_energies==True:
    # Select specific adiabatic invariant coordinates (K, Mu)
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    # Set time range for the plot
    time_start, time_stop = start_date, stop_date

    # time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0)
    # time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0)

    # Configure colormap to represent time progression
    colormap_name = 'viridis'
    cmap = plt.cm.get_cmap(colormap_name)
    vmin = mdates.date2num(time_start)
    vmax = mdates.date2num(time_stop)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Iterate through satellites and plot their data points
    for satellite, sat_data in REPT_data.items():
        # Create masks for valid time, energy, and L-shell values
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        # extract energy values for the selected Mu and flatten to 1D
        energy_plot = energyofmualpha[satellite][k].values[:,i_mu].copy().flatten()
        energy_mask = (energy_plot > 0) & (energy_plot != np.nan)
        lstar_mask = (sat_data['Lstar'][:,i_K] > 0).flatten()
        combined_mask = sat_iepoch_mask & energy_mask & lstar_mask
        
        # Scatter plot: L* vs Energy, colored by Time
        scatter_plot = ax.scatter(sat_data['Lstar'][combined_mask,i_K], energy_plot[combined_mask], 
                               c=mdates.date2num(sat_data['Epoch'].UTC[combined_mask]), cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add horizontal colorbar for Time
    cbar = fig.colorbar(scatter_plot, ax=ax, orientation='horizontal', pad=0.11)
    cbar.set_label('Time (UTC)', fontsize=textsize)
    cbar.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    cbar.ax.tick_params(labelsize=textsize-2)

    # Annotate plot with K and Mu parameters
    ax.text(0.5, 0.92, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$", 
            transform=ax.transAxes, fontsize=textsize)
    
    # Axis limits and labels
    ax.set_xlim(3.8,5.2)
    ax.set_ylim(1,3.5)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlabel(r"L*", fontsize=textsize)
    ax.set_ylabel(r"Energy (MeV)", fontsize=textsize)
    ax.grid(True)
    

#%% --- Plot 4: Phase Space Density (PSD) ---
if plot_psd==True:
    # Select specific adiabatic invariant coordinates (K, Mu)
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    fig, ax = plt.subplots(figsize=(16, 4))

    # Configure custom colormap for high dynamic range PSD
    colorscheme = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))
    cmap = colors.ListedColormap(colorscheme)

    # Set log-scale color limits for PSD
    min_val = np.nanmin(np.log10(1e-12))
    max_val = np.nanmax(np.log10(1e-5))

    # Iterate through satellites and plot their data points
    for satellite, sat_data in REPT_data.items():
        # Get PSD values for specific K/Mu
        psd_plot = REPT_data[satellite]['PSD'][k].values[:,i_mu].copy().flatten()
        
        # Filter valid PSD values and L-shells
        psd_mask = (psd_plot > 0) & (psd_plot != np.nan)
        lstar_mask = sat_data['Lstar'][:,0]>0
        combined_mask = psd_mask & lstar_mask

        # Scatter plot: Time vs L*, colored by log10(PSD)
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], sat_data['Lstar'][combined_mask,i_K],
                            c=np.log10(psd_plot[combined_mask]), cmap=cmap, vmin=min_val, vmax=max_val)

    # Axis limits and labels
    ax.set_title(f"RBSP A&B REPT, K={k:.1f} $G^{{1/2}}R_E$, $\\mu$={mu:.0f} $MeV/G$", fontsize=textsize + 2)
    ax.set_ylabel(r"L*", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    # Force X-axis (Time) limits based on storm start/stop 
    min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    ax.set_ylim(3, 6)
    ax.grid(True)

    # Add colorbar for PSD
    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    tick_locations = np.arange(min_val, max_val + 1)
    cbar.set_ticks(tick_locations)
    cbar.set_label(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95) 

    plt.show()

#%% --- Plot 5: PSD Radial Profiles (Without L* Averaging) ---
if plot_radial==True:
    # Select RBSP satellite to plot
    sat_select = 'rbspb'
    sat_data = REPT_data[sat_select]

    # Select specific adiabatic invariant coordinates (K, Mu)
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    # Set PSD limits for plotting
    min_val = np.nanmin(1e-11)
    max_val = np.nanmax(1e-5)

    # Set time range for the plot
    time_start, time_stop = start_date, stop_date

    time_start = dt.datetime(start_date.year, 5, 11, 0, 0, 0)
    time_stop = dt.datetime(stop_date.year, 5, 12, 6, 0, 0)

    # Convert Epoch_A and Epoch_B to NumPy arrays of datetimes
    Epoch_np = np.array(REPT_data[sat_select]['Epoch'].UTC)
    # Generate Lstar interval boundaries within the time range.
    time_mask = (Epoch_np >= time_start) & (Epoch_np <= time_stop)
    time_range = Epoch_np[time_mask]

    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Configure time-based colormap
    colormap_name = 'viridis'
    cmap = plt.cm.get_cmap(colormap_name)
    time_range_timestamps = mdates.date2num(time_range)
    vmin = mdates.date2num(time_start)
    vmax = mdates.date2num(time_stop)
    norm = colors.Normalize(vmin=vmin,vmax=vmax)

    # Scatter plot: L* vs PSD, colored by Time to show evolution
    scatter_plot = ax.scatter(
        sat_data['Lstar'][time_mask,i_K],
        sat_data['PSD'][k].values[:,i_mu][time_mask],
        c=time_range_timestamps, # Color by Epoch datetime objects
        cmap=cmap,
        norm=norm,
        marker='o')

    # Format Colorbar (Time)
    cbar = fig.colorbar(scatter_plot, ax=ax, orientation='horizontal', pad=0.15)
    cbar.set_label('Time (UTC)', fontsize=textsize)
    cbar.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    cbar.ax.tick_params(labelsize=textsize-2)

    # Axis Limits and Log Scale
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlim(3.4, 5)
    ax.set_xlabel(r"L*", fontsize=textsize)
    ax.set_ylim(min_val, max_val)
    ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize)
    plt.yscale('log')
    ax.grid(True)

    # Annotate K/Mu values and Satellite Legend
    ax.text(0.02, 0.98, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$",
            transform=ax.transAxes, fontsize=textsize-2, verticalalignment='top') #add the text

    if sat_select == 'rbspa': rbsp_label = 'RBSP-A'
    elif sat_select == 'rbspb': rbsp_label = 'RBSP-B'

    # Create generic legend handle for the satellite markers
    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=10, label=rbsp_label)
    # Create the first legend (for RBSP-B and GPS)
    ax.legend(handles=[handle_rbsp],
                title = 'Satellite',
                title_fontsize = textsize-2,
                loc='upper right',
                bbox_to_anchor=(1.15, 1.0),
                handlelength=1,
                fontsize=textsize-4)

    # Set the plot title to the time interval
    title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title_str, fontsize = textsize)
    plt.show()

#%% --- Plot 6: PSD Radial Profiles with Averaging (L* Binned) ---
if plot_radial_Lstar==True:
    # Select RBSP satellite to plot
    sat_select = 'rbspa'
    sat_data = REPT_data[sat_select]

    # Select specific adiabatic invariant coordinates (K)
    k = 0.1

    # Set time range for the plot
    time_start, time_stop = start_date, stop_date

    time_start = dt.datetime(start_date.year, 5, 11, 0, 0, 0)
    time_stop = dt.datetime(stop_date.year, 5, 12, 0, 0, 0)

    # time_start  = dt.datetime(2017, 4, 24, 17, 7, 0)
    # time_stop   = dt.datetime(2017, 4, 24, 21, 35, 0)

    # time_start  = dt.datetime(2017, 4, 25, 15, 30, 0)
    # time_stop   = dt.datetime(2017, 4, 25, 19, 57, 0)

    # Define specific time window for the event
    time_mask = (sat_data['Epoch'] >= time_start) & (sat_data['Epoch'] <= time_stop)

    # Define L* bins for averaging
    lstar_delta = 0.1
    lstar_range = sat_data['Lstar'][:,0][time_mask]
    # Filter valid L* values to determine range
    valid_lstar = lstar_range[(lstar_range > 0) & (~np.isnan(lstar_range))]
    lstar_min, lstar_max = np.min(valid_lstar), np.max(valid_lstar)
    
    # Create bin edges
    lstar_intervals = np.arange(np.floor(lstar_min / lstar_delta) * lstar_delta, 
                                np.ceil(lstar_max / lstar_delta) * lstar_delta + lstar_delta, lstar_delta)

    # Initialize arrays for averaged PSD results
    averaged_psd = np.zeros((len(lstar_intervals), len(sat_data['PSD'][k].values[:,0].flatten())))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Generate distinct colors for each Mu channel (Modified Spectral + Teal override)
    color_set = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(Mu_set), dtype=int)]
    color_set[3] = [0, 1, 1, 1]  # Teal

    # --- Binning and Averaging Loop ---
    for mu_index in range(len(Mu_set)):
        for i, lstar_val in enumerate(lstar_intervals):
            # Define bin boundaries
            lstar_start = lstar_val - 1/2 * lstar_delta
            lstar_end = lstar_val + 1/2 * lstar_delta
            # Find indices in time/L* range
            interval_indices = np.where(time_mask & 
                                        (sat_data['Lstar'][:,0] >= lstar_start) & 
                                        (sat_data['Lstar'][:,0] < lstar_end))[0]
            
            # Compute mean PSD for this bin (ignoring NaNs)
            if len(interval_indices) > 0:
                averaged_psd[i, mu_index] = np.nanmean(sat_data['PSD'][k].values[interval_indices, mu_index])
        
        # Create a mask to filter out NaN values
        psd_mask = (averaged_psd[:,mu_index] > 0) & (averaged_psd[:,mu_index] != np.nan)
        
        # Plot averaged profile for this Mu channel
        ax.plot(lstar_intervals[psd_mask], averaged_psd[psd_mask,mu_index],
            color=color_set[mu_index], linewidth=2, marker='o', markersize=4,
            label=f"{Mu_set[mu_index]:.0f}")

    # Formatting
    ax.set_xlim(3, 5.5)
    ax.set_xlabel(r"L*", fontsize=textsize - 2)
    ax.set_ylim(1e-13, 1e-5)
    ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize - 2)
    plt.yscale('log')
    ax.grid(True)

    # Legend and Annotations
    ax.legend(title=r"$\mu$ (MeV/G)", loc='center right', bbox_to_anchor=(1.25, 0.5),
              fontsize='small', title_fontsize='medium', markerscale=0.7)
    # Add K text to the plot
    ax.text(0.02, 0.98, r"K = " + f"{k:.1f} $G^{{1/2}}R_E$", 
            transform=ax.transAxes, fontsize=textsize-4, verticalalignment='top') #add the text

    # Set the plot title to the time interval
    title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title_str)

    plt.tight_layout()
    plt.show()
