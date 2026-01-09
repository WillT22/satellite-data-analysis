#%% Importing relevant libraries
import os
import sys
import time
import datetime as dt
import importlib
import numpy as np
import scipy.constants as sc
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.animation as animation

# Add current directory to path for local imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_script_dir)

# --- Import Custom Modules ---
import GPS_PSD_func
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import (QinDenton_period, import_GPS, data_period, data_from_gps, 
                          find_Loss_Cone, load_data, AlphaOfK, EnergyofMuAlpha, 
                          energy_spectra, find_psd, find_Lstar)

import Zhao_2018_PAD_Model
importlib.reload(Zhao_2018_PAD_Model)
from Zhao_2018_PAD_Model import (import_zhao_coeffs, find_Zhao_PAD_coeffs, 
                                 create_PAD, PAD_Scale_Factor)

#%% Global Variables
textsize = 22
Re = 6378.137 #Earth's Radius

# Adiabatic Invariant Targets
Mu_set = np.array((2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G (1st Invariant)
K_set = np.array((0.1, 1, 2)) # R_E*G^(1/2) (2nd Invariant)

# Workflow Control
mode = 'load'          # 'save' (calculate & save) or 'load' (load existing npz)
storm_name = 'sep2019storm' 
extMag = 'TS04'        # Magnetic Model: 'T89c' or 'TS04'

plot_flux = True
plot_flux_all=True
plot_processes_flux = True
plot_psd = True
plot_combined_psd = True
plot_energies = True
PAD_calculate = True
plot_PAD = True
if plot_PAD == True:
    PAD_calculate = True
plot_radial = True
plot_radial_dynamic = True
SHOW_GPS_DATA = True

# Data Paths
GPS_data_root = '/home/wzt0020/sat_data_analysis/GPS_data/'
input_folder = os.path.join(GPS_data_root, storm_name)
base_save_folder = os.path.join(GPS_data_root, storm_name)

# --- Storm Date Definitions (Dictionary Map) ---
storm_dates = {
    'april2017storm':   (dt.datetime(2017, 4, 21), dt.datetime(2017, 4, 26)),
    'aug2018storm':     (dt.datetime(2018, 8, 25), dt.datetime(2018, 8, 28)),
    'oct2012storm':     (dt.datetime(2012, 10, 7), dt.datetime(2012, 10, 11)),
    'latefeb2019storm': (dt.datetime(2019, 2, 27), dt.datetime(2019, 3, 4)),
    'may2019storm':     (dt.datetime(2019, 5, 10), dt.datetime(2019, 5, 17)),
    'sep2019storm':     (dt.datetime(2019, 8, 31), dt.datetime(2019, 9, 3))
}

if storm_name in storm_dates:
    start_date, stop_date = storm_dates[storm_name]
else:
    raise ValueError(f"Storm '{storm_name}' not defined in date dictionary.")

# Physical Constants
# electron mass in MeV is (m_e [kg] * c^2 [m^2/s^2]) [J] / (sc.eV [J/eV] * 10^6 [eV/MeV])
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6) # this is m_0*c^2
# b_satellite and b_equator are in Gauss: 1 G = 10^5 nT

start_time = time.perf_counter()

# --- Load External Models ---
# Qin-Denton OMNI data for magnetic field modeling
QD_storm_data = QinDenton_period(start_date, stop_date)
# Zhao 2018 PAD model coefficients
Zhao_coeffs = import_zhao_coeffs()

#%% Main
if __name__ == '__main__':

    ### 1. Data Ingestion ###
    # Be mindful of ns60 and ns69 data as they have poorer fits and more noise
    raw_save_path = os.path.join(base_save_folder, 'raw_gps.npz')
    if mode == 'save':
        loaded_data = import_GPS(input_folder)
        print("Saving Raw GPS Data...")
        np.savez(raw_save_path, **loaded_data)
        print("Data Saved \n")
    elif mode == 'load':
        raw_data_load = np.load(raw_save_path, allow_pickle=True)
        loaded_data = load_data(raw_data_load)
        raw_data_load.close()
        del raw_data_load
    
    ### 2. Preprocessing & Filtering ###
    # Restrict to time window, convert coordinates to GSM, filter by L-shell/Quality
    processed_save_path = os.path.join(base_save_folder, 'processed_gps.npz')
    
    if mode == 'save':
        storm_data_raw = {}
        for satellite, sat_data in loaded_data.items():
            print(f'Restricting Time Period for satellite {satellite}', end='\r')
            storm_data_raw[satellite] = data_period(sat_data, start_date, stop_date)
        del loaded_data

        print('\nProcessing Data for each Satellite (L-shell & Efit filtering)...')
        storm_data = data_from_gps(storm_data_raw, Lshell=6)
        del storm_data_raw

        print("Saving Processed GPS Data...")
        np.savez(processed_save_path, **storm_data)
        print("Data Saved \n")

    elif mode == 'load':
        storm_data_load = np.load(processed_save_path, allow_pickle=True)
        storm_data = load_data(storm_data_load)
        storm_data_load.close()
        del storm_data_load

    ### 3. Calculate Equatorial Pitch Angles (Alpha) ###
    # Determine the pitch angle required to conserve K (2nd Invariant)
    alphaofK_filename = f"alphaofK_{extMag}.npz"
    alphaofK_save_path = os.path.join(base_save_folder, alphaofK_filename)

    if mode == 'save':
        alphaofK = {}
        for satellite, sat_data in storm_data.items():
            print(f"Calculating Pitch Angle for satellite {satellite}", end='\r')
            alphaofK[satellite] = AlphaOfK(sat_data, K_set, extMag=extMag)

        print("Saving AlphaofK Data...")
        np.savez(alphaofK_save_path, **alphaofK)
        print("Data Saved \n")
        
    elif mode == 'load':   
        alphaofK_load = np.load(alphaofK_save_path, allow_pickle=True)
        alphaofK = load_data(alphaofK_load)
        # Restore DataFrame structure lost in npz save
        for satellite, sat_data in storm_data.items():
            epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
            alphaofK[satellite] = pd.DataFrame(alphaofK[satellite], index=epoch_str, columns=K_set)
        alphaofK_load.close()
        del alphaofK_load
    
    ### 4. Calculate Energies ###
    # Determine the Energy required to conserve Mu (1st Invariant) at the calculated Alpha
    energyofmualpha = {}
    energyofmualpha_filename = f"energyofmualpha_{extMag}.npz"
    energyofmualpha_save_path = os.path.join(base_save_folder, energyofmualpha_filename)
    
    for satellite, sat_data in storm_data.items():
        print(f"Calculating Energy of Mu and Alpha for satellite {satellite}", end='\r')
        energyofmualpha[satellite] = EnergyofMuAlpha(sat_data, Mu_set, alphaofK[satellite])
    
    if mode == 'save':
        print("\nSaving Energy Data...")
        np.savez(energyofmualpha_save_path, **energyofmualpha)
        print("Data Saved \n")

    ### 5. Calculate Omnidirectional Flux at Target Coordinates ###
    # Interpolate/Fit instrument spectrum to the specific Energies calculated above
    flux_energyofmualpha = {}
    for satellite, sat_data in storm_data.items():
        print(f"Calculating Energy Spectra for satellite {satellite}", end='\r')
        flux_energyofmualpha[satellite] = energy_spectra(sat_data, energyofmualpha[satellite])

    ### 6. Pitch Angle Distribution (PAD) Modeling ###
    PAD_filename = f"PAD_model_{extMag}.npz"
    PAD_save_path = os.path.join(base_save_folder, PAD_filename)
    
    # Generate Zhao 2018 PAD models for the specific conditions
    if PAD_calculate:    
        if mode == 'save':
            PAD_models = {}
            for satellite, sat_data in storm_data.items():
                print(f"Modeling PAD for satellite {satellite}", end='\r')
                PAD_models[satellite] = create_PAD(sat_data, QD_storm_data, energyofmualpha[satellite], extMag)

            print("\nSaving GPS PAD Model Data...")
            np.savez(PAD_save_path, **PAD_models)
            print("Data Saved \n")

        elif mode == 'load': 
            PAD_models_load = np.load(PAD_save_path, allow_pickle=True)
            PAD_models = load_data(PAD_models_load)
            PAD_models_load.close()
            del PAD_models_load

    ### 7. Calculate Geometric Scale Factor ###
    # Ratio: Model Value (at Alpha) / Integrated Model Flux
    # Used to convert Omnidirectional Flux -> Directional Flux
    scale_factor = {}
    PAD_int = {}
    for satellite, sat_data in storm_data.items():
        print(f"Calculating Scale Factor for satellite {satellite}", end='\r')
        # Returns Tuple: (Scale Factor, Integral)
        scale_factor[satellite] = PAD_Scale_Factor(sat_data, QD_storm_data, energyofmualpha[satellite], alphaofK[satellite], extMag) 
    print('Scale Factor Calculated\n')

    ### 8. Calculate Final Directional Flux ###
    # Directional Flux = Omni_Flux * (Model_Value / Model_Integral) * Geometry_Factors
    flux = {}
    for satellite, sat_data in storm_data.items():
        flux[satellite] = {}
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        
        for i_K, K_value in enumerate(K_set):
            # Scale Factor Application:
            flux_mag = flux_energyofmualpha[satellite][K_value].values
            scale_val = scale_factor[satellite][K_value].values
            
            # Apply scaling
            # 2 * 2 * pi: accounts for 2*hemispheric detection (Hemispheric -> Directional normalization)
            directional_flux = flux_mag * (4 * np.pi) * scale_val
            
            flux[satellite][K_value] = pd.DataFrame(directional_flux, index=epoch_str, columns=Mu_set)

    ### 9. Calculate Phase Space Density (PSD) ###
    # PSD = Flux / p^2 (with relativistic corrections)
    psd = {}
    for satellite, sat_data in storm_data.items():
        print(f"Calculating PSD for satellite {satellite}", end='\r')
        psd[satellite] = find_psd(flux[satellite], energyofmualpha[satellite])

    ### 10. Calculate L* (Roederer L) ###
    # Computationally expensive tracing step
    complete_filename = f"storm_data_{extMag}.npz"
    complete_save_path = os.path.join(base_save_folder, complete_filename)
    
    if mode == 'save':
        for satellite, sat_data in storm_data.items():
            print(f"Calculating L* for satellite {satellite}")
            storm_data[satellite] = find_Lstar(sat_data, alphaofK[satellite], extMag=extMag)

        print("\nSaving Final Processed GPS Data...")
        np.savez(complete_save_path, **storm_data)
        print("Data Saved \n")
        
    elif mode == 'load': 
        complete_load = np.load(complete_save_path, allow_pickle=True)
        storm_data = load_data(complete_load)
        complete_load.close()
        del complete_load

    # --- Runtime Statistics ---
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    def format_runtime(elapsed_time):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        return f"Script runtime: {hours}h {minutes}m {seconds:.2f}s"

    print(format_runtime(elapsed_time))

#%% Plot 1: GPS Flux for a Single Energy Channel
if plot_flux:
    # 1. Setup Parameters
    energy = 2.1 # MeV
    # Find index of closest energy channel
    energy_channels = storm_data[list(storm_data.keys())[0]]['Energy_Channels']
    i_energy = np.argmin(np.abs(energy_channels - energy))

    k = 0.1
    i_K = np.where(K_set == k)[0]

    # 2. Configure Limits
    min_val = np.nanmin(np.log10(1e2))
    max_val = np.nanmax(np.log10(1e6))
    extMag_label = 'T89' if extMag == 'T89c' else extMag

    # 3. Create Plot
    fig, ax = plt.subplots(figsize=(16, 4))
    
    for satellite, sat_data in storm_data.items():     
        # Filter valid flux data
        flux_plot = sat_data['electron_diff_flux'][:, i_energy]
        flux_mask = (flux_plot > 0) & (~np.isnan(flux_plot))

        # Scatter Plot: Time vs L, colored by log10(Flux)
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[flux_mask], 
                               sat_data[f'L_LGM_{extMag_label}IGRF'][flux_mask],
                               c=np.log10(flux_plot[flux_mask]), 
                               vmin=min_val, vmax=max_val)

    # 4. Format Axes
    ax.set_title(f"GPS CXD, {energy} MeV Electron Differential Flux", fontsize=textsize + 2)
    ax.set_ylabel(r"McIlwain L", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # Calculate Axis Limits (Midnight to Midnight)
    min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    ax.set_ylim(3, 6)
    
    # Time formatting
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    ax.grid(True)

    # 5. Add Colorbar
    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, 
                        format=ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    cbar.set_label(label=r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)
    plt.show()

#%% Plot 2: Flux from REPT and CXD for All Energy Channels
if plot_flux_all:
    # 1. Load Reference REPT Data
    # Load the specific REPT processed file for this storm
    save_path = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'rept_data_{extMag}.npz')
    complete_load = np.load(save_path, allow_pickle=True)
    REPT_data = load_data(complete_load)
    complete_load.close()
    del complete_load
    
    # 2. Select Energy Channels
    # Filter for energies < 4 MeV to focus on the core radiation belt population
    energy_channels = REPT_data[list(REPT_data.keys())[0]]['Energy_Channels']
    energy_channels = energy_channels[energy_channels < 4]

    time_start = start_date
    time_stop = stop_date
    extMag_label = 'T89' if extMag == 'T89c' else extMag

    # 3. Setup Multi-Panel Plot
    # Create a vertical stack of subplots: one per energy channel + one for DST
    fig, axes = plt.subplots(len(energy_channels) + 1, 1, figsize=(24, 10), sharex=True, sharey=False)
    
    colormap_name = 'viridis'
    cmap = plt.cm.get_cmap(colormap_name)
    
    # Loop through each energy channel (Top N subplots)
    for i_energy, energy in enumerate(energy_channels):
        ax = axes[i_energy]
        
        # --- A. Plot REPT Data (Background/Reference) ---
        for satellite, sat_data in REPT_data.items():
            # Time Masking
            sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
            
            # Extract and Average Flux
            flux_slice = sat_data['FEDU_averaged'][sat_iepoch_mask, :, i_energy]
            # Replace invalid values with NaN before averaging
            flux_temp_mask = np.where(flux_slice >= 0, flux_slice, np.nan)
            # Average over pitch angles (axis 1) and divide by 2 (hemispheric to omni approx)
            flux_plot = np.nanmean(flux_temp_mask, axis=1) / 2
            
            # Filter Valid Data
            flux_mask = (flux_plot > 0) & (flux_plot != np.nan)
            
            # Reconstruct combined mask to match full array length
            combined_mask = np.zeros_like(sat_iepoch_mask, dtype=bool)
            combined_mask[sat_iepoch_mask] = flux_mask
            
            # Set fixed max for colorbar consistency
            vmax = 7 
            
            # Plot REPT Scatter (Circles)
            scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], 
                                   sat_data[f'L_LGM_{extMag_label}IGRF'][combined_mask],
                                   c=np.log10(flux_plot[flux_mask]), 
                                   cmap=cmap, vmin=0, vmax=vmax, zorder=2)

        # --- B. Plot GPS CXD Data (Overlay) ---
        target_K_set = [0.1] # Use K=0.1 for comparison
        
        for satellite, sat_data in storm_data.items():    
            # Prepare Input for Spectral Fit
            energy_input = {}
            sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
            epoch_index = sat_data['Epoch'].UTC
            
            # Create dummy energy series for the target energy
            for k in target_K_set:
                energy_input[k] = {}
                energy_input[k][energy] = pd.Series(
                    data=np.full(len(epoch_index), energy), 
                    index=epoch_index
                )
            
            # Calculate Flux at Target Energy using Spectral Fit
            flux_result = energy_spectra(sat_data, energy_input)
            flux_plot = flux_result[target_K_set[0]][energy]
            
            # Filter Valid Flux
            flux_mask = (flux_plot > 0) & (flux_plot != np.nan)

            # Plot GPS Scatter (Stars)
            scatter_B = ax.scatter(sat_data['Epoch'].UTC[flux_mask], 
                                   sat_data[f'L_LGM_{extMag_label}IGRF'][flux_mask],
                                   marker='*', s=80, alpha=0.7,
                                   c=np.log10(flux_plot[flux_mask]), 
                                   vmin=0, vmax=vmax, zorder=1)

        # --- C. Subplot Formatting ---
        ax.set_title(f"{energy:.2f} MeV", fontsize=textsize+4)
        ax.tick_params(axis='both', labelsize=textsize, pad=5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_ylim(3, 6.5)
        ax.grid(True)

    # --- D. Colorbar Setup ---
    # Add a dedicated axis for the colorbar on the right
    cbar_ax = fig.add_axes([0.96, 0.27, 0.02, 0.61]) 
    cbar = fig.colorbar(scatter_A, cax=cbar_ax, 
                        format=ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    cbar.set_label(label=r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize, labelpad=5)
    cbar.ax.tick_params(labelsize=textsize - 2)

    # --- E. Plot DST Index (Bottom Subplot) ---
    ax = axes[-1]
    QD_dates_array = np.array(QD_storm_data['DateTime'])
    # Mask DST data to time range
    iepoch_mask = (QD_dates_array >= time_start) & (QD_dates_array <= time_stop)
    
    ax.plot(QD_dates_array[iepoch_mask], QD_storm_data['Dst'][iepoch_mask], color='black')
    
    # DST Formatting
    ax.tick_params(axis='both', labelsize=textsize, pad=5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    
    # Calculate nice time bounds (rounded to 12 hours)
    min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    
    # Add zero line for DST
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0)
    ax.set_ylabel(r'DST (nT)', fontsize=textsize)
    ax.grid(True)

    # --- F. X-Axis Formatting (Bottom Plot Only) ---
    # Since sharex=True, setting this on the last plot affects the timeline for all
    if i_energy >= len(energy_channels)-1:
        ax.set_xlabel('Time (UTC)', fontsize=textsize+2, labelpad=2)
        ax.set_xlim(time_start, time_stop)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        ax.tick_params(axis='x', labelsize=textsize+2, pad=12)

    # --- G. Global Labels and Legends ---
    fig.text(0.08, 0.575, r'McIlwain L', fontsize=textsize+2, rotation='vertical', va='center')

    # Custom Legend
    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=10, label='RBSP') 
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None',
                               markersize=12, label='GPS') 

    # Create the first legend (for RBSP-B and GPS)
    legend1 = fig.legend(handles=[handle_rbsp, handle_gps],
                    title = 'Satellite',
                    title_fontsize = textsize,
                    loc='upper right',
                    bbox_to_anchor=(1.03, 1.05),
                    handlelength=1,
                    fontsize=textsize-2)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(right=0.95, hspace=0.35)
    plt.show()
    
#%% Plot 3: Phase Space Density (PSD)
if plot_psd:
    from matplotlib import colors
    
    # 1. Setup Parameters
    # Select specific adiabatic coordinates (K, Mu)
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]
    
    # Magnetic field model label
    extMag_label = 'T89' if extMag == 'T89c' else extMag

    # 2. Configure Plot
    fig, ax = plt.subplots(figsize=(16, 4))

    # Custom Colormap for High Dynamic Range PSD
    colorscheme = plt.cm.get_cmap('turbo')(np.linspace(0, 0.85, 256))
    cmap = colors.ListedColormap(colorscheme)

    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1e-12))
    max_val = np.nanmax(np.log10(1e-5))

    # 3. Plot Data
    for satellite, sat_data in storm_data.items():
        # Extract PSD for the specific K/Mu channel
        psd_plot = psd[satellite][k].values[:,i_mu].copy().flatten()
        
        # Mask invalid data
        psd_mask = (psd_plot > 0) & (~np.isnan(psd_plot))
        
        # Scatter Plot: Time vs L*, colored by log10(PSD)
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[psd_mask], 
                               sat_data['Lstar'][psd_mask,i_K],
                               c=np.log10(psd_plot[psd_mask]), 
                               cmap=cmap, vmin=min_val, vmax=max_val, marker='*', s=80, alpha=0.7)

    # 4. Format Axes
    ax.set_title(f"GPS CXD, K={k:.1f} $G^{{1/2}}R_E$, $\\mu$={mu:.0f} $MeV/G$", fontsize=textsize + 2)
    ax.set_ylabel(r"L*", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # Force X-axis (Time) limits based on storm start/stop
    min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    ax.set_ylim(3, 5.5)
    
    # Time Formatting
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    ax.grid(True)

    # 5. Add Colorbar
    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, 
                        format=ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    
    # Ticks for every order of magnitude
    tick_locations = np.arange(min_val, max_val + 1)
    cbar.set_ticks(tick_locations)
    cbar.set_label(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)
    plt.show()

#%% Plot 4: Combined PSD with REPT data
if plot_combined_psd:
    # 1. Setup Parameters
    k = 0.1   
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    # Time range
    time_start, time_stop = start_date, stop_date

    # 2. Configure Colormap (Turbo for High Contrast)
    min_val = np.nanmin(np.log10(1e-12))
    max_val = np.nanmax(np.log10(1e-5))

    colorscheme = plt.cm.get_cmap('turbo')(np.linspace(0, 0.85, 256))
    cmap = colors.ListedColormap(colorscheme)

    # 3. Load REPT Data
    # Assuming REPT data needs to be loaded separately for this combined plot
    # Ideally this would be done once at the top, but following the logic provided
    save_path = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'rept_data_{extMag}.npz')
    complete_load = np.load(save_path, allow_pickle=True)
    REPT_data = load_data(complete_load)
    complete_load.close()
    del complete_load

    # 4. Create Plot
    # Wide aspect ratio
    fig, ax = plt.subplots(figsize=(24, 2.5)) 

    # Plot REPT Data (Circles, zorder=2 to be on top of grid but below GPS if needed)
    for satellite, sat_data in REPT_data.items():
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        
        # Extract PSD
        psd_plot = REPT_data[satellite]['PSD'][k].values[:,i_mu].copy().flatten()
        
        # Mask
        psd_mask = (psd_plot > 0) & (~np.isnan(psd_plot))
        lstar_mask = sat_data['Lstar'][:,0] > 0
        combined_mask = psd_mask & lstar_mask & sat_iepoch_mask
        
        # Plot
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], 
                               sat_data['Lstar'][combined_mask,i_K],
                               c=np.log10(psd_plot[combined_mask]), 
                               cmap=cmap, vmin=min_val, vmax=max_val, zorder=2)

    # Plot GPS Data (Stars, zorder=1 to be behind REPT or distinct)
    for satellite, sat_data in storm_data.items():
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        
        # Extract PSD
        psd_plot = psd[satellite][k].values[:,i_mu].copy().flatten()
        
        # Mask
        psd_mask = (psd_plot > 0) & (~np.isnan(psd_plot))
        combined_mask = psd_mask & sat_iepoch_mask
        
        # Plot
        scatter_B = ax.scatter(sat_data['Epoch'].UTC[combined_mask], 
                               sat_data['Lstar'][combined_mask,i_K], 
                               marker='*', s=80, alpha=0.7,
                               c=np.log10(psd_plot[combined_mask]), 
                               cmap=cmap, vmin=min_val, vmax=max_val, zorder=1)

    # 5. Format Axes
    ax.set_title(f"RBSP REPT & GPS CXD Phase Space Density, K={k:.1f} $G^{{1/2}}R_E$, $\\mu$={mu:.0f} $MeV/G$", 
                 fontsize=textsize+10, y=1.1)
    ax.set_ylabel(r"L*", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    
    # Time Axis
    ax.set_xlim(time_start, time_stop)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_ylim(3.6, 5.4)
    ax.grid(True)

    # 6. Add Colorbar
    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, 
                        format=ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    
    tick_locations = np.arange(min_val, max_val + 1)
    cbar.set_ticks(tick_locations)
    cbar.set_label(r"PSD $(c/MeV/cm)^3$", fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    # 7. Legend
    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=10, label='RBSP') 
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None',
                               markersize=12, label='GPS') 
    
    legend1 = ax.legend(handles=[handle_rbsp, handle_gps],
                        title = 'Satellite',
                        title_fontsize = textsize,
                        loc='upper right',
                        bbox_to_anchor=(1.09, 1.85),
                        handlelength=1,
                        fontsize=textsize-2)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=1, right=0.95)

    plt.show()

#%% Plot 5: Energies corresponding to Mu and K across L*
if plot_energies:
    # 1. Setup Parameters
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    time_start, time_stop = start_date, stop_date

    # 2. Configure Colormap (Time)
    colormap_name = 'plasma'
    plasma = plt.cm.get_cmap(colormap_name)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('truncated_plasma',plasma(np.linspace(0, 0.9, 256)))
    
    vmin = mdates.date2num(time_start)
    vmax = mdates.date2num(time_stop)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # 3. Plot Data
    fig, ax = plt.subplots(figsize=(20, 8))
    
    for satellite, sat_data in storm_data.items():
        # Masking
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        
        # Get energy values for the specific Mu/K
        energy_plot = energyofmualpha[satellite][k].values[:, i_mu].copy().flatten()
        
        # Valid Data Mask
        energy_mask = (energy_plot > 0) & (~np.isnan(energy_plot))
        lstar_mask = (sat_data['Lstar'][:, i_K] > 0).flatten()
        combined_mask = energy_mask & sat_iepoch_mask & lstar_mask
        
        # Scatter Plot: L* vs Energy, colored by Time
        scatter_plot = ax.scatter(sat_data['Lstar'][combined_mask, i_K], 
                                  energy_plot[combined_mask], 
                                  c=mdates.date2num(sat_data['Epoch'].UTC[combined_mask]), 
                                  cmap=cmap, vmin=vmin, vmax=vmax)
    
    # 4. Format Plot
    # Colorbar
    cbar = fig.colorbar(scatter_plot, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Time (UTC)', fontsize=textsize)
    cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    cbar.ax.tick_params(labelsize=textsize-2)

    # Annotations
    ax.text(0.5, 0.92, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$", 
            transform=ax.transAxes, fontsize=textsize) 

    # Axes
    ax.set_xlim(3.8, 5.2)
    ax.set_ylim(1, 5)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlabel(r"L*", fontsize=textsize)
    ax.set_ylabel(r"Energy (MeV)", fontsize=textsize)
    ax.grid(True)
    
    plt.show()


#%% Plot 6: Pitch Angle Distribution (PAD) Comparison
if plot_PAD:
    # 1. Setup Parameters
    # Define a specific time snapshot to analyze detailed PAD structure
    # latefeb2019, PAD poor fit 12:30-13:30, 18-22
    # sep2019, PAD poor fit 14:30-17
    time_select = dt.datetime(start_date.year, 8, 31, 8, 30, 0)
    
    # Select satellite and invariant coordinates
    sat_select = 'rbspa'
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    # Time window for GPS data collection around the snapshot
    time_window = dt.timedelta(minutes=30)
    
    # Y-axis limits for flux
    lower_bound = 1e3
    upper_bound = 1e6

    # Labels
    if sat_select == 'rbspa': rbsp_label = 'RBSP-A'
    elif sat_select == 'rbspb': rbsp_label = 'RBSP-B'
    extMag_label = 'T89' if extMag == 'T89c' else extMag

    # 2. Load REPT Data for Comparison
    # Load Main Data
    save_path = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'rept_data_{extMag}.npz')
    complete_load = np.load(save_path, allow_pickle=True)
    REPT_data = load_data(complete_load)
    complete_load.close()
    del complete_load

    # Load Energy/Mu Data
    energyofmualpha_REPT_save = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'energyofmualpha_{extMag}.npz')
    complete_load = np.load(energyofmualpha_REPT_save, allow_pickle=True)
    REPT_energyofmualpha = load_data(complete_load)
    complete_load.close()
    del complete_load

    # Load Alpha/K Data
    alphaofK_REPT_save = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'alphaofK_{extMag}.npz')
    alphaofK_load = np.load(alphaofK_REPT_save, allow_pickle=True)
    REPT_alphaofK = load_data(alphaofK_load)
    # Restore DataFrame structure
    for satellite, sat_data in REPT_data.items():
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        REPT_alphaofK[satellite] = pd.DataFrame(REPT_alphaofK[satellite], index=epoch_str, columns=K_set)
    alphaofK_load.close()
    del alphaofK_load

    # 3. Find Nearest REPT Epoch
    nearest_it_REPT = np.argmin(np.abs(REPT_data[sat_select]['Epoch'].UTC - time_select))
    nearest_time_REPT = REPT_data[sat_select]['Epoch'].UTC[nearest_it_REPT]

    # Get DST for this time (for Zhao model lookup)
    QD_dates_array = np.array(QD_storm_data['DateTime'])
    DST_it_time = np.argmin(np.abs(QD_dates_array - nearest_time_REPT))
    DST_at_time = QD_storm_data['Dst'][DST_it_time]
    
    print(f'DST at time {nearest_time_REPT.strftime("%Y-%m-%d %H:%M")} is {DST_at_time} nT')
    
    if DST_at_time > -20: i_dst = 'Dst > -20 nT'
    elif DST_at_time <= -20 and DST_at_time > -50: i_dst = '-50 nT < Dst < -20 nT'
    elif DST_at_time <= -50: i_dst = 'Dst < -50 nT'

    # 4. Get REPT Parameters at Snapshot
    energy_at_mutime = REPT_energyofmualpha[sat_select][k][mu].iloc[nearest_it_REPT]
    nearest_REPT_ienergy = np.argmin(np.abs(REPT_data[sat_select]['Energy_Channels'][0:6] - energy_at_mutime))
    nearest_REPT_energy = REPT_data[sat_select]['Energy_Channels'][nearest_REPT_ienergy]

    # Get Binning Info from Zhao Coeffs (Need to access structure via energy/dst)
    # Note: Accessing nested dictionary structure manually here
    # Ideally, use a helper function, but following provided logic
    MLT_bins = Zhao_coeffs[nearest_REPT_energy][i_dst]['c2']['MLT_values']
    MLT_ref = REPT_data[sat_select]['MLT'][nearest_it_REPT]
    MLT_ibin = np.argmin(np.abs(MLT_bins - MLT_ref))
    MLT_bin = MLT_bins[MLT_ibin]

    L_bins = Zhao_coeffs[nearest_REPT_energy][i_dst]['c2']['L_values']
    L_ref = REPT_data[sat_select][f'L_LGM_{extMag_label}IGRF'][nearest_it_REPT]
    L_ibin = np.argmin(np.abs(L_bins - L_ref))
    L_bin = L_bins[L_ibin]

    print(f'{sat_select} found at MLT = {MLT_ref:.2f} and L = {L_ref:.2f} with Energy = {nearest_REPT_energy:.2f}')

    # 5. Process REPT Pitch Angles
    REPT_PA_local = REPT_data[sat_select]['Pitch_Angles']
    # Calculate Equatorial Pitch Angle from Local Measurement
    # Mapping formula: sin(alpha_eq) = sqrt(B_min / B_sat) * sin(alpha_local)
    # Note: arcsin requires argument <= 1. If B_min/B_sat ratio is off due to model/measurement mismatch, this can fail.
    # Added clipping for robustness
    arg_val = np.sin(np.deg2rad(REPT_PA_local))**2 * (REPT_data[sat_select]['b_min'][nearest_it_REPT] / 
                                                      REPT_data[sat_select]['b_satellite'][nearest_it_REPT])
    arg_val = np.clip(arg_val, 0, 1) # Prevent NaNs
    REPT_PA_eq = np.rad2deg(np.arcsin(np.sqrt(arg_val)))
    
    # Create symmetric array (0-180)
    REPT_PA_eq = np.unique(np.concatenate((REPT_PA_eq, 180 - REPT_PA_eq)))
    REPT_PA_local90 = REPT_PA_eq[len(REPT_PA_local)-1]
    
    # Get Flux Data and Mirror it
    REPT_PAD = REPT_data[sat_select]['FEDU'][nearest_it_REPT, :, nearest_REPT_ienergy]
    # Insert the 90-degree value to handle symmetry boundary
    REPT_PAD = np.insert(REPT_PAD, len(REPT_PA_local), REPT_PAD[len(REPT_PA_local)-1])

    # 6. Generate REPT Model for Snapshot
    # Extract single-epoch data structures to feed into create_PAD
    REPT_data_epoch = {}
    for key, key_data in REPT_data[sat_select].items():
        if key in ['Energy_Channels', 'Pitch_Angles']:
            REPT_data_epoch[key] = REPT_data[sat_select][key]
        elif key in ['FEDU', 'FEDU_averaged']:
            REPT_data_epoch[key] = REPT_data[sat_select][key][nearest_it_REPT,:,:]
        elif key == 'Mu_calc': continue
        elif key == 'PSD':
            REPT_data_epoch[key] = {}
            REPT_data_epoch[key][k] = pd.DataFrame(
                REPT_data[sat_select]['PSD'][k][mu].values[nearest_it_REPT], 
                index=[nearest_time_REPT], columns=[mu]
            )
        elif key == 'Lstar':
            REPT_data_epoch[key] = REPT_data[sat_select][key][nearest_it_REPT, i_K]
        else:
            REPT_data_epoch[key] = REPT_data[sat_select][key][nearest_it_REPT]

    # Reconstruct input dictionaries for model function
    REPT_energyofmualpha_val = REPT_energyofmualpha[sat_select][k][mu].iloc[nearest_it_REPT]
    REPT_energyofmualpha_epoch = {sat_select: {k: pd.DataFrame(REPT_energyofmualpha_val, index=[str(nearest_time_REPT)], columns=[mu])}}
    
    REPT_alpha_val = REPT_alphaofK[sat_select][k].iloc[nearest_it_REPT]
    REPT_alphaofK_epoch = {sat_select: pd.DataFrame(REPT_alpha_val, index=[str(nearest_time_REPT)], columns=[k])}
    
    PAD_models_REPT = {}
    PAD_models_REPT[sat_select] = create_PAD(REPT_data_epoch, QD_storm_data, REPT_energyofmualpha_epoch[sat_select], extMag)

    # Calculate Scaling Factor (Ratio of Measured to Model)
    Model_PA = PAD_models_REPT[sat_select][k][mu]['pitch_angles'].values[0]
    Model_closest_PA = np.argmin(np.abs(Model_PA - REPT_PA_local90))
    Model_PAD = PAD_models_REPT[sat_select][k][mu]['Model'].values[0]
    Model_PAD_scale = REPT_PAD[len(REPT_PA_local)-1] / Model_PAD[Model_closest_PA]

    # 7. Collect GPS Data for Comparison
    time_lower_bound = time_select - time_window
    time_upper_bound = time_select + time_window

    Model_GPS_PA = {}
    Model_GPS_PAD = {}
    near_time_GPS_index = {}
    near_time_GPS = {}
    
    for satellite, sat_data in storm_data.items():
        # Time Mask
        time_range_mask = (sat_data['Epoch'].UTC >= time_lower_bound) & (sat_data['Epoch'].UTC <= time_upper_bound)
        near_time_GPS_index_sat = np.where(time_range_mask)[0]
        
        # Initialize lists
        Model_GPS_PA[satellite] = []
        Model_GPS_PAD[satellite] = []
        near_time_GPS_index[satellite] = []
        near_time_GPS[satellite] = []
        
        for ii, i_time in enumerate(near_time_GPS_index_sat):
            near_time_GPS_temp = sat_data['Epoch'].UTC[i_time]

            # Get current parameters
            current_E = energyofmualpha[satellite][k][mu].iloc[i_time]
            current_L = sat_data[f'L_LGM_{extMag_label}IGRF'][i_time]
            
            # Find bins
            current_L_ibin = np.argmin(np.abs(L_bins - current_L))
            
            mlt_val = np.atleast_1d(sat_data['MLT'])[i_time]
            current_MLT = mlt_val
            current_MLT_ibin = int(((mlt_val + 1) % 24) // 2)
            
            # Check conditions (Energy, L, MLT match)
            e_mask = np.argmin(np.abs(REPT_data[sat_select]['Energy_Channels'][0:6] - current_E)) == nearest_REPT_ienergy
            l_shell_mask = current_L_ibin == L_ibin
            mlt_mask = current_MLT_ibin == MLT_ibin
            
            if e_mask and l_shell_mask and mlt_mask:
                near_time_GPS_index[satellite].append(i_time)
                near_time_GPS[satellite].append(near_time_GPS_temp)

                # Store Model Output for this specific time/location
                pa_array = PAD_models[satellite][k][mu]['pitch_angles'].values[i_time, :]
                Model_GPS_PA[satellite].append(pa_array)
                
                pad_array = PAD_models[satellite][k][mu]['Model'].values[i_time, :]
                Model_GPS_PAD[satellite].append(pad_array)
                
                print(f'GPS satellite {satellite} found at time {near_time_GPS_temp.strftime("%Y-%m-%d %H:%M")} in MLT = {current_MLT:.1f} and L = {current_L:.2f} with Energy = {current_E:.2f}')
        
        # Convert to numpy arrays if data found
        if len(Model_GPS_PA[satellite]) > 0:
            Model_GPS_PA[satellite] = np.array(Model_GPS_PA[satellite])
            Model_GPS_PAD[satellite] = np.array(Model_GPS_PAD[satellite])
        else:
             # Remove empty entries to keep iteration clean
             del Model_GPS_PA[satellite]
             del Model_GPS_PAD[satellite]
             
    if not Model_GPS_PAD:
        print(f'No GPS satellites found at this time in MLT = {MLT_bin} and L = {L_bin}')

    # 8. Create Plot
    fig, ax = plt.subplots(figsize=(9, 9))

    # A. Plot REPT Data (Reference)
    REPT_Flux_plot = ax.scatter(REPT_PA_eq[REPT_PAD > 0], REPT_PAD[REPT_PAD > 0], 
                                label=rbsp_label, zorder=3, color='black', marker='+', s=200)
    
    # B. Plot REPT Model Line
    REPT_PAD_plot = ax.plot(Model_PA, Model_PAD * Model_PAD_scale, 
                            label='RBSP Model', zorder=2, linewidth=4, alpha=0.7, linestyle='solid')

    # C. Plot GPS Data (Overlay)
    GPS_model_scale = {}
    GPS_local90 = {}
    GPS_loss_cone = {}
    GPS_alpha = {}
    
    for satellite, GPS_PAD in Model_GPS_PAD.items():
        GPS_local90[satellite] = []
        GPS_loss_cone[satellite] = []
        GPS_alpha[satellite] = []
        
        if satellite in near_time_GPS_index:
            for i, i_time in enumerate(near_time_GPS_index[satellite]):
                # Get flux scaling for this epoch
                GPS_model_scale_epoch = flux[satellite][k][mu].values[i_time]
                
                # Plot the GPS Model Scaled to Flux (Dotted Line)
                GPS_PAD_plot = ax.plot(Model_GPS_PA[satellite][i], GPS_PAD[i] * GPS_model_scale_epoch, 
                                       label=satellite, zorder=1, alpha=0.7, linewidth=3, linestyle='dotted')
                
                # Plot Vertical Lines for Context (Loss Cone, Local 90, Alpha_K)
                GPS_loss_cone_temp = storm_data[satellite]['loss_cone'][i_time]
                GPS_loss_cone[satellite].append(GPS_loss_cone_temp)

                GPS_local90_temp = storm_data[satellite]['local90PA'][i_time]
                GPS_local90[satellite].append(GPS_local90_temp)

                GPS_alpha_temp = alphaofK[satellite][k].iloc[i_time]
                GPS_alpha[satellite].append(GPS_alpha_temp)
            
                current_color = GPS_PAD_plot[0].get_color()
                
                # Plot lines on both sides (0-90 and 90-180)
                for x_val, style in [(GPS_loss_cone_temp, '-.'), (GPS_local90_temp, '-'), (GPS_alpha_temp, '--')]:
                    ax.vlines(x=x_val, ymin=0, ymax=1e8, color=current_color, linestyle=style)
                    ax.vlines(x=180 - x_val, ymin=0, ymax=1e8, color=current_color, linestyle=style)

    # 9. Format Axes and Legend
    ax.text(0.54, 0.96, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$", 
            transform=ax.transAxes, fontsize=textsize)

    # Create Custom Legend Handles
    LEGEND_GRAY = [0.6, 0.6, 0.6]
    handle_loss_cone = mlines.Line2D([], [], color=LEGEND_GRAY, linestyle='-.', 
                                     linewidth=2, label='GPS Loss Cone')
    handle_local90 = mlines.Line2D([], [], color=LEGEND_GRAY, linestyle='-', 
                                   linewidth=2, label='GPS Local 90')
    handle_alpha_k = mlines.Line2D([], [], color=LEGEND_GRAY, linestyle='--', 
                                   linewidth=2, label=r'PA at K=' + f'{k:.1f}')

    existing_handles, existing_labels = ax.get_legend_handles_labels()
    new_handles = existing_handles + [handle_loss_cone, handle_local90, handle_alpha_k]
    new_labels = existing_labels + [handle_loss_cone.get_label(), handle_local90.get_label(), handle_alpha_k.get_label()]

    ax.legend(handles=new_handles, labels=new_labels, fontsize=textsize-4, loc='lower center')

    ax.set_xlim(0, 180)
    # Dynamic Y-limits based on model range
    y_min_log = np.floor(np.log10(np.nanmin(Model_PAD * Model_PAD_scale)))
    y_max_log = np.ceil(np.log10(np.nanmax(Model_PAD * Model_PAD_scale)))
    ax.set_ylim(10**y_min_log, 10**y_max_log)
    
    plt.yscale('log')
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlabel(r"Equatorial Pitch Angle (degrees)", fontsize=textsize)
    ax.set_ylabel(r'Directional Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
    ax.grid(True)

    title_str = f"Time: {time_select.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title_str, fontsize=textsize)
    plt.show()

#%% Plot 7: PSD Radial Profiles with Averaging (L* Binned)
if plot_radial==True:
    # 1. Setup Parameters
    sat_select = 'rbspa'
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    # Constants
    gps_scale = 1 # default = 1
    MLT_range = 12 # hours, default = 3 which corresponds to +-1.5 hours
    lstar_delta = 0.1 # default = 0.1
    time_delta = 30 # minutes, default = 30

    min_val = np.nanmin(1e-9)
    max_val = np.nanmax(1e-5)

    # 2. Load REPT Data
    REPT_data_root = '/home/wzt0020/sat_data_analysis/REPT_data/'
    save_path = os.path.join(REPT_data_root, storm_name, f'rept_data_{extMag}.npz')
    complete_load = np.load(save_path, allow_pickle=True)
    REPT_data = load_data(complete_load)
    complete_load.close()
    del complete_load

    # 3. Define Time Windows
    time_start  = start_date
    time_stop   = stop_date

    # time_start = dt.datetime(start_date.year, 2, 28, 8, 0, 0)
    # time_stop = dt.datetime(stop_date.year, 3, 1, 6, 0, 0)

    time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0) # for sep2019storm
    time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0) # for sep2019storm

    # time_start = dt.datetime(start_date.year, 8, 26, 0, 0, 0) # for aug2018storm
    # time_stop = dt.datetime(stop_date.year, 8, 27, 0, 0, 0) # for aug2018storm

    # GPS Data Collection Window
    gps_time_start  = time_start
    gps_time_stop   = time_stop

    # gps_time_start = dt.datetime(start_date.year, 2, 28, 10, 40, 0)
    # gps_time_stop = dt.datetime(stop_date.year, 3, 1, 5, 0, 0)

    gps_time_start = dt.datetime(start_date.year, 8, 31, 10, 0, 0) # for sep2019storm
    gps_time_stop = dt.datetime(stop_date.year, 8, 31, 14, 0, 0) # for sep2019storm

    # gps_time_start = dt.datetime(start_date.year, 8, 26, 6, 40, 0) # for aug2018storm
    # gps_time_stop = dt.datetime(stop_date.year, 8, 26, 13, 0, 0) # for aug2018storm

    # Generate Time Steps
    time_intervals_GPS = np.arange(gps_time_start, gps_time_stop+dt.timedelta(minutes=time_delta), dt.timedelta(minutes=time_delta)).astype(dt.datetime)
    #time_intervals_GPS = time_intervals_GPS[0::2]

    # 4. Collect GPS Data
    temp_data = []
    for satellite, sat_data in storm_data.items():
        # Apply Time Mask
        window_start = time_start - dt.timedelta(minutes=time_delta-1)
        window_stop = time_stop + dt.timedelta(minutes=time_delta-1)
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= window_start) & (sat_data['Epoch'].UTC <= window_stop)

        # Extract Data for the current satellite
        sat_epoch = sat_data['Epoch'].UTC[sat_iepoch_mask]
        sat_MLT = sat_data['MLT'][sat_iepoch_mask]
        sat_Lstar = sat_data['Lstar'][sat_iepoch_mask, i_K].flatten() # Flatten Lstar to a 1D array
        
        # ASSUMPTION: 'psd' is available and correctly indexed by satellite/k/i_mu
        # We apply the mask to the psd data, and flatten it for consistency
        sat_PSD = psd[satellite][k].values[sat_iepoch_mask, i_mu].flatten()
        
        # Create a satellite name array for this block
        sat_name_array = np.full(len(sat_epoch), satellite, dtype='<U10') # Use a 10-character string dtype

        # Combine arrays column-wise (Epoch, Name, Lstar, MLT, PSD)
        # Filter out any data where Lstar or PSD is NaN (optional, but good practice for clean data)
        valid_mask = ~np.isnan(sat_Lstar) & ~np.isnan(sat_PSD)

        # Use a structured array to hold mixed data types
        combined_satellite_data = np.vstack((
            sat_epoch[valid_mask],
            sat_name_array[valid_mask],
            sat_Lstar[valid_mask],
            sat_MLT[valid_mask],
            sat_PSD[valid_mask]
        )).T # Transpose to make it an N x 5 matrix (N rows, 5 columns)
        temp_data.append(combined_satellite_data)

    GPS_plot_data = np.concatenate(temp_data, axis=0)
    GPS_plot_data = GPS_plot_data[GPS_plot_data[:, 0].argsort()]

    # 5. MLT Filtering against REPT
    # Find nearest REPT time for each GPS point to compare MLT
    nearest_time = np.zeros(len(GPS_plot_data),dtype=int)
    MLT_mask = np.zeros(len(GPS_plot_data),dtype=bool)
    
    for i_epoch, epoch in enumerate(GPS_plot_data[:,0]):
        nearest_time[i_epoch] = np.argmin(np.abs(REPT_data[sat_select]['Epoch'].UTC-GPS_plot_data[i_epoch,0]))

        MLT_ref = REPT_data[sat_select]['MLT'][nearest_time[i_epoch]]
        MLT_gps = GPS_plot_data[:,3][i_epoch]
        
        # Calculate MLT difference considering wrap-around at 24 hours
        mlt_diff = np.minimum(np.abs(MLT_ref - MLT_gps), 24-np.abs(MLT_ref - MLT_gps))
        MLT_mask[i_epoch] = (mlt_diff <= MLT_range/2)

    # 6. Plotting Setup (RBSP Background)
    Epoch_np = np.array(REPT_data[sat_select]['Epoch'].UTC)
    time_mask_REPT = (Epoch_np >= time_start) & (Epoch_np <= time_stop)
    time_range_REPT = Epoch_np[time_mask_REPT]
    time_intervals_REPT = np.arange(time_start, time_stop+dt.timedelta(minutes=time_delta), dt.timedelta(minutes=time_delta)).astype(dt.datetime)
    
    # Color mapping based on time
    time_range_num = mdates.date2num(time_range_REPT)
    sort_indices = np.argsort(time_range_num)
    
    lstar_range = REPT_data[sat_select]['Lstar'][time_mask_REPT, i_K].flatten()
    psd_range = REPT_data[sat_select]['PSD'][k].values[:, i_mu].flatten()[time_mask_REPT]
    
    lstar_range_sorted = lstar_range[sort_indices]
    psd_range_sorted = psd_range[sort_indices]
    time_range_num_sorted = time_range_num[sort_indices]

    # Calculate binned averages for GPS lines
    lstar_min = np.nanmin(lstar_range[lstar_range > 0])
    lstar_max = np.nanmax(lstar_range)
    lstar_intervals = np.arange(np.floor(lstar_min/lstar_delta)*lstar_delta, 
                                np.ceil(lstar_max/lstar_delta)*lstar_delta + lstar_delta, 
                                lstar_delta)

    avg_psd = np.zeros((len(time_intervals_GPS), len(lstar_intervals))) * np.nan
    
    for i_time, time_int in enumerate(time_intervals_GPS):
        t_start = time_int - dt.timedelta(minutes=time_delta/2)
        t_end = time_int + dt.timedelta(minutes=time_delta/2)
        time_mask_GPS = (GPS_plot_data[:,0] >= t_start) & (GPS_plot_data[:,0] < t_end)
        
        for i_lstar, lstar_val in enumerate(lstar_intervals):
            lstar_mask = (GPS_plot_data[:,2] >= (lstar_val - lstar_delta/2)) & \
                         (GPS_plot_data[:,2] < (lstar_val + lstar_delta/2))
            
            combined_mask = time_mask_GPS & lstar_mask & MLT_mask
            
            if np.sum(combined_mask) > 1:
                psd_data = GPS_plot_data[combined_mask, 4].astype(float)
                valid_psd = psd_data[(~np.isnan(psd_data)) & (psd_data > min_val)]
                if len(valid_psd) > 0:
                    avg_psd[i_time, i_lstar] = np.nanmean(valid_psd) * gps_scale

    # 7. Create Figure                
    fig, ax = plt.subplots(figsize=(24, 10)) # 30, 9

    colormap_name = 'plasma'
    plasma = plt.cm.get_cmap(colormap_name)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('truncated_plasma',plasma(np.linspace(0, 0.9, 256)))
    
    vmin = mdates.date2num(time_start)
    vmax = mdates.date2num(time_stop)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Plot Background REPT Data
    scatter_plot = ax.scatter(lstar_range_sorted, psd_range_sorted, 
                              c=time_range_num_sorted, cmap=cmap, norm=norm, 
                              marker='o', s=20, alpha=0.3)
    
    # scatter_plot = ax.scatter(
    #     GPS_plot_data[MLT_mask,2],
    #     GPS_plot_data[MLT_mask,4],
    #     c=mdates.date2num(GPS_plot_data[MLT_mask,0]), # Color by Epoch datetime objects
    #     cmap=cmap,
    #     norm=norm,
    #     marker='*',
    #     s=80)

    # Colorbar
    cbar = fig.colorbar(scatter_plot, ax=ax, orientation='vertical', pad=0.04)
    cbar.solids.set_alpha(1)
    cbar.set_label('Time (UTC)', fontsize=textsize+2, labelpad=20)
    cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    cbar.ax.tick_params(labelsize=textsize)

    # Plot GPS Lines
    for i_time, time_int in enumerate(time_intervals_GPS):
        if np.sum(~np.isnan(avg_psd[i_time,:]) & (avg_psd[i_time,:] > min_val)) > 0:
            range_mask = ~np.isnan(avg_psd[i_time,:]) & (avg_psd[i_time,:] > min_val)
            
            # Plot line
            ax.plot(lstar_intervals[range_mask], avg_psd[i_time, range_mask],
                    marker='*', markersize=16,
                    color=cmap(norm(mdates.date2num(time_int))),
                    label=time_int.strftime("%d-%m-%Y %H:%M"))
            
            # Add Annotation
            yoff = -5 if i_time == len(time_intervals_GPS)-1 else -10
            
            ax.annotate(time_int.strftime("%H:%M"), 
                        (lstar_intervals[~np.isnan(avg_psd[i_time,:])][-1], 
                         avg_psd[i_time,:][~np.isnan(avg_psd[i_time,:])][-1]), 
                        xytext=(10, yoff),
                        textcoords='offset points',
                        fontsize=textsize+2, 
                        color=cmap(norm(mdates.date2num(time_int))),
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

    # Format Axes
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlim(4, 5.3)
    ax.set_xlabel(r"L*", fontsize=textsize+2, labelpad=10)
    ax.set_ylim(min_val, max_val)
    ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize+2)
    plt.yscale('log')
    ax.grid(True)

    # Text info
    ax.text(0.02, 0.98, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$", 
            transform=ax.transAxes, fontsize=textsize+4, verticalalignment='top')

    # Legend
    if sat_select == 'rbspa': rbsp_label = 'RBSP-A'
    elif sat_select == 'rbspb': rbsp_label = 'RBSP-B'
    
    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=10, label=rbsp_label) 
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None',
                               markersize=12, label='GPS') 
    
    ax.legend(handles=[handle_rbsp, handle_gps],
              title='Satellite',
              title_fontsize=textsize,
              loc='lower right',
              bbox_to_anchor=(1.0, 0),
              handlelength=1,
              fontsize=textsize-2)

    title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title_str, fontsize=textsize+10)
    plt.show()

#%% Plot PSD Radial Profile with REPT data (Movie)
if plot_radial_dynamic==True:
    # for feb: rbspb, Lstar 4-5.3, PSD_min=1e-10, time_interval = 2 hours, 50fps
    # for sep: rbspa, Lstar 4-5.3, PSD_min=1e-9, time_interval = 2 hours, 30fps
    
    # 1. Setup Parameters
    sat_select = 'rbspa'
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]
    anim_name = f'test'

    # Constants
    gps_scale = 1 # default = 1
    MLT_range = 12 # hours, default = 3 which corresponds to +-1.5 hours
    lstar_delta = 0.1 # default = 0.1
    time_delta = 30 # minutes, default = 30

    min_val = np.nanmin(1e-9)
    max_val = np.nanmax(1e-5)

    # 2. Load REPT Data
    REPT_data_root = '/home/wzt0020/sat_data_analysis/REPT_data/'
    save_path = os.path.join(REPT_data_root, storm_name, f'rept_data_{extMag}.npz')
    complete_load = np.load(save_path, allow_pickle=True)
    REPT_data = load_data(complete_load)
    complete_load.close()
    del complete_load

    # 3. Define Time Windows
    time_start  = start_date
    time_stop   = stop_date

    # time_start = dt.datetime(start_date.year, 2, 28, 8, 0, 0)
    # time_stop = dt.datetime(stop_date.year, 3, 1, 6, 0, 0)

    time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0) # for sep2019storm
    time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0) # for sep2019storm

    # time_start = dt.datetime(start_date.year, 8, 26, 0, 0, 0) # for aug2018storm
    # time_stop = dt.datetime(stop_date.year, 8, 27, 0, 0, 0) # for aug2018storm

    # GPS Data Collection Window
    gps_time_start  = time_start
    gps_time_stop   = time_stop

    # gps_time_start = dt.datetime(start_date.year, 2, 28, 10, 40, 0)
    # gps_time_stop = dt.datetime(stop_date.year, 3, 1, 5, 0, 0)

    gps_time_start = dt.datetime(start_date.year, 8, 31, 10, 0, 0) # for sep2019storm
    gps_time_stop = dt.datetime(stop_date.year, 8, 31, 14, 0, 0) # for sep2019storm

    # gps_time_start = dt.datetime(start_date.year, 8, 26, 6, 40, 0) # for aug2018storm
    # gps_time_stop = dt.datetime(stop_date.year, 8, 26, 13, 0, 0) # for aug2018storm

    # 4. Collect GPS Data (Same logic as static plot)
    temp_data = []
    for satellite, sat_data in storm_data.items():
        window_start = time_start - dt.timedelta(minutes=time_delta-1)
        window_stop = time_stop + dt.timedelta(minutes=time_delta-1)
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= window_start) & (sat_data['Epoch'].UTC <= window_stop)
        
        sat_epoch = sat_data['Epoch'].UTC[sat_iepoch_mask]
        sat_MLT = sat_data['MLT'][sat_iepoch_mask]
        sat_Lstar = sat_data['Lstar'][sat_iepoch_mask, i_K].flatten()
        sat_PSD = psd[satellite][k].values[sat_iepoch_mask, i_mu].flatten()
        sat_name_array = np.full(len(sat_epoch), satellite, dtype='<U10')

        valid_mask = ~np.isnan(sat_Lstar) & ~np.isnan(sat_PSD)
        combined_satellite_data = np.vstack((
            sat_epoch[valid_mask],
            sat_name_array[valid_mask],
            sat_Lstar[valid_mask],
            sat_MLT[valid_mask],
            sat_PSD[valid_mask]
        )).T
        temp_data.append(combined_satellite_data)

    GPS_plot_data = np.concatenate(temp_data, axis=0)
    GPS_plot_data = GPS_plot_data[GPS_plot_data[:, 0].argsort()]

    # MLT Filtering Logic
    rept_epochs = REPT_data[sat_select]['Epoch'].UTC
    MLT_mask = np.zeros(len(GPS_plot_data), dtype=bool)
    
    for i_epoch, epoch in enumerate(GPS_plot_data[:,0]):
        nearest_idx = np.argmin(np.abs(rept_epochs - epoch))
        MLT_ref = REPT_data[sat_select]['MLT'][nearest_idx]
        MLT_gps = GPS_plot_data[i_epoch, 3]
        mlt_diff = np.minimum(np.abs(MLT_ref - MLT_gps), 24 - np.abs(MLT_ref - MLT_gps))
        MLT_mask[i_epoch] = (mlt_diff <= MLT_range/2)

    # 5. Animation Setup
    Epoch_np = np.array(REPT_data[sat_select]['Epoch'].UTC)
    time_mask_REPT = (Epoch_np >= time_start) & (Epoch_np <= time_stop)
    time_range_REPT = Epoch_np[time_mask_REPT]
    
    import matplotlib.dates as mdates
    time_range_num = mdates.date2num(time_range_REPT)
    sort_indices = np.argsort(time_range_num)
    time_range_REPT_sorted = time_range_REPT[sort_indices]
    time_range_num_sorted = time_range_num[sort_indices]

    lstar_range_sorted = REPT_data[sat_select]['Lstar'][time_mask_REPT, i_K].flatten()[sort_indices]
    psd_range_sorted = REPT_data[sat_select]['PSD'][k].values[:, i_mu].flatten()[time_mask_REPT][sort_indices]

    # Pre-calculate Averaged GPS Data for Lines
    lstar_min = np.nanmin(lstar_range_sorted[lstar_range_sorted > 0])
    lstar_max = np.nanmax(lstar_range_sorted)
    lstar_intervals = np.arange(np.floor(lstar_min/lstar_delta)*lstar_delta, 
                                np.ceil(lstar_max/lstar_delta)*lstar_delta + lstar_delta, 
                                lstar_delta)

    avg_psd = np.zeros((len(time_intervals_GPS), len(lstar_intervals))) * np.nan
    for i_time, time_int in enumerate(time_intervals_GPS):
        t_start = time_int - dt.timedelta(minutes=time_delta/2)
        t_end = time_int + dt.timedelta(minutes=time_delta/2)
        time_mask_GPS = (GPS_plot_data[:,0] >= t_start) & (GPS_plot_data[:,0] < t_end)
        
        for i_lstar, lstar_val in enumerate(lstar_intervals):
            lstar_mask = (GPS_plot_data[:,2] >= (lstar_val - lstar_delta/2)) & \
                         (GPS_plot_data[:,2] < (lstar_val + lstar_delta/2))
            combined_mask = time_mask_GPS & lstar_mask & MLT_mask
            
            if np.sum(combined_mask) > 1:
                psd_data = GPS_plot_data[combined_mask, 4].astype(float)
                valid_psd = psd_data[(~np.isnan(psd_data)) & (psd_data > min_val)]
                if len(valid_psd) > 0:
                    avg_psd[i_time, i_lstar] = np.nanmean(valid_psd) * gps_scale

    # 6. Initialize Figure
    fig, ax = plt.subplots(figsize=(24, 8), dpi=100)
    
    colormap_name = 'plasma'
    plasma = plt.cm.get_cmap(colormap_name)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('truncated_plasma', plasma(np.linspace(0, 0.9, 256)))
    
    vmin = mdates.date2num(time_start)
    vmax = mdates.date2num(time_stop)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Empty Scatter for Animation
    scatter_plot = ax.scatter([], [], c=[], cmap=cmap, norm=norm, marker='o', s=30, alpha=0.3)
    scatter_plot.set_clim(vmin, vmax)

    # Pre-create GPS Artists (Hidden)
    gps_artists = []
    if SHOW_GPS_DATA:
        for i_time, time_int in enumerate(time_intervals_GPS):
            if np.sum(~np.isnan(avg_psd[i_time,:]) & (avg_psd[i_time,:] > min_val)) > 0:
                valid_idx = ~np.isnan(avg_psd[i_time,:]) & (avg_psd[i_time,:] > min_val)
                x_vals = lstar_intervals[valid_idx]
                y_vals = avg_psd[i_time, valid_idx]
                
                # Line
                line, = ax.plot(x_vals, y_vals, marker='*', markersize=16,
                                color=cmap(norm(mdates.date2num(time_int))),
                                label=time_int.strftime("%d-%m-%Y %H:%M"),
                                visible=False)
                
                # Annotation
                yoff = -5 if i_time == len(time_intervals_GPS)-1 else -10
                ann = ax.annotate(time_int.strftime("%H:%M"), 
                            (x_vals[-1], y_vals[-1]), 
                            xytext=(10, yoff), textcoords='offset points',
                            fontsize=textsize+2, color=cmap(norm(mdates.date2num(time_int))),
                            fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
                            visible=False)
                
                gps_artists.append({'trigger_time': time_int, 'line': line, 'annotation': ann, 'shown': False})

    # Formatting
    cbar = fig.colorbar(scatter_plot, ax=ax, orientation='vertical', pad=0.04)
    cbar.solids.set_alpha(1)
    cbar.set_label('Time (UTC)', fontsize=textsize+2, labelpad=20)
    cbar.ax.yaxis.set_major_locator(mdates.HourLocator(interval=2))
    cbar.ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    cbar.ax.tick_params(labelsize=textsize)
    cbar_line = cbar.ax.axhline(vmin, color='white', lw=3)

    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlim(4, 5.3) 
    ax.set_xlabel(r"L*", fontsize=textsize+2, labelpad=10)
    ax.set_ylim(min_val, max_val)
    ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize+2)
    plt.yscale('log')
    ax.grid(True)
    
    ax.text(0.02, 0.98, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$", 
            transform=ax.transAxes, fontsize=textsize+4, verticalalignment='top')
    
    if sat_select == 'rbspa': rbsp_label = 'RBSP-A'
    elif sat_select == 'rbspb': rbsp_label = 'RBSP-B'
    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label=rbsp_label) 
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None', markersize=12, label='GPS') 
    ax.legend(handles=[handle_rbsp, handle_gps], title='Satellite', title_fontsize=textsize, loc='lower right', 
              bbox_to_anchor=(1.0, 0), handlelength=1, fontsize=textsize-2)

    plt.tight_layout()

    # 7. Animation Function
    def update(frame):
        # Update Scatter Data (Cumulative)
        current_time_val = time_range_num_sorted[frame]
        current_x = lstar_range_sorted[:frame+1]
        current_y = psd_range_sorted[:frame+1]
        current_c = time_range_num_sorted[:frame+1]
        
        if len(current_x) > 0:
            pos_data = np.column_stack((current_x, current_y))
            scatter_plot.set_offsets(pos_data)
            scatter_plot.set_array(current_c)

        # Update Colorbar Indicator
        cbar_line.set_ydata([current_time_val, current_time_val])
        
        # Reveal GPS Artists
        current_time_dt = time_range_REPT_sorted[frame]
        for item in gps_artists:
            if not item['shown'] and current_time_dt >= item['trigger_time']:
                item['line'].set_visible(True)
                item['annotation'].set_visible(True)
                item['shown'] = True 
        
        # Return dynamic artists for blitting
        all_dynamic_artists = [scatter_plot, cbar_line]
        for item in gps_artists:
            all_dynamic_artists.append(item['line'])
            all_dynamic_artists.append(item['annotation'])
            
        return all_dynamic_artists

    # 8. Render and Save
    ani = animation.FuncAnimation(fig, update, frames=len(time_range_num_sorted), interval=20, blit=True)
    print("Saving Animation as MP4...")
    try:
        ani.save(f'{anim_name}.mp4', writer='ffmpeg', fps=30, dpi=100)
        print(f"Success! Saved as {anim_name}.mp4")
    except Exception as e:
        print(f"Error saving animation: {e}")