#%% Importing relevant libraries
import os
import sys
import time
import datetime as dt
import importlib
import numpy as np
import scipy.constants as sc
import pandas as pd
import gc # Garbage Collection for memory management

# Add current directory to path for local imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_script_dir)

# --- Import Custom Modules ---
import all_PSD_func
importlib.reload(all_PSD_func)
from all_PSD_func import (QinDenton_period, data_period, load_data, 
                          AlphaOfK, EnergyofMuAlpha, find_psd, find_Lstar)

import GPS_PSD_func
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import (import_GPS, data_from_gps, energy_spectra)

import Zhao_2018_PAD_Model
importlib.reload(Zhao_2018_PAD_Model)
from Zhao_2018_PAD_Model import (import_zhao_coeffs, create_PAD, PAD_Scale_Factor)

import plotting_functions
importlib.reload(plotting_functions)
from plotting_functions import (plot_monoenergetic_flux, plot_allenergy_flux, plot_psd, plot_energy_mu_alpha, 
                               plot_combined_flux_all_channels, plot_combined_psd, 
                               plot_pad_comparison, plot_radial_profile_static, plot_radial_profile_dynamic)

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
#%% Main Execution
if __name__ == '__main__':

    # === File Paths ===
    raw_save_path = os.path.join(base_save_folder, 'raw_gps.npz')
    processed_save_path = os.path.join(base_save_folder, 'processed_gps.npz')

    complete_save_path = os.path.join(base_save_folder, f"storm_data_{extMag}.npz")
    alpha_save_path = os.path.join(base_save_folder, f"alphaofK_{extMag}.npz")
    energy_save_path = os.path.join(base_save_folder, f"energyofmualpha_{extMag}.npz")
    pad_save_path = os.path.join(base_save_folder, f"PAD_model_{extMag}.npz")

    # ==========================================
    # 1. Data Ingestion & Preprocessing
    # ==========================================
    if mode == 'save':
        # --- Step 1: Raw Import ---
        print("Importing Raw GPS Data...")
        loaded_data = import_GPS(input_folder)
        np.savez(raw_save_path, **loaded_data) # Save raw backup
        
        # --- Step 2: Preprocessing ---
        storm_data_raw = {}
        for satellite, sat_data in loaded_data.items():
            print(f'Restricting Time Period for {satellite}', end='\r')
            storm_data_raw[satellite] = data_period(sat_data, start_date, stop_date)
        del loaded_data

        print('Processing Data (L-shell & Efit filtering)...')
        storm_data = data_from_gps(storm_data_raw, Lshell=6, extMag=extMag)
        del storm_data_raw
        
        print("Saving Processed GPS Data...")
        np.savez(processed_save_path, **storm_data)
        print("Processed Data Saved \n")

        # ==========================================
        # Vertical Pipeline: Steps 3 - 10
        # Process one satellite fully, then clear memory
        # ==========================================
        final_results = {}
        
        # We need these lists to store aux data if you really need to plot them later
        # (Optional: If you don't plot 'energy vs L' specifically, you don't need to save these)
        energyofmualpha = {} 
        alphaofK = {}
        PAD_models = {}

        satellites = list(storm_data.keys())
        
        for satellite in satellites:
            print(f"--- Running Pipeline for {satellite} ---")
            sat_data = storm_data[satellite]

            # 3. Alpha
            print(f"Calculating Alpha...                    ")
            alpha = AlphaOfK(sat_data, K_set, extMag=extMag)
            
            # 4. Energy
            print(f"Calculating Energy...                   ")
            energy = EnergyofMuAlpha(sat_data, Mu_set, alpha)

            # 5. Omni Flux
            print(f"Calculating (quasi) Omni Flux...           ")
            flux_energy = energy_spectra(sat_data, energy)

            # 6. PAD Modeling
            print(f"Modeling PAD...                         ")
            pad_model = create_PAD(sat_data, QD_storm_data, energy, extMag)
            
            # 7. Scale Factor
            print(f"Calculating Scale Factor...             ")
            scale_factor = PAD_Scale_Factor(sat_data, QD_storm_data, energy, alpha, extMag)

            # 8. Directional Flux
            print(f"Calculating Directional Flux...         ")
            # Initialize storage for this satellite's flux
            epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
            sat_flux_result = {}
            
            for K_value in K_set:
                flux_mag = flux_energy[K_value].values
                scale_val = scale_factor[K_value].values
                # 2 * 2 * pi: accounts for 2*hemispheric detection (Hemispheric -> Directional normalization)
                directional_flux = flux_mag * (4 * np.pi) * scale_val
                sat_flux_result[K_value] = pd.DataFrame(directional_flux, index=epoch_str, columns=Mu_set)

            # 9. PSD
            print(f"Calculating PSD...                      ")
            psd_result = find_psd(sat_flux_result, energy)

            # 10. L* (Roederer L)
            print(f"Calculating L*...                       ")
            lstar_data = find_Lstar(sat_data, alpha, extMag=extMag)
            
            # --- Store Results ---
            # Attach other results to lstar_data dict
            lstar_data['Flux'] = sat_flux_result
            lstar_data['PSD'] = psd_result

            # Save to master dict
            final_results[satellite] = lstar_data
            
            # Optional: Save aux data if needed for specific plots
            energyofmualpha[satellite] = energy
            alphaofK[satellite] = alpha
            PAD_models[satellite] = pad_model

            # --- Clean Memory ---
            del sat_data, alpha, energy, flux_energy, pad_model, scale_factor, sat_flux_result, psd_result, lstar_data
            gc.collect()

# Update main variable
        storm_data = final_results
        
        print("Saving Data...")
        np.savez(complete_save_path, **storm_data)
        np.savez(alpha_save_path, **alphaofK)
        np.savez(energy_save_path, **energyofmualpha)
        np.savez(pad_save_path, **PAD_models)
        
        print("Pipeline Complete. \n")

    elif mode == 'load':
        print("Loading Final Processed Data...")
        complete_load = np.load(complete_save_path, allow_pickle=True)
        storm_data = load_data(complete_load)
        complete_load.close()
        del complete_load
        
        # --- Restore Alpha, Energy, and PAD ---
        print("Loading Aux Data (Alpha, Energy, PAD)...")
        
        # Load Raw Files
        alpha_load = np.load(alpha_save_path, allow_pickle=True)
        energy_load = np.load(energy_save_path, allow_pickle=True)
        pad_load = np.load(pad_save_path, allow_pickle=True)
        
        # Convert using helper
        alpha_raw = load_data(alpha_load)
        energyofmualpha = load_data(energy_load)
        PAD_models = load_data(pad_load) # Restore PAD Dictionary
        
        # Initialize containers for reconstructed DataFrames
        alphaofK = {}     
        for satellite, sat_data in storm_data.items():
            if satellite in alpha_raw:
                epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
                
                # Restore Alpha DataFrame
                alphaofK[satellite] = pd.DataFrame(
                    alpha_raw[satellite], index=epoch_str, columns=K_set
                )
        
        # Cleanup
        alpha_load.close()
        energy_load.close()
        pad_load.close()
        del alpha_load, energy_load, pad_load, alpha_raw
        gc.collect()

    # --- Runtime Statistics ---
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    def format_runtime(elapsed_time):
        # Calculate whole hours
        hours = int(elapsed_time // 3600)
        # Calculate remaining minutes
        minutes = int((elapsed_time % 3600) // 60)
        # Calculate remaining seconds (including decimals)
        seconds = elapsed_time % 60
        return f"Script runtime: {hours}h {minutes}m {seconds:.2f}s"
    
    print(format_runtime(elapsed_time))

#%% Plot Data
# Control Plotting Options
plot_monoenergetic_flux_flag = False
plot_allenergy_flux_flag = False
plot_flux_all_flag = False
plot_psd_flag = False
plot_combined_psd_flag = False
plot_energies_flag = False
plot_PAD_flag = False
plot_radial_flag = True
plot_radial_dynamic_flag = False

# --- LOAD REFERENCE REPT DATA (Once for all plots) ---
# REPT data is required for certain plots
rept_needed = (plot_flux_all_flag or plot_combined_psd_flag or plot_PAD_flag or plot_radial_flag or plot_radial_dynamic_flag)

REPT_data = None
REPT_energyofmualpha = None

if rept_needed:
    print("Loading Reference REPT Data...")
    rept_root = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/')
    rept_path = os.path.join(rept_root, f'rept_data_{extMag}.npz')
    
    if os.path.exists(rept_path):
        complete_load = np.load(rept_path, allow_pickle=True)
        REPT_data = load_data(complete_load)
        complete_load.close()
        
        # Load Aux Data if Plot 6 (PAD Comparison) is enabled
        if plot_PAD_flag:
            e_path = os.path.join(rept_root, f'energyofmualpha_{extMag}.npz')
            a_path = os.path.join(rept_root, f'alphaofK_{extMag}.npz')
            
            if os.path.exists(e_path) and os.path.exists(a_path):
                e_load = np.load(e_path, allow_pickle=True)
                REPT_energyofmualpha = load_data(e_load)
                e_load.close()
            else:
                print("Warning: REPT Energy/Alpha aux files missing. Plot 6 may fail.")
    else:
        print(f"Error: REPT data not found at {rept_path}")

# Plot GPS Flux for a Single Energy Channel
if  plot_monoenergetic_flux_flag:
    print("Generating Plot: Single Energy Flux...")
    plot_monoenergetic_flux(
        satellite_data=storm_data,
        start_date=start_date,
        stop_date=stop_date,
        extMag=extMag,
        target_energy=2.1, # You can change this
        min_val=1e0, max_val=1e7,
        figsize=(16, 4), textsize=textsize
    )

# Plot GPS Flux for ALL Energy Channels
if plot_allenergy_flux_flag:
    print("Generating Plot: Flux for All Energy Channels...")
    plot_allenergy_flux(
        satellite_data=storm_data, QD_storm_data=QD_storm_data,
        start_date=start_date,
        stop_date=stop_date,
        extMag=extMag,
        min_energy=1.8, max_energy=4,
        min_val=1e0, max_val=1e7,
        figsize=(24, 10), textsize=textsize
    )

# Plot Flux from REPT and CXD for All Energy Channels
if plot_flux_all_flag:
    print("Generating Plot: Combined Flux All Channels...")
    plot_combined_flux_all_channels(
        gps_data=storm_data,
        REPT_data=REPT_data,
        QD_storm_data=QD_storm_data,
        start_date=start_date,
        stop_date=stop_date,
        extMag=extMag,
        figsize=(24, 12), textsize=textsize
    )

# Plot Phase Space Density (PSD) for GPS data   
if plot_psd_flag:
    print("Generating Plot: GPS PSD...")
    plot_psd(
        satellite_data=storm_data,
        start_date=start_date,
        stop_date=stop_date,
        K=0.1, Mu=2000,
        textsize=textsize
    )

# Plot Combined Phase Space Density (PSD) from REPT and GPS CXD
if plot_combined_psd_flag:
    print("Generating Plot: Combined PSD...")
    plot_combined_psd(
        gps_data=storm_data,
        REPT_data=REPT_data,
        start_date=start_date,
        stop_date=stop_date,
        K=0.1, Mu=2000,
        textsize=textsize
    )

# Plot Energies corresponding to Mu and Alpha across L*
if plot_energies_flag:
    print("Generating Plot: Energy vs L*...")
    plot_energy_mu_alpha(
        satellite_data=storm_data,
        energyofmualpha=energyofmualpha,
        start_date=start_date,
        stop_date=stop_date,
        K=0.1, Mu=2000,
        textsize=textsize
    )

# Plot PAD Comparison between GPS and REPT
if plot_PAD_flag:
    print("Generating Plot: PAD Comparison...")
    #time_select = dt.datetime(start_date.year, 8, 31, 8, 30, 0)
    time_select = dt.datetime(start_date.year, 10, 9, 13, 30, 0)
    # Call Function
    plot_pad_comparison(
        gps_data=storm_data,            
        gps_energy=energyofmualpha, 
        gps_alpha=alphaofK,
        REPT_data=REPT_data,
        REPT_energyofmualpha=REPT_energyofmualpha,
        QD_storm_data=QD_storm_data,
        time_select=time_select, 
        gps_pad_models=PAD_models,
        REPT_sat_select='rbspa',
        extMag=extMag,
        K=0.1, Mu=2000, 
        textsize=textsize
    )

# Plot PSD Radial Profile with REPT and CXD data (Static)
if plot_radial_flag:
    '''
    time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0) # for sep2019storm
    time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0) # for sep2019storm
    
    gps_time_start = dt.datetime(start_date.year, 8, 31, 10, 0, 0) # for sep2019storm
    gps_time_stop = dt.datetime(stop_date.year, 8, 31, 14, 0, 0) # for sep2019storm
    '''
    time_start = dt.datetime(start_date.year, 10, 8, 4, 0, 0) # for oct2012storm
    time_stop = dt.datetime(stop_date.year, 10, 10, 0, 0) # for oct2012storm

    # oct2012storm: 8,4 to 8,12 ; 8,12 8,20 ; 8,20 to 9,4 ; 9,4 to 9,12 ; 9,12 to 9,20
    gps_time_start = dt.datetime(start_date.year, 10, 9, 4, 0, 0) # for oct2012storm
    gps_time_stop = dt.datetime(stop_date.year, 10, 9, 12, 0, 0) # for oct2012storm

    print("Generating Plot: Static Radial Profile...")
    plot_radial_profile_static(
        gps_data=storm_data, REPT_data=REPT_data,
        time_start=time_start, time_stop=time_stop,
        gps_time_start=gps_time_start, gps_time_stop=gps_time_stop, SHOW_GPS_DATA=True,
        REPT_sat_select='rbspa', K=0.1, Mu=2000, 
        MLT_range=12, time_delta=30, skip_interval=4,
        lstar_min = 3.4, lstar_max = 5.4, lstar_delta=0.1,
        min_val = 1e-11, max_val = 1e-5, textsize=16)

# Plot PSD Radial Profile with REPT and CXD data (Dynamic)
if plot_radial_dynamic_flag:
    time_start = dt.datetime(start_date.year, 10, 8, 4, 0, 0) # for oct2012storm
    time_stop = dt.datetime(stop_date.year, 10, 10, 0, 0) # for oct2012storm

    gps_time_start = dt.datetime(start_date.year, 10, 8, 4, 0, 0) # for oct2012storm
    gps_time_stop = dt.datetime(stop_date.year, 10, 10, 0, 0, 0) # for oct2012storm

    print("Generating Plot: Dynamic Radial Profile...")
    plot_radial_profile_dynamic(
        gps_data=storm_data, REPT_data=REPT_data,
        time_start=time_start, time_stop=time_stop,
        gps_time_start=gps_time_start, gps_time_stop=gps_time_stop, SHOW_GPS_DATA=True,
        REPT_sat_select='rbspb', K=0.1, Mu=2000,
        MLT_range=12, time_delta=30, skip_interval=1,
        lstar_min = 3.4, lstar_max = 5.4, lstar_delta=0.1,
        min_val = 1e-11, max_val = 1e-5, textsize=16, 
        base_save_folder = base_save_folder, anim_name = f'{storm_name}_radial_profile_sliding', sliding_window=True)
# %%
