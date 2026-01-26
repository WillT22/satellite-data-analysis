#%% Importing relevant libraries
import glob
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

# --- Import Custom Libraries ---
import all_PSD_func
importlib.reload(all_PSD_func)
from all_PSD_func import (QinDenton_period, data_period, load_data, find_Loss_Cone, find_local90PA, 
                          AlphaOfK, EnergyofMuAlpha, find_psd, find_McIlwain_L, find_Lstar)

import REPT_PSD_func
importlib.reload(REPT_PSD_func)
from REPT_PSD_func import (process_l3_data, time_average, find_mag, Average_FluxbyPA, Interp_Flux)

import Zhao_2018_PAD_Model
importlib.reload(Zhao_2018_PAD_Model)
from Zhao_2018_PAD_Model import (create_PAD)

import plotting_functions
importlib.reload(plotting_functions)
from plotting_functions import (plot_monoenergetic_flux, plot_allenergy_flux, plot_psd, 
                               plot_radial_profile_static, plot_energy_mu_alpha)

#%% Global Variables
textsize = 22
Re = 6378.137 #Earth's Radius

# Adiabatic Invariant Targets
Mu_set = np.array((2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G (1st Invariant)
K_set = np.array((0.1, 1, 2)) # R_E*G^(1/2) (2nd Invariant)

# Workflow Control
mode = 'save'          # 'save' (calculate & save) or 'load' (load existing npz)
storm_name = 'oct2012storm' 
extMag = 'TS04'        # Magnetic Model: 'T89c' or 'TS04'

# Data Paths
REPT_data_root = '/home/wzt0020/sat_data_analysis/REPT_data/'
input_folder = os.path.join(REPT_data_root, storm_name)
base_save_folder = os.path.join(REPT_data_root, storm_name)

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

#%% Main Execution
if __name__ == '__main__':

    # === File Paths ===
    raw_save_path = os.path.join(base_save_folder, 'raw_rept.npz')
    
    complete_save_path = os.path.join(base_save_folder, f"rept_data_{extMag}.npz")
    alpha_save_path = os.path.join(base_save_folder, f"alphaofK_{extMag}.npz")
    energy_save_path = os.path.join(base_save_folder, f"energyofmualpha_{extMag}.npz")
    pad_save_path = os.path.join(base_save_folder, f"REPT_PAD_model_{extMag}.npz")
    
    # ==========================================
    # 1. Data Ingestion & Preprocessing
    # ==========================================
    if mode == 'save':
        # --- Step 1: Raw Import ---
        if not os.path.exists(raw_save_path):
            print("Processing Raw CDF Files...")
            if not os.path.exists(input_folder):
                raise FileNotFoundError(f"Error: Folder path not found: {input_folder}")
            
            file_paths_l3_A = glob.glob(input_folder + "/rbspa*[!r]*.cdf") 
            file_paths_l3_B = glob.glob(input_folder + "/rbspb*[!r]*.cdf")
            
            REPT_data_raw = {}
            if len(file_paths_l3_A) != 0:
                REPT_data_raw['rbspa'] = process_l3_data(file_paths_l3_A)
            if len(file_paths_l3_B) != 0:
                REPT_data_raw['rbspb'] = process_l3_data(file_paths_l3_B)
            
            print("Saving Raw REPT Data...")
            np.savez(raw_save_path, **REPT_data_raw)
            print("Raw Data Saved \n")
        else:
            print("Loading existing Raw REPT Data...")
            raw_data_load = np.load(raw_save_path, allow_pickle=True)
            REPT_data_raw = load_data(raw_data_load)
            raw_data_load.close()

        # ==========================================
        # Vertical Pipeline: Steps 2 - 12
        # ==========================================
        final_results = {}
        
        # Aux containers for saving
        alphaofK = {}
        energyofmualpha = {}
        PAD_models = {}

        satellites = list(REPT_data_raw.keys())

        for satellite in satellites:
            print(f"\n--- Running Pipeline for {satellite} ---")
            
            # --- 2. Time Filtering & Averaging ---
            print(f"Filtering and Averaging...              ", end='\r')
            sat_data = data_period(REPT_data_raw[satellite], start_date, stop_date)
            sat_data = time_average(sat_data, satellite)

            # --- 3. Magnetic Field & Flux Avg ---
            print(f"Extracting B-Field & Averaging Flux...  ", end='\r')
            sat_data = find_mag(sat_data, satellite)
            sat_data = Average_FluxbyPA(sat_data, satellite)

            # --- 4. Calculate Nominal Mu (Pre-calc) ---
            print(f"Calculating Nominal Mu...               ", end='\r')
            energy_grid, alpha_grid, blocal_grid = np.meshgrid(
                sat_data['Energy_Channels'], np.deg2rad(sat_data['Pitch_Angles']), sat_data['b_satellite'], 
                indexing='ij')
            # Formula: Mu = E_perp / B = (E^2 + 2*E*E0) * sin^2(alpha) / (2*E0*B)
            sat_data['Mu_calc'] = (energy_grid**2 + 2 * energy_grid * E0) * np.sin(alpha_grid)**2 / (2 * E0 * blocal_grid)
            del energy_grid, alpha_grid, blocal_grid

            # --- 5. Alpha ---
            print(f"Calculating Alpha...                    ", end='\r')
            alpha = AlphaOfK(sat_data, K_set, extMag)

            # --- 6. Loss Cone & Eq B ---
            print(f"Calculating Loss Cone...                ", end='\r')
            sat_data['b_min'], sat_data['P_min'], sat_data['b_footpoint'], sat_data['loss_cone'] = find_Loss_Cone(sat_data, extMag=extMag)
            sat_data['local90PA'] = find_local90PA(sat_data)
    
            # --- 7. Energy --- 
            print(f"Calculating Energy...                   ", end='\r')
            # Safe numeric conversion for Energy (handling Inf/Nan)
            energy = EnergyofMuAlpha(sat_data, Mu_set, alpha)

            # --- 8. Interpolate Flux ---
            print(f"Interpolating Flux...                   ", end='\r')
            flux_result, _ = Interp_Flux(sat_data, alpha, energy)

            # --- 9. PSD ---
            print(f"Calculating PSD...                      ", end='\r')
            psd_result = find_psd(flux_result, energy)

            # --- 10. L-Shell (McIlwain & L*) ---
            print(f"Calculating McIlwain L...               ", end='\r')
            # First calculate McIlwain L
            sat_data = find_McIlwain_L(sat_data, extMag=extMag)
            print(f"Calculating L*...                       ", end='\r')
            # Then calculate L*
            lstar_data = find_Lstar(sat_data, alpha, extMag=extMag)

            # --- 12. PAD Modeling (extra step) ---
            print(f"Modeling PAD...                         ", end='\r')
            pad_model = create_PAD(sat_data, QD_storm_data, energy, extMag)

            # --- Store Results ---
            lstar_data['Flux'] = flux_result
            lstar_data['PSD'] = psd_result
            
            # Persist aux data
            energyofmualpha[satellite] = energy
            alphaofK[satellite] = alpha
            PAD_models[satellite] = pad_model
            final_results[satellite] = lstar_data

            # --- Clean Memory ---
            del sat_data, alpha, energy, flux_result, psd_result, lstar_data, pad_model
            gc.collect()

        # Update main variable
        REPT_data = final_results
        
        # Clean up the raw dict entirely
        del REPT_data_raw
        gc.collect()

        print("\nSaving Processed Data...               ")
        np.savez(complete_save_path, **REPT_data)
        np.savez(alpha_save_path, **alphaofK)
        np.savez(energy_save_path, **energyofmualpha)
        np.savez(pad_save_path, **PAD_models)
        
        print("Pipeline Complete.                       \n")

    elif mode == 'load':
        print("Loading Final Processed Data...")
        complete_load = np.load(complete_save_path, allow_pickle=True)
        REPT_data = load_data(complete_load)
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
        for satellite, sat_data in REPT_data.items():
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
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        return f"Script runtime: {hours}h {minutes}m {seconds:.2f}s"
    
    print(format_runtime(elapsed_time))

#%% PLOTTING SECTION
# Control Plotting Options
plot_monoenergetic_flux_flag = False
plot_allenergy_flux_flag = True
plot_psd_flag = True
plot_energies_flag = True
plot_radial_flag = False
plot_radial_Lstar_flag = False

# Plot REPT Flux for a Single Energy Channel
if plot_monoenergetic_flux_flag:
    print("Generating Plot: Single Energy Flux...")
    plot_monoenergetic_flux(
        satellite_data=REPT_data,
        start_date=start_date,
        stop_date=stop_date,
        extMag=extMag,
        target_energy=2.1,
        min_val=1e2, max_val=1e6,
        figsize=(16, 4), textsize=textsize
    )

# Plot REPT Flux for ALL Energy Channels
if plot_allenergy_flux_flag:
    print("Generating Plot: Flux for All Energy Channels...")
    plot_allenergy_flux(
        satellite_data=REPT_data, QD_storm_data=QD_storm_data,
        start_date=start_date,
        stop_date=stop_date,
        extMag=extMag,
        max_energy=4,
        min_val=1e2, max_val=1e7,
        figsize=(24, 10), textsize=textsize
    )

# Plot Phase Space Density (PSD) for REPT data   
if plot_psd_flag:
    print("Generating Plot: REPT PSD...")
    plot_psd(
        satellite_data=REPT_data,
        start_date=start_date,
        stop_date=stop_date,
        K=0.1, Mu=2000,
        textsize=textsize
    )

# Plot Energies corresponding to Mu and Alpha across L*
if plot_energies_flag:
    print("Generating Plot: Energy vs L*...")
    plot_energy_mu_alpha(
        satellite_data=REPT_data,
        energyofmualpha=energyofmualpha,
        start_date=start_date,
        stop_date=stop_date,
        K=0.1, Mu=2000,
        textsize=textsize
    )

# Plot PSD Radial Profile of REPT data (Static)
if plot_radial_flag:
    # time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0) # for sep2019storm
    # time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0) # for sep2019storm

    time_start = dt.datetime(start_date.year, 2, 28, 0, 0, 0) # for latefeb2019storm
    time_stop = dt.datetime(stop_date.year, 3, 1, 0, 0, 0) # for latefeb2019storm

    print("Generating Plot: Static Radial Profile...")
    plot_radial_profile_static(
        satellite_data=None, REPT_data=REPT_data,
        time_start=time_start, time_stop=time_stop,
        SHOW_GPS_DATA=False,
        REPT_sat_select='rbspa', K=0.1, Mu=2000, 
        MLT_range=12, time_delta=30, skip_interval=1,
        lstar_min = 3.5, lstar_max = 6.0, lstar_delta=0.1,
        min_val = 1e-9, max_val = 1e-5, textsize=textsize)