#%% Importing relevant libraries
import os
import sys
import time
import datetime as dt
import importlib
import numpy as np
import scipy.constants as sc
import pandas as pd

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

import plotting_functions
from plotting_functions import (plot_gps_flux, plot_gps_psd, plot_energy_mu_alpha, 
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
    for satellite, sat_data in storm_data.items():
        storm_data[satellite]['PSD'] = {}
        storm_data[satellite]['PSD'] = find_psd(flux[satellite], energyofmualpha[satellite])

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

#%% Plot Data
# Control Plotting Options
plot_flux_flag = False
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
if  plot_flux_flag:
    print("Generating Plot: Single Energy Flux...")
    plot_gps_flux(
        gps_data=storm_data,
        start_date=start_date,
        stop_date=stop_date,
        extMag=extMag,
        target_energy=2.1, # You can change this
        textsize=textsize
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
        textsize=textsize
    )

# Plot Phase Space Density (PSD) for GPS data   
if plot_psd_flag:
    print("Generating Plot: GPS PSD...")
    plot_gps_psd(
        gps_data=storm_data,
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
        gps_data=storm_data,
        energyofmualpha=energyofmualpha,
        start_date=start_date,
        stop_date=stop_date,
        K=0.1, Mu=2000,
        textsize=textsize
    )

# Plot PAD Comparison between GPS and REPT
if plot_PAD_flag:
    print("Generating Plot: PAD Comparison...")
    time_select = dt.datetime(start_date.year, 8, 31, 8, 30, 0)
    # Call Function
    plot_pad_comparison(
        gps_data=storm_data,
        gps_flux=flux,              
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
    time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0) # for sep2019storm
    time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0) # for sep2019storm
    
    gps_time_start = dt.datetime(start_date.year, 8, 31, 10, 0, 0) # for sep2019storm
    gps_time_stop = dt.datetime(stop_date.year, 8, 31, 14, 0, 0) # for sep2019storm

    print("Generating Plot: Static Radial Profile...")
    plot_radial_profile_static(
        gps_data=storm_data, REPT_data=REPT_data,
        time_start=time_start, time_stop=time_stop,
        gps_time_start=gps_time_start, gps_time_stop=gps_time_stop, SHOW_GPS_DATA=True,
        REPT_sat_select='rbspa', K=0.1, Mu=2000, 
        MLT_range=12, lstar_delta=0.1, time_delta=30, 
        min_val = 1e-9, max_val = 1e-5, textsize=textsize)

# Plot PSD Radial Profile with REPT and CXD data (Dynamic)
if plot_radial_dynamic_flag:
    time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0) # for sep2019storm
    time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0) # for sep2019storm
    
    gps_time_start = dt.datetime(start_date.year, 8, 31, 10, 0, 0) # for sep2019storm
    gps_time_stop = dt.datetime(stop_date.year, 8, 31, 14, 0, 0) # for sep2019storm

    print("Generating Plot: Dynamic Radial Profile...")
    plot_radial_profile_dynamic(
        gps_data=storm_data, REPT_data=REPT_data,
        time_start=time_start, time_stop=time_stop,
        gps_time_start=gps_time_start, gps_time_stop=gps_time_stop, SHOW_GPS_DATA=True,
        REPT_sat_select='rbspa', K=0.1, Mu=2000,
        anim_name = f'test', 
        MLT_range=12, lstar_delta=0.1, time_delta=30, 
        min_val = 1e-9, max_val = 1e-5, textsize=textsize)