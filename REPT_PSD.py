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

import plotting_functions
importlib.reload(plotting_functions)
from plotting_functions import (plot_monoenergetic_flux, plot_allenergy_flux, plot_gps_psd, plot_energy_mu_alpha,
                                plot_radial_profile_static)

#%% Global Variables
textsize = 22
Re = 6378.137 #Earth's Radius

# Adiabatic Invariant Targets
Mu_set = np.array((2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G (1st Invariant)
K_set = np.array((0.1, 1, 2)) # R_E*G^(1/2) (2nd Invariant)

# Workflow Control
mode = 'load'          # 'save' (calculate & save) or 'load' (load existing npz)
storm_name = 'latefeb2019storm' 
extMag = 'TS04'        # Magnetic Model: 'T89c' or 'TS04'

# Data paths initialization
REPT_data_root = '/home/wzt0020/sat_data_analysis/REPT_data/'
input_folder = os.path.join(REPT_data_root, storm_name)
base_save_folder = os.path.join(REPT_data_root, storm_name)

# Define storm time periods based on name
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
    save_path = os.path.join(base_save_folder, f'rept_data_{extMag}.npz')
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
    PAD_filename = f"REPT_PAD_model_{extMag}.npz"
    PAD_save_path = os.path.join(base_save_folder, PAD_filename)
    if mode == 'save':
        from Zhao_2018_PAD_Model import (create_PAD)

        PAD_models = {}
        for satellite, sat_data in REPT_data.items():
            print(f"Modeling PAD for satellite {satellite}", end='\r')
            PAD_models[satellite] = create_PAD(sat_data, QD_storm_data, energyofmualpha[satellite], extMag)

        print("\nSaving RBSP PAD Model Data...")
        np.savez(PAD_save_path, **PAD_models)
        print("Data Saved \n")

    if mode == 'load':
        PAD_model_load = np.load(PAD_save_path, allow_pickle=True)
        REPT_PAD_Model = load_data(PAD_model_load)
        PAD_model_load.close()
        del PAD_model_load


    ### Execution time tracking ###
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
plot_allenergy_flux_flag = False
plot_psd_flag = False
plot_energies_flag = False
plot_radial_flag = True
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
    plot_gps_psd(
        gps_data=REPT_data,
        start_date=start_date,
        stop_date=stop_date,
        K=0.1, Mu=2000,
        textsize=textsize
    )

# Plot Energies corresponding to Mu and Alpha across L*
if plot_energies_flag:
    print("Generating Plot: Energy vs L*...")
    plot_energy_mu_alpha(
        gps_data=REPT_data,
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
        gps_data=None, REPT_data=REPT_data,
        time_start=time_start, time_stop=time_stop,
        SHOW_GPS_DATA=False,
        REPT_sat_select='rbspa', K=0.1, Mu=2000, 
        MLT_range=12, lstar_delta=0.1, time_delta=30, 
        min_val = 1e-9, max_val = 1e-5, textsize=textsize)