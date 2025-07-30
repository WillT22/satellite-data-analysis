#%% Importing relevant libraries
import os
import glob
import sys
import datetime as dt
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_script_dir)
import numpy as np
import scipy.constants as sc
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import importlib
import GPS_PSD_func
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import (QinDenton_period, load_data, data_period, AlphaOfK, find_Loss_Cone, EnergyofMuAlpha, find_psd, find_Lstar)
import REPT_PSD_func
importlib.reload(REPT_PSD_func)
from REPT_PSD_func import (process_l3_data, Average_FluxbyPA, find_mag, Interp_Flux)

#%% Global Variables
textsize = 16
Re = 6378.137 #Earth's Radius
Mu_set = np.array((4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G
K_set = np.array(0.1) # R_E*G^(1/2)

# Conversions
# electron mass in MeV is (m_e [kg] * c^2 [m^2/s^2]) [J] / (sc.eV [J/eV] * 10^6 [eV/MeV])
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6) # this is m_0*c^2
# b_satellite and b_equator are in Gauss: 1 G = 10^5 nT

base_save_folder = "/home/will/REPT_data/april2017storm/"
extMag = 'T89c'

# start_date  = dt.datetime(2017, 4, 21, 00, 00, 0)
# stop_date   = dt.datetime(2017, 4, 26, 00, 00, 0)

start_date  = dt.datetime(2017, 4, 23, 19, 30, 0)
stop_date    = dt.datetime(2017, 4, 23, 23, 00, 0)

QD_storm_data = QinDenton_period(start_date, stop_date)

#%% Main
if __name__ == '__main__':

### Load in data ###
    input_folder = "/home/will/REPT_data/april2017storm/"

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Error: Folder path not found: {input_folder}")
    
    # Get all CDF file paths in the folder
    # file_paths_l3_A = glob.glob(input_folder + "rbspa*[!r]*.cdf") 
    # file_paths_l3_B = glob.glob(input_folder + "rbspb*[!r]*.cdf") 
    
    # REPT_data_raw = {}
    # REPT_data_raw['rbspa'] = process_l3_data(file_paths_l3_A)
    # REPT_data_raw['rbspb'] = process_l3_data(file_paths_l3_B)

    raw_save_path = os.path.join(base_save_folder, 'raw_rept.npz')
    
    #Save Data for later recall:
    # print("Saving Raw REPT Data...")
    # np.savez(raw_save_path, **REPT_data_raw)
    # print("Data Saved \n")
    
    # Read in data from previous save
    raw_data_load = np.load(raw_save_path, allow_pickle=True)
    REPT_data_raw = load_data(raw_data_load)
    raw_data_load.close()
    del raw_data_load

### Restric Time Period ###
    REPT_data = {}
    for satellite, sat_data in REPT_data_raw.items():
        print(f'Restricting Time Period for satellite {satellite}')
        REPT_data[satellite] = data_period(sat_data, start_date, stop_date)
    del REPT_data_raw

### Extract Magnetometer Data ###
    for satellite, sat_data in REPT_data.items():
        print(f"Extracting Magnetic Field Data for satellite {satellite}...")
        REPT_data[satellite] = find_mag(sat_data, satellite)


### Average fluxes with the same pitch angles (assume symmetry about 90 degrees) ###
    for satellite, sat_data in REPT_data.items():
        print(f"Averaging Fluxes with the same PA for satellite {satellite}...")
        REPT_data[satellite] = Average_FluxbyPA(sat_data, satellite)

### Calculate Mu from E, alpha, and B ###
    for satellite, sat_data in REPT_data.items():
        print(f"Calculating Mu from nominal energies and pitch angles for satellite {satellite}...")
        energy_grid, alpha_grid, blocal_grid = np.meshgrid(sat_data['Energy_Channels'], np.deg2rad(sat_data['Pitch_Angles']), sat_data['b_satellite'], indexing='ij')
        REPT_data[satellite]['Mu_calc'] = (energy_grid**2 + 2 * energy_grid * E0) * np.sin(alpha_grid)**2 / (2 * E0 * blocal_grid)
    del energy_grid, alpha_grid, blocal_grid

### Find Alpha at each time point for given K ###
    alphaofK = {}
    for satellite, sat_data in REPT_data.items():
        print(f"Calculating Pitch Angle for satellite {satellite}...")
        alphaofK[satellite] = AlphaOfK(sat_data, K_set, extMag)

### Find Loss Cone and Equatorial B ###
    for satellite, sat_data in REPT_data.items():
        print(f"Calculating Equatorial B-field for satellite {satellite}...")
        REPT_data[satellite]['b_min'], REPT_data[satellite]['b_footpoint'], _ = find_Loss_Cone(sat_data, extMag=extMag)
 
### Find Energy for set Mu and Alpha ###
    energyofmualpha = {}
    for satellite, sat_data in REPT_data.items():
        print(f"Calculating Energy of Mu and Alpha for satellite {satellite}")
        energyofmualpha[satellite] = EnergyofMuAlpha(sat_data, Mu_set, alphaofK[satellite])

### Find Flux at Set Mu and K ####
    flux = {}
    flux_alpha = {}
    for satellite, sat_data in REPT_data.items():
        print(f"Interpolating flux for satellite {satellite}")
        flux[satellite], flux_alpha[satellite] = Interp_Flux(sat_data, alphaofK[satellite], energyofmualpha[satellite])

### Calculate PSD ###
    psd = {}
    for satellite, sat_data in REPT_data.items():
        print(f"Calculating PSD for satellite {satellite}")
        psd[satellite] = find_psd(flux[satellite], energyofmualpha[satellite])

### Calculate L* ####
    for satellite, sat_data in REPT_data.items():
        print(f"Calculating L* for satellite {satellite}")
        REPT_data[satellite] = find_Lstar(sat_data, alphaofK[satellite], extMag='T89c')

# %%
