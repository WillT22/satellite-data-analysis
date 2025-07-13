#%% Importing relevant libraries
import os
import sys
import datetime as dt
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_script_dir)
import numpy as np
import scipy.constants as sc
import math
import matplotlib
import matplotlib.pyplot as plt

import importlib
import GPS_PSD_func
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import (import_GPS, data_period, QinDenton_period, data_from_gps, load_data,
                            AlphaOfK, MuofEnergyAlpha, EnergyofMuAlpha, PAD_Integral)
import Zhao2018_PAD_Model
importlib.reload(Zhao2018_PAD_Model)
from Zhao2018_PAD_Model import (import_Zhao_coeffs, find_Zhao_PAD_coeffs, create_PAD)

#%% Global Variables
textsize = 16
Re = 6378.137 #Earth's Radius
K_set = [0.1,1] # R_E*G^(1/2)

# Conversions
# electron mass in MeV is (m_e [kg] * c^2 [m^2/s^2]) [J] / (sc.eV [J/eV] * 10^6 [eV/MeV])
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6) # this is m_0*c^2
# b_satellite and b_equator are in Gauss: 1 G = 10^5 nT

start_date  = Zhao2018_PAD_Model.start_date
stop_date   = Zhao2018_PAD_Model.stop_date # exclusive, end of the last day you want to see

QD_storm_data = Zhao2018_PAD_Model.QD_data

base_save_folder = "/home/will/GPS_data/april2017storm/"
extMag = 'T89'

#%% Main
if __name__ == '__main__':

    # Load in data
    input_folder = "/home/will/GPS_data/april2017storm/"
    #loaded_data = import_GPS(input_folder)

    raw_save_path = os.path.join(base_save_folder, 'raw_gps.npz')
    '''
    # Save Data for later recall:
    print("Saving Raw GPS Data...")
    np.savez(raw_save_path, **loaded_data)
    print("Data Saved \n")
    '''
    # Read in data from previous save
    raw_data_load = np.load(raw_save_path, allow_pickle=True)
    loaded_data = load_data(raw_data_load)

### Preprocessing ###    
    # Restrict to time period
    storm_data_raw = data_period(loaded_data, start_date, stop_date)
    
    # Limit to relevant Lshells, convert satellite position from spherical GEO to GSM and extract relevant data
    # (Takes a few minutes)
    #storm_data = data_from_gps(storm_data_raw, Lshell=6, extMag= 'T89')
    
    processed_save_path = os.path.join(base_save_folder, 'processed_gps.npz')
    '''
    # Save Data for later recall:
    print("Saving Processed GPS Data...")
    np.savez(processed_save_path, **storm_data)
    print("Data Saved \n")
    '''
    # Read in data from previous save
    storm_data_load = np.load(processed_save_path, allow_pickle=True)
    storm_data = load_data(storm_data_load)

### Find Pitch Angles ###
    # Find pitch angle corresponding to set K
    #alphaofK = AlphaOfK(storm_data, K_set, extMag)

    alphaofK_filename = f"alphaofK_{extMag}.npz"
    alphaofK_save_path = os.path.join(base_save_folder, alphaofK_filename)
    '''
    # Save Data for later recall:
    print("Saving AlphaofK Data...")
    np.savez(alphaofK_save_path, **alphaofK)
    print("Data Saved \n")
    '''
    # Load data from previous save
    alphaofK_load = np.load(alphaofK_save_path, allow_pickle=True)
    alphaofK = load_data(alphaofK_load)

### Find Energies from Mu and AlphaofK ###
    # Find Mu spread of energy channels
    muofenergyalpha, Mu_bounds = MuofEnergyAlpha(storm_data, alphaofK)
    Mu_set = np.unique(np.sort(np.concatenate(([Mu_bounds['Rounded'][0]], 
                       np.logspace(np.log10(Mu_bounds['Rounded'][0]),
                                   np.log10(math.ceil(Mu_bounds['Rounded'][1]/2000)*2000), 
                                   math.ceil(Mu_bounds['Rounded'][1]/2000)-math.floor(Mu_bounds['Rounded'][0]/2000)+1)[1:]))))

    energyofmualpha = EnergyofMuAlpha(storm_data, Mu_set, alphaofK)

### Find Flux at Set Pitch Angle ####
    #--- Extract Zhao Coefficients at each Epoch ---
    #Zhao_epoch_coeffs = find_Zhao_PAD_coeffs(storm_data, energyofmualpha)

    Zhao_epoch_coeffs_filename = f"Zhao_epoch_coeffs.npz"
    Zhao_epoch_coeffs_save_path = os.path.join(base_save_folder, Zhao_epoch_coeffs_filename)
    '''
    # Save Data for later recall:
    print("Saving Zhao coefficients for each Epoch...")
    np.savez(Zhao_epoch_coeffs_save_path, **Zhao_epoch_coeffs)
    print("Data Saved \n")
    '''
    # Load data from previous save
    Zhao_epoch_coeffs_load = np.load(Zhao_epoch_coeffs_save_path, allow_pickle=True)
    Zhao_epoch_coeffs = load_data(Zhao_epoch_coeffs_load)

    #--- Create Pitch Angle Distribution (PAD) from Coefficients ---
    #PAD_models = create_PAD(storm_data, Zhao_epoch_coeffs, alphaofK)

    PAD_models_filename = f"PAD_models.npz"
    PAD_models_save_path = os.path.join(base_save_folder, PAD_models_filename)
    '''
    # Save Data for later recall:
    print("Saving PAD models ...")
    np.savez(PAD_models_save_path, **PAD_models)
    print("Data Saved \n")
    '''
    # Load data from previous save
    PAD_models_load = np.load(PAD_models_save_path, allow_pickle=True)
    PAD_models = load_data(PAD_models_load) 

#%% Test PAD Integral
satellite = 'ns63'
k = 0.1
i_mu = 8
mu = Mu_set[i_mu]
i_epoch = 180

Zhao_test_coeffs = Zhao_epoch_coeffs[satellite][k][mu].values[i_epoch,:]
b_sat = storm_data[satellite]['b_satellite'][i_epoch]
b_eq = storm_data[satellite]['b_equator'][i_epoch]
b_fpt = storm_data[satellite]['b_footpoint'][i_epoch]
b_min = storm_data[satellite]['b_min'][i_epoch]
integral_test = PAD_Integral(b_sat,b_eq,b_fpt,b_min,Zhao_test_coeffs)
print(integral_test)

def Zhao_Integral_batch(a, C):
    P = np.array([
        2 / a -       2 / a**2,
        2 / a -  20 / 3 / a**2 +  14 / 3 / a**3,
        2 / a -      14 / a**2 + 126 / 5 / a**3 -    66 / 5 / a**4,
        2 / a -      24 / a**2 + 396 / 5 / a**3 - 3432 / 35 / a**4 +    286 / 7 / a**5,
        2 / a - 110 / 3 / a**2 + 572 / 3 / a**3 -  2860 / 7 / a**4 + 24310 / 63 / a**5 - 8398 / 63 / a**6
    ]) * a / 2
    return np.sum(C * P) + 2/a

test_batch = Zhao_Integral_batch(b_sat/b_eq,Zhao_test_coeffs)
print(test_batch)

PAD_epoch = PAD_models[satellite][k][mu]['Model'].values[i_epoch,:]
alpha_epoch = PAD_models[satellite][k][mu]['pitch_angles'].values[i_epoch,:]
loss_cone_epoch = storm_data[satellite]['loss_cone'][i_epoch]
local90_epoch = storm_data[satellite]['local90PA'][i_epoch]
integral_mask = (alpha_epoch >= loss_cone_epoch) & (alpha_epoch <= local90_epoch)
alpha_epoch_rad = np.deg2rad(alpha_epoch)
integrand = PAD_epoch * np.sin(alpha_epoch_rad)
numerical_integral_test = 2 * 2*np.pi*np.trapz(PAD_epoch[integral_mask]*np.sin(alpha_epoch_rad[integral_mask]),alpha_epoch_rad[integral_mask])
print(numerical_integral_test)

integral2_mask = (alpha_epoch <= local90_epoch)
numerical_integral2_test = 2 * 2*np.pi * np.trapz(PAD_epoch[integral2_mask]*np.sin(alpha_epoch_rad[integral2_mask]),np.deg2rad(alpha_epoch[integral2_mask]))
print(numerical_integral2_test)

def P0_int(x):
    return x
def P2_int(x):
    return 1/2*(-x + x**3)
def P4_int(x):
    return 1/8*(3*x - 10*x**3 + 7*x**5)
def P6_int(x):
    return 1/16*(-5*x + 35*x**3 - 63*x**5 + 33*x**7)
def P8_int(x):
    return 1/128*(35*x - 420*x**3 + 1386*x**5 - 1716*x**7 + 715*x**9)
def P10_int(x):
    return 1/256*(-63*x + 1155*x**3 - 6006*x**5 + 12870*x**7 - 12155*x**9 + 4199*x**11)

def Zhao_Integral_batch(a, b, C):
    x = np.sqrt(1 - 1/a)
    y = np.sqrt(1 - 1/b)
    P = np.array([
        P2_int(y) - P2_int(x),
        P4_int(y) - P4_int(x),
        P6_int(y) - P6_int(x),
        P8_int(y) - P8_int(x),
        P10_int(y) - P10_int(x),
    ])
    return 2 * 2*np.pi * (np.sum(C * P) + (P0_int(y) - P0_int(x)))

test_batch = Zhao_Integral_batch(b_sat/b_eq,b_fpt/b_min,Zhao_test_coeffs)
print(test_batch)

def Zhao_Integral_batch2(a, C):
    x = np.sqrt(1 - 1/a)
    P = np.array([
        P2_int(1) - P2_int(x),
        P4_int(1) - P4_int(x),
        P6_int(1) - P6_int(x),
        P8_int(1) - P8_int(x),
        P10_int(1) - P10_int(x),
    ])
    return 2 * 2*np.pi * (np.sum(C * P) + (P0_int(1) - P0_int(x)))

test_batch2 = Zhao_Integral_batch2(b_sat/b_eq,Zhao_test_coeffs)
print(test_batch2)

# %%
