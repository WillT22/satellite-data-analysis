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
                            AlphaOfK, MuofEnergyAlpha, EnergyofMuAlpha)
import Zhao2018_PAD_Model
importlib.reload(Zhao2018_PAD_Model)
from Zhao2018_PAD_Model import (import_Zhao_coeffs, find_Zhao_PAD_coeffs, create_PAD, PAD_Scale_Factor, define_Legendre, define_Legendre_Int, P0_int)

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
    '''
    PAD_models = create_PAD(Zhao_epoch_coeffs, alphaofK)

    PAD_models_filename = f"PAD_models.npz"
    PAD_models_save_path = os.path.join(base_save_folder, PAD_models_filename)
    
    # Save Data for later recall:
    print("Saving PAD models ...")
    np.savez(PAD_models_save_path, **PAD_models)
    print("Data Saved \n")
    
    # Load data from previous save
    PAD_models_load = np.load(PAD_models_save_path, allow_pickle=True)
    PAD_models = load_data(PAD_models_load) 
    '''

    #--- Find Scale Factor from alphaofK and PAD Model ---#
    scale_factor = PAD_Scale_Factor(storm_data,Zhao_epoch_coeffs,alphaofK)

#%% Test PAD Integral
satellite = 'ns63'
k = 1
i_mu = 4
mu = Mu_set[i_mu]
i_epoch = 180

Zhao_test_coeffs = Zhao_epoch_coeffs[satellite][k][mu].values[i_epoch,:]
b_sat = storm_data[satellite]['b_satellite'][i_epoch]
b_eq = storm_data[satellite]['b_equator'][i_epoch]
b_fpt = storm_data[satellite]['b_footpoint'][i_epoch]
b_min = storm_data[satellite]['b_min'][i_epoch]

a = b_sat/b_eq
b = b_fpt/b_min

x = np.sqrt(1 - 1/a)
y = np.sqrt(1 - 1/b)

coeffs = Zhao_epoch_coeffs[satellite][k][mu].values
alphaofK_data = alphaofK[satellite]['AlphaofK'].values
P = define_Legendre(alphaofK_data[:,1])
PAD_models = np.sum(coeffs * P, axis=1) + 1
P_int = define_Legendre_Int(x,y)
PAD_integral = 2 * 2*np.pi * (np.sum(coeffs * P_int, axis=1) + (P0_int(y) - P0_int(x)))

fig, ax = plt.subplots(figsize=(8, 5))
for i_Mu in range(len(Mu_set)):
    mu = Mu_set[i_Mu]
    ax.scatter(storm_data[satellite]['local90PA'], scale_factor[satellite][k].values[:,i_Mu],
               label=f'Mu = {mu:.2f}')
#ax.set_xlim(min(alpha_list), max(alpha_list))
ax.legend(title='Mu Values', loc='best', fontsize=10)

# %%
