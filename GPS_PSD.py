
#%% Importing relevant libraries
import os
import sys
import datetime as dt
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_script_dir)
import numpy as np
import scipy.constants as sc
import math

import importlib
import GPS_PSD_func
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import (import_GPS, data_period, QinDenton_period, data_from_gps, load_data,
                            find_local90PA, AlphaOfK, MuofEnergyAlpha, EnergyofMuAlpha)
import Zhao2018_PAD_Model
importlib.reload(Zhao2018_PAD_Model)
from Zhao2018_PAD_Model import (import_Zhao_coeffs, find_Zhao_PAD_coeffs)

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
    """ 
    # Save Data for later recall:
    print("Saving Raw GPS Data...")
    np.savez(raw_save_path, **loaded_data)
    print("Data Saved \n")
    """
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
    """ 
    # Save Data for later recall:
    print("Saving Processed GPS Data...")
    np.savez(processed_save_path, **storm_data)
    print("Data Saved \n")
    """
    # Read in data from previous save
    storm_data_load = np.load(processed_save_path, allow_pickle=True)
    storm_data = load_data(storm_data_load)

### Find Pitch Angles ###
    # Find local pitch angle
    local90PA = find_local90PA(storm_data)

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
                       np.linspace(math.floor(Mu_bounds['Rounded'][0]/2000)*2000,
                                   math.ceil(Mu_bounds['Rounded'][1]/2000)*2000, 
                                   math.ceil(Mu_bounds['Rounded'][1]/2000)-math.floor(Mu_bounds['Rounded'][0]/2000)+1)[1:]))))

    energyofmualpha = EnergyofMuAlpha(storm_data, Mu_set, alphaofK)

### Find Flux at Set Pitch Angle ####
    Zhao_epoch_coeffs = find_Zhao_PAD_coeffs(storm_data, energyofmualpha)

    Zhao_epoch_coeffs_filename = f"Zhao_epoch_coeffs.npz"
    Zhao_epoch_coeffs_save_path = os.path.join(base_save_folder, Zhao_epoch_coeffs_filename)
    '''
    # Save Data for later recall:
    print("Saving Zhao coefficients for each Epoch...")
    np.savez(zhao_epoch_save_path, **Zhao_epoch_coeffs)
    print("Data Saved \n")
    '''
    # Load data from previous save
    Zhao_epoch_coeffs_load = np.load(Zhao_epoch_coeffs_save_path, allow_pickle=True)
    Zhao_epoch_coeffs = load_data(Zhao_epoch_coeffs_load)

    
    # Test set
    satellite = 'ns58'
    k = 0.1
    i_epoch = 368
    mu = 4000
    i_mu = np.where(Mu_set == mu)[0][0]
    test_coeffs = Zhao_epoch_coeffs[satellite][k][mu].values[:,i_epoch]
    test_epoch = np.array(storm_data[satellite]['Epoch'].UTC)[i_epoch]
    test_channels = np.array(storm_data[satellite]['Energy_Channels'])
    test_energy = energyofmualpha[satellite]['EnergyofMuAlpha'][k].values[i_mu,i_epoch]
    energy_bins = np.array(list(Zhao2018_PAD_Model.Zhao_coeffs.keys()), dtype=float)
    i_energy = np.argmin(np.abs(test_energy-energy_bins))
    ebin_value = energy_bins[i_energy]
    test_dst = Zhao2018_PAD_Model.QD_data['Dst'][i_epoch]
    if test_dst > -20:
        i_dst = 'Dst > -20 nT'
    elif test_dst < -20 and test_dst > -50:
        i_dst = '-50 nT < Dst < -20 nT'
    elif test_dst < -50:
        i_dst = 'Dst < -50 nT'
    test_MLT = storm_data[satellite]['MLT'][i_epoch]
    i_MLT = int(((test_MLT + 1) % 24) // 2)
    test_Lshell = storm_data[satellite]['L_LGM_T89IGRF'][i_epoch]
    if ebin_value < 1:
        i_L = int((test_Lshell - 0.9) // 0.2)
    elif ebin_value >= 1:
        i_L = int((test_Lshell - 2.9) // 0.2)
    test_coeff_check = np.zeros(3)
    test_coeff_check[0] = Zhao2018_PAD_Model.Zhao_coeffs[ebin_value][i_dst]['c2']['data_matrix'].values[i_MLT,i_L]
    test_coeff_check[1] = Zhao2018_PAD_Model.Zhao_coeffs[ebin_value][i_dst]['c4']['data_matrix'].values[i_MLT,i_L]
    test_coeff_check[2] = Zhao2018_PAD_Model.Zhao_coeffs[ebin_value][i_dst]['c6']['data_matrix'].values[i_MLT,i_L]
    