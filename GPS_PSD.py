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
import pandas as pd

import importlib
import GPS_PSD_func
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import (import_GPS, data_period, QinDenton_period, data_from_gps, load_data,
                            AlphaOfK, MuofEnergyAlpha, EnergyofMuAlpha, energy_spectra)
import Zhao2018_PAD_Model
importlib.reload(Zhao2018_PAD_Model)
from Zhao2018_PAD_Model import (import_Zhao_coeffs, find_Zhao_PAD_coeffs, create_PAD, PAD_Scale_Factor, define_Legendre, define_Legendre_Int, P0_int)

#%% Global Variables
textsize = 16
Re = 6378.137 #Earth's Radius
K_set = np.array((0.1,1)) # R_E*G^(1/2)

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

### Find Flux at Set Energy ###
    flux_energyofmualpha = energy_spectra(storm_data, energyofmualpha)

### Find Flux at Set Pitch Angle and Energy
    flux = {}
    for satellite, sat_data in storm_data.items():
        flux[satellite] = {}
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        for i_K, K_value in enumerate(K_set):
            flux[satellite][K_value] = flux_energyofmualpha[satellite][K_value].values * scale_factor[satellite][K_value].values
            flux[satellite][K_value] = pd.DataFrame(flux[satellite][K_value], index = epoch_str, columns=Mu_set)

#%% Test PAD Integral
satellite = 'ns63'
k = 0.1
i_K = np.where(K_set == k)[0]
i_mu = 4
mu = Mu_set[i_mu]
i_epoch = 180

Zhao_test_coeffs = Zhao_epoch_coeffs[satellite][k][mu].values
b_sat = storm_data[satellite]['b_satellite']
b_eq = storm_data[satellite]['b_equator']
b_fpt = storm_data[satellite]['b_footpoint']
b_min = storm_data[satellite]['b_min']

a = b_sat/b_eq
b = b_fpt/b_min

x = np.sqrt(1 - 1/a)
y = np.sqrt(1 - 1/b)

coeffs = Zhao_epoch_coeffs[satellite][k][mu].values
alphaofK_data = alphaofK[satellite]['AlphaofK'].values
P = define_Legendre(alphaofK_data[:,1])
PAD_models = np.sum(coeffs * P, axis=1) + 1
P_int = define_Legendre_Int(x,y)
PAD_integral = np.array(2 * 2*np.pi * (np.sum(coeffs * P_int, axis=1) + (P0_int(y) - P0_int(x))))


local90PA = storm_data[satellite]['local90PA']
loss_cone = storm_data[satellite]['loss_cone']
alphaofK_data = alphaofK[satellite]['AlphaofK'].values
alpha_mask = (alphaofK_data[:,i_K].flatten() > loss_cone) & (alphaofK_data[:,i_K].flatten() <= local90PA)
fig, ax = plt.subplots(figsize=(8, 5))
for i_Mu in range(len(Mu_set)):
    mu = Mu_set[i_Mu]
    coeffs = Zhao_epoch_coeffs[satellite][k][mu].values
    coeff_mask = np.sum(coeffs,axis=1) != 0
    mask = coeff_mask & alpha_mask
    ax.scatter(local90PA[mask], scale_factor[satellite][k].values[mask,i_Mu],
               label=f'Mu = {mu:.2f}')
#ax.set_xlim(min(alpha_list), max(alpha_list))
ax.legend(title='Mu Values', loc='best', fontsize=10)
ax.set_xlabel(r'Local Pitch Angle $\alpha$')
ax.set_ylabel(r'Scale Factor')

# %%
vis_E = energyofmualpha[satellite][k]
vis_fluxE = flux_energyofmualpha[satellite][k]
import pandas as pd
import scipy
energy_channels = storm_data[satellite]['Energy_Channels']
vis_fluxdata = pd.DataFrame(storm_data[satellite]['electron_diff_flux'],columns=energy_channels)

energies = energyofmualpha[satellite][k].values[:,i_mu]
efitpars = storm_data[satellite]['efitpars']
n1      = efitpars[:,0]     # number density of MJ1
T1      = efitpars[:,1]     # temperature of MJ1
n2      = efitpars[:,2]     # number density of MJ2
T2      = efitpars[:,3]     # temperature of MJ2
n3      = efitpars[:,4]     # number density of MJ3
T3      = efitpars[:,5]     # temperature of MJ3
nG      = efitpars[:,6]     # number density of Gaussian
muG     = efitpars[:,7]     # reletavistic momentum at Gaussian peak
sigma   = efitpars[:,8]     # standard deviation of Gaussian
c_cms = sc.c * 10**2
p = np.sqrt((energies + E0)**2 - E0**2)# reletavistic momentum in MeV/c
K2 = np.array(scipy.special.kn(2, E0/T1)) # modified Bessel function of the second kind
j_MJ = n1 * c_cms /(4*np.pi*T1*K2*np.exp(E0/T1)) * p**2/E0**2 * np.exp(-energies/T1)
# %%
