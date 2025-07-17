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
from GPS_PSD_func import (QinDenton_period, import_GPS, data_period, QinDenton_period, data_from_gps, load_data,
                            AlphaOfK, MuofEnergyAlpha, EnergyofMuAlpha, energy_spectra, find_psd, find_Lstar)
import Zhao2018_PAD_Model
importlib.reload(Zhao2018_PAD_Model)
from Zhao2018_PAD_Model import (import_Zhao_coeffs, find_Zhao_PAD_coeffs, create_PAD, PAD_Scale_Factor)

#%% Global Variables
textsize = 16
Re = 6378.137 #Earth's Radius
Mu_set = np.array((4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G
K_set = np.array((0.1,1)) # R_E*G^(1/2)

# Conversions
# electron mass in MeV is (m_e [kg] * c^2 [m^2/s^2]) [J] / (sc.eV [J/eV] * 10^6 [eV/MeV])
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6) # this is m_0*c^2
# b_satellite and b_equator are in Gauss: 1 G = 10^5 nT

base_save_folder = "/home/will/GPS_data/april2017storm/"
extMag = 'T89'

start_date  = "04/21/2017"
stop_date   = "04/26/2017" # exclusive, end of the last day you want to see

QD_storm_data = QinDenton_period(start_date, stop_date)

Zhao_coeffs = import_Zhao_coeffs()

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
    raw_data_load.close()

### Preprocessing ###    
    # Restrict to time period
    #storm_data_raw = data_period(loaded_data, start_date, stop_date)
    
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
    storm_data_load.close()

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
    for satellite, sat_data in storm_data.items():
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        alphaofK[satellite] = pd.DataFrame(alphaofK[satellite], index=epoch_str, columns=K_set)
    alphaofK_load.close()
    
### Find Energies from Mu and AlphaofK ###
    # Find Mu spread of energy channels
    muofenergyalpha, Mu_bounds = MuofEnergyAlpha(storm_data, alphaofK)
    '''
    Mu_set = np.unique(np.sort(np.concatenate(([Mu_bounds['Rounded'][0]], 
                       np.logspace(np.log10(Mu_bounds['Rounded'][0]),
                                   np.log10(math.ceil(Mu_bounds['Rounded'][1]/2000)*2000), 
                                   math.ceil(Mu_bounds['Rounded'][1]/2000)-math.floor(Mu_bounds['Rounded'][0]/2000)+1)[1:]))))
    '''
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
    Zhao_epoch_coeffs_load.close()
    
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
    PAD_models_load.close()
    
    #--- Find Scale Factor from alphaofK and PAD Model ---#
    scale_factor = PAD_Scale_Factor(storm_data,Zhao_epoch_coeffs,alphaofK)

### Find Flux at Set Energy ###
    flux_energyofmualpha = energy_spectra(storm_data, energyofmualpha)

### Find Flux at Set Pitch Angle and Energy ###
    flux = {}
    for satellite, sat_data in storm_data.items():
        flux[satellite] = {}
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        for i_K, K_value in enumerate(K_set):
            flux[satellite][K_value] = flux_energyofmualpha[satellite][K_value].values * scale_factor[satellite][K_value].values
            flux[satellite][K_value] = pd.DataFrame(flux[satellite][K_value], index = epoch_str, columns=Mu_set)

### Calculated PSD ###
    psd = find_psd(flux,energyofmualpha)

### Calculate Lstar ###
    storm_data_complete = find_Lstar(storm_data,alphaofK)

    complete_filename = f"storm_data_complete.npz"
    complete_save_path = os.path.join(base_save_folder, complete_filename)
    '''
    # Save Data for later recall:
    print("Saving Processed GPS Data...")
    np.savez(complete_save_path, **storm_data_complete )
    print("Data Saved \n")
    '''
    # Read in data from previous save
    complete_load = np.load(complete_save_path, allow_pickle=True)
    storm_data_complete = load_data(complete_load)
    complete_load.close()

#%% Test PAD Integral
satellite = 'ns63'
k = 0.1
i_K = np.where(K_set == k)[0]
i_mu = 2
mu = Mu_set[i_mu]
i_epoch = 180

Zhao_test_coeffs = Zhao_epoch_coeffs[satellite][k][mu].values
b_sat = storm_data[satellite]['b_satellite']
b_eq = storm_data[satellite]['b_equator']
b_fpt = storm_data[satellite]['b_footpoint']
b_min = storm_data[satellite]['b_min']

coeffs = Zhao_epoch_coeffs[satellite][k][mu].values
alphaofK_data = alphaofK[satellite].values

local90PA = storm_data[satellite]['local90PA']
loss_cone = storm_data[satellite]['loss_cone']
alpha_mask = (alphaofK_data[:,i_K].flatten() >= loss_cone) & (alphaofK_data[:,i_K].flatten() <= local90PA)
fig, ax = plt.subplots(figsize=(14, 8))
color_set = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(Mu_set), dtype=int)]
for i_Mu in range(len(Mu_set)):
    mu_temp = Mu_set[i_Mu]
    coeffs_temp = Zhao_epoch_coeffs[satellite][k][mu_temp].values
    coeff_mask = np.sum(coeffs_temp,axis=1) != 0
    mask = coeff_mask & alpha_mask
    ax.scatter(local90PA[mask], scale_factor[satellite][k].values[mask,i_Mu],
               label=f'Mu = {mu_temp:.2f}',color=color_set[i_Mu])
#ax.set_xlim(min(alpha_list), max(alpha_list))
ax.legend(title='Mu Values', loc='upper right', fontsize=12, title_fontsize=12)
ax.set_xlabel(r'Local Pitch Angle $\alpha$',fontsize=12)
ax.set_ylabel(r'Scale Factor',fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)

#%%
def Zhao_Integral_batch(a, C):
    P = np.array([
        2 / a -       2 / a**2,
        2 / a -  20 / 3 / a**2 +  14 / 3 / a**3,
        2 / a -      14 / a**2 + 126 / 5 / a**3 -    66 / 5 / a**4,
        2 / a -      24 / a**2 + 396 / 5 / a**3 - 3432 / 35 / a**4 +    286 / 7 / a**5,
        2 / a - 110 / 3 / a**2 + 572 / 3 / a**3 -  2860 / 7 / a**4 + 24310 / 63 / a**5 - 8398 / 63 / a**6
    ]) * a / 2

    return np.sum(C * P.transpose(), axis=1) + 1

from scipy.special import legendre
def Zhao_batch(alpha, C):
    alpha = np.deg2rad(alpha)
    P = np.array([legendre(2)(np.cos(alpha)), legendre(4)(np.cos(alpha)), legendre(6)(np.cos(alpha)),
                  legendre(8)(np.cos(alpha)), legendre(10)(np.cos(alpha))]).transpose()
    return np.sum(C * P, axis=1) + 1

C_shape = Zhao_Integral_batch(b_sat / b_eq, coeffs)
P_shape = Zhao_batch(alphaofK_data[:,i_K].flatten(),coeffs)
norm = P_shape / C_shape

fig, ax = plt.subplots(figsize=(14, 8))
color_set = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(Mu_set), dtype=int)]
ax.scatter(local90PA[mask], norm[mask],
            label=f'with loss cone',color='C1',marker='s')
ax.scatter(local90PA[mask], scale_factor[satellite][k].values[mask,i_mu],
            label=f'without loss cone',color='C0')
#ax.set_xlim(min(alpha_list), max(alpha_list))
ax.legend(title=f'Model for Mu = {mu:.2f}', loc='upper right', fontsize=12, title_fontsize=12)
ax.set_xlabel(r'Local Pitch Angle $\alpha$',fontsize=12)
ax.set_ylabel(r'Scale Factor',fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)

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


#%% Plot PSD lineplots
fig, ax = plt.subplots(figsize=(5.5, 5))
color_set = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(Mu_set), dtype=int)]
color_set[3] = [0, 1, 1, 1]  # Teal

time_start  = dt.datetime(2017, 4, 23, 18, 45, 0)
time_stop   = dt.datetime(2017, 4, 23, 22, 58, 0)
    
#time_start  = dt.datetime(2017, 4, 24, 17, 7, 0)
#time_stop   = dt.datetime(2017, 4, 24, 21, 35, 0)
    
#time_start  = dt.datetime(2017, 4, 25, 15, 30, 0)
#time_stop   = dt.datetime(2017, 4, 25, 19, 57, 0)

for satellite, sat_data in storm_data.items():
    time_mask = (storm_data[satellite]['Epoch'].UTC >= time_start) & (storm_data[satellite]['Epoch'].UTC < time_stop)
    MLT_mask = []
    for i_Mu in range(len(Mu_set)):
        mu_temp = Mu_set[i_Mu]
        if satellite == 'ns64':
            ax.scatter(storm_data[satellite]['L_LGM_T89IGRF'][time_mask], psd[satellite][k].values[time_mask,i_Mu],
                label=f'{mu_temp:.0f}',color=color_set[i_Mu])
        else:
            ax.scatter(storm_data[satellite]['L_LGM_T89IGRF'][time_mask], psd[satellite][k].values[time_mask,i_Mu],color=color_set[i_Mu])
ax.set_ylim(1e-13, 1e-5)
plt.yscale('log')
ax.grid()
ax.legend(title='Mu Values', loc='center right', bbox_to_anchor=(1.3, 0.5),fontsize=12, title_fontsize=12)
ax.set_xlabel(r'McIlwain L',fontsize=12)
ax.set_ylabel(r'PSD',fontsize=12)
# Set the plot title to the time interval
title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
ax.set_title(title_str)
# %%
