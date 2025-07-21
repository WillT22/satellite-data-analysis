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

#start_date  = dt.datetime(2017, 4, 21, 00, 00, 0)
#stop_date   = dt.datetime(2017, 4, 26, 00, 00, 0)

start_date = dt.datetime(2017, 4, 24, 17, 7, 0)
stop_date = dt.datetime(2017, 4, 24, 21, 35, 0)#

QD_storm_data = QinDenton_period(start_date, stop_date)

Zhao_coeffs = import_Zhao_coeffs()

#%% Main
if __name__ == '__main__':

    # Load in data
    input_folder = "/home/will/GPS_data/april2017storm/"
    # Be mindful of ns60 and ns69 data as they have poorer fits and more noise
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
    storm_data_raw = data_period(loaded_data, start_date, stop_date)
    
    # Limit to relevant Lshells, convert satellite position from spherical GEO to GSM and extract relevant data
    # (Takes a few minutes)
    storm_data = data_from_gps(storm_data_raw, Lshell=6, extMag= 'T89')
    
    processed_save_path = os.path.join(base_save_folder, 'processed_gps.npz')
    '''
    # Save Data for later recall:
    print("Saving Processed GPS Data...")
    np.savez(processed_save_path, **storm_data)
    print("Data Saved \n")
    '''
    '''
    # Read in data from previous save
    storm_data_load = np.load(processed_save_path, allow_pickle=True)
    storm_data = load_data(storm_data_load)
    storm_data_load.close()
    '''
### Find Pitch Angles ###
    # Find pitch angle corresponding to set K
    alphaofK = AlphaOfK(storm_data, K_set, extMag)

    alphaofK_filename = f"alphaofK_{extMag}.npz"
    alphaofK_save_path = os.path.join(base_save_folder, alphaofK_filename)
    '''
    # Save Data for later recall:
    print("Saving AlphaofK Data...")
    np.savez(alphaofK_save_path, **alphaofK)
    print("Data Saved \n")
    '''
    '''
    # Load data from previous save
    alphaofK_load = np.load(alphaofK_save_path, allow_pickle=True)
    alphaofK = load_data(alphaofK_load)
    for satellite, sat_data in storm_data.items():
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        alphaofK[satellite] = pd.DataFrame(alphaofK[satellite], index=epoch_str, columns=K_set)
    alphaofK_load.close()
    '''
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
    Zhao_epoch_coeffs = find_Zhao_PAD_coeffs(storm_data, QD_storm_data, energyofmualpha)

    Zhao_epoch_coeffs_filename = f"Zhao_epoch_coeffs.npz"
    Zhao_epoch_coeffs_save_path = os.path.join(base_save_folder, Zhao_epoch_coeffs_filename)
    '''
    # Save Data for later recall:
    print("Saving Zhao coefficients for each Epoch...")
    np.savez(Zhao_epoch_coeffs_save_path, **Zhao_epoch_coeffs)
    print("Data Saved \n")
    '''
    '''
    # Load data from previous save
    Zhao_epoch_coeffs_load = np.load(Zhao_epoch_coeffs_save_path, allow_pickle=True)
    Zhao_epoch_coeffs = load_data(Zhao_epoch_coeffs_load)
    Zhao_epoch_coeffs_load.close()
    '''
    #--- Create Pitch Angle Distribution (PAD) from Coefficients ---
    PAD_models = create_PAD(storm_data, Zhao_epoch_coeffs, alphaofK)

    PAD_models_filename = f"PAD_models.npz"
    PAD_models_save_path = os.path.join(base_save_folder, PAD_models_filename)
    '''
    # Save Data for later recall:
    print("Saving PAD models ...")
    np.savez(PAD_models_save_path, **PAD_models)
    print("Data Saved \n")
    '''
    ''''
    # Load data from previous save
    PAD_models_load = np.load(PAD_models_save_path, allow_pickle=True)
    PAD_models = load_data(PAD_models_load)
    PAD_models_load.close()
    '''
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
    '''
    # Read in data from previous save
    complete_load = np.load(complete_save_path, allow_pickle=True)
    storm_data_complete = load_data(complete_load)
    complete_load.close()
    '''

#%% Plot PSD lineplots
k = 0.1
i_K = np.where(K_set == k)[0]
mu = 12000
i_mu = np.where(Mu_set == mu)[0]

REPTB_load = np.load('/mnt/box/Multipoint_Box/REPT_Data/plot_data.npz', allow_pickle=True)
REPTB_data = load_data(REPTB_load)
REPTB_load.close()

#time_start  = dt.datetime(2017, 4, 23, 19, 30, 0)
#time_stop   = dt.datetime(2017, 4, 23, 23, 00, 0)
    
time_start  = dt.datetime(2017, 4, 24, 17, 7, 0)
time_stop   = dt.datetime(2017, 4, 24, 21, 35, 0)
   
#time_start  = dt.datetime(2017, 4, 25, 15, 30, 0)
#time_stop   = dt.datetime(2017, 4, 25, 19, 50, 0)

#time_start  = dt.datetime(2017, 4, 21, 10, 16, 0)
#time_stop   = dt.datetime(2017, 4, 21, 13, 46, 0)

#time_start  = dt.datetime(2017, 4, 21, 0, 0, 0)
#time_stop   = dt.datetime(2017, 4, 26, 0, 0, 0)

# Convert Epoch_A and Epoch_B to NumPy arrays of datetimes
Epoch_B_np = np.array(REPTB_data['Epoch_B_averaged'])

# Define Lstar delta
lstar_delta = 0.1

# Generate Lstar interval boundaries within the time range.
time_range = Epoch_B_np[(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
lstar_range = REPTB_data['Lstar_B_set'][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
psd_range = REPTB_data['psd_B'][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]

energy_range = REPTB_data['energy_B_set'][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
interpa_range = REPTB_data['FEDU_B_interpa'][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
interpaE_range = REPTB_data['FEDU_B_interpaE'][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]

fig, ax = plt.subplots(figsize=(12, 8))
colormap_name = 'viridis_r' # Good choice for continuous data
cmap = plt.cm.get_cmap(colormap_name)

import matplotlib.dates as mdates
time_range_timestamps = mdates.date2num(time_range)

from matplotlib import colors
vmin = mdates.date2num(time_start) #- dt.timedelta(minutes=(time_start.minute % 30))
vmax = mdates.date2num(time_stop ) #+ dt.timedelta(minutes=30 - (time_stop.minute % 30))
norm = colors.Normalize(vmin=vmin,
                        vmax=vmax)

# Apply the mask to both averaged_lstar and averaged_psd
scatter_plot = ax.scatter(
    lstar_range,
    psd_range[:, i_mu],
    c=time_range_timestamps, # Color by Epoch datetime objects
    cmap=cmap,
    norm=norm,
    marker='o')

cbar = fig.colorbar(scatter_plot, ax=ax, orientation='horizontal', pad=0.1)
cbar.set_label('Time (UTC)', fontsize=plt.rcParams['axes.labelsize'])
cbar.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))

rounded_dt_obj = time_range[0] + dt.timedelta(minutes=10 - (time_range[0].minute % 10))
rounded_dt_obj = rounded_dt_obj.replace(second=0, microsecond=0)
# Generate sequence by iteratively adding timedelta
half_hours = []
current_point = rounded_dt_obj  
while current_point < time_stop:
    half_hours.append(current_point)
    current_point += dt.timedelta(minutes=30)  
half_hours = np.array(half_hours, dtype=object)

for dt_obj in half_hours:
    if dt_obj.second >= 30:
        dt_obj = dt_obj + dt.timedelta(minutes=1)
        dt_obj.replace(second=0, microsecond=0)
half_hours = np.array([dt_obj.replace(tzinfo=None) for dt_obj in half_hours], dtype=object)
color_set = plt.cm.get_cmap('viridis_r')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(half_hours), dtype=int)]

sat_valid = np.zeros((len(storm_data_complete),len(half_hours)),dtype=bool)
avg_Lstar = np.zeros((len(storm_data_complete),len(half_hours)))
avg_psd = np.zeros((len(storm_data_complete),len(half_hours)))
for i_hh, half_hour in enumerate(half_hours):
    for satellite, sat_data in storm_data_complete.items():
        time_mask = (sat_data['Epoch'].UTC >= (half_hour-dt.timedelta(minutes=15))) & (sat_data['Epoch'].UTC < (half_hour+dt.timedelta(minutes=15)))
        if sum(time_mask) == 0:
            continue

        nearest_time = np.zeros(sum(time_mask),dtype=int)
        for i_epoch, epoch in enumerate(sat_data['Epoch'].UTC[time_mask]):
            nearest_time[i_epoch] = np.argmin(np.abs(REPTB_data['Epoch_B_averaged']-epoch))

        MLT_mask = ((sat_data['MLT'][time_mask] >= (REPTB_data['MLT_B'][nearest_time]-1.5)) 
                    & (sat_data['MLT'][time_mask] < (REPTB_data['MLT_B'][nearest_time]+1.5)))
        valid_mask = MLT_mask
        
        if np.sum(valid_mask)>0:
            i_sat = list(storm_data_complete.keys()).index(satellite)
            psd_masked = psd[satellite][k].values[time_mask,i_mu][valid_mask]
            psd_masked = psd_masked[~np.isnan(psd_masked)]
            if len(psd_masked) > 0:
                sat_valid[i_sat,i_hh] = True
                avg_Lstar[i_sat,i_hh] = np.average(sat_data['Lstar'][time_mask,i_K][valid_mask])
                avg_psd[i_sat,i_hh] = np.exp(np.nanmean(np.log(psd_masked)))
    color_for_point = cmap(norm(mdates.date2num(half_hour)))
    ax.plot(avg_Lstar[sat_valid[:,i_hh], i_hh], avg_psd[sat_valid[:,i_hh], i_hh],
                marker='*', markersize=12,
                color=color_for_point) # Use the calculated color
                #label=half_hour.strftime("%d-%m-%Y %H:%M")) # Label for each star
sat_valid = pd.DataFrame(sat_valid,index=list(storm_data_complete.keys()),columns=half_hours)


#ax.set_xlim(3, 5.5)
ax.set_xlabel(r"L*", fontsize=textsize - 2)
ax.set_ylim(1e-13, 1e-5)
ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize - 2)
plt.yscale('log')
ax.grid(True)

# Add K and Mu text to the plot
ax.text(0.02, 0.98, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$", transform=ax.transAxes, fontsize=textsize-4, verticalalignment='top') #add the text

# Add legend
ax.legend(
    title=r"Time",
    loc='center right',
    bbox_to_anchor=(1.2, 0.5),
    markerscale=0.7,
    handlelength=1
)

# Set the plot title to the time interval
title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
ax.set_title(title_str)

plt.show()

# %% Let's see why PSD is different:
satellite = 'ns67'
sat_data = storm_data_complete[satellite]
time_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC < time_stop)

storm_time_data = {}
for item, item_data in storm_data_complete[satellite].items():
    if item == 'Epoch':
        storm_time_data[item] = item_data.UTC[time_mask]
    elif item == 'Energy_Channels':
        storm_time_data[item] = item_data[0]
    else:
        storm_time_data[item] = item_data[time_mask]

storm_time_data['alpha'] = alphaofK[satellite][0.1][time_mask]
storm_time_data['alpha_local'] = np.rad2deg(np.arcsin(np.sqrt(sat_data['b_equator'][time_mask]/sat_data['b_satellite'][time_mask])*np.sin(np.deg2rad(storm_time_data['alpha']))))
storm_time_data['alpha_ratio'] = storm_time_data['b_equator']/np.sin(np.deg2rad(alphaofK[satellite][0.1][time_mask]))**2
storm_time_data['Flux'] = flux[satellite][0.1].values[time_mask]


nearest_time = np.zeros(sum(time_mask),dtype=int)
for i_epoch, epoch in enumerate(sat_data['Epoch'].UTC[time_mask]):
    nearest_time[i_epoch] = np.argmin(np.abs(REPTB_data['Epoch_B_averaged']-epoch))

REPTB_test = {}
for item, item_data in REPTB_data.items():
    REPTB_test[item] = item_data[nearest_time]

REPTB_test['Blocal_B'] = REPTB_test['Blocal_B'] * 1e-5
REPTB_test['alpha_ratio'] = REPTB_test['Blocal_B']/np.sin(np.deg2rad(REPTB_test['alpha_B_set']))**2
REPTB_test['alpha_eq'] = np.rad2deg(np.arcsin(np.sqrt(REPTB_test['Blocal_B']/sat_data['b_equator'][time_mask])*np.sin(np.deg2rad(REPTB_test['alpha_B_set']))))

struct_temp = {}
struct_temp['REPT_B'] = {}
from spacepy.time import Ticktock
struct_temp['REPT_B']['Epoch'] = Ticktock(REPTB_test['Epoch_B_averaged'])
from spacepy.coordinates import Coords
struct_temp['REPT_B']['Position'] = Coords(REPTB_test['Position_B_averaged'])
aofK_REPTB = AlphaOfK(struct_temp,K_set)
# %%
