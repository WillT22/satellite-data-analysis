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
from REPT_PSD_func import (process_l3_data, time_average, find_mag, Average_FluxbyPA, Interp_Flux)

#%% Global Variables
textsize = 16
Re = 6378.137 #Earth's Radius
Mu_set = np.array((2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G
K_set = np.array((0.1,1,2)) # R_E*G^(1/2)
mode = 'save' # 'save' or 'load'

input_folder = "/home/wzt0020/REPT_data/april2017storm/"
base_save_folder = "/home/wzt0020/REPT_data/april2017storm/"
extMag = 'T89c'

start_date  = dt.datetime(2017, 4, 21, 00, 00, 0)
stop_date   = dt.datetime(2017, 4, 26, 00, 00, 0)

# start_date = dt.datetime(2018, 8, 25, 0, 0, 0)
# stop_date = dt.datetime(2018, 8, 28, 0, 0, 0)

# start_date  = dt.datetime(2012, 10, 7, 00, 00, 0)
# stop_date   = dt.datetime(2012, 10, 11, 00, 00, 0)

# Conversions
# electron mass in MeV is (m_e [kg] * c^2 [m^2/s^2]) [J] / (sc.eV [J/eV] * 10^6 [eV/MeV])
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6) # this is m_0*c^2
# b_satellite and b_equator are in Gauss: 1 G = 10^5 nT

# Import
QD_storm_data = QinDenton_period(start_date, stop_date)

#%% Main
if __name__ == '__main__':

### Load in data ###
    raw_save_path = os.path.join(base_save_folder, 'raw_rept.npz')
    if mode == 'save':
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Error: Folder path not found: {input_folder}")
        
        # Get all CDF file paths in the folder
        file_paths_l3_A = glob.glob(input_folder + "rbspa*[!r]*.cdf") 
        file_paths_l3_B = glob.glob(input_folder + "rbspb*[!r]*.cdf")
        
        REPT_data_raw = {}
        REPT_data_raw['rbspa'] = process_l3_data(file_paths_l3_A)
        REPT_data_raw['rbspb'] = process_l3_data(file_paths_l3_B)
    
        #Save Data for later recall:
        print("Saving Raw REPT Data...")
        np.savez(raw_save_path, **REPT_data_raw)
        print("Data Saved \n")
    elif mode == 'load':
        # Read in data from previous save
        raw_data_load = np.load(raw_save_path, allow_pickle=True)
        REPT_data_raw = load_data(raw_data_load)
        raw_data_load.close()
        del raw_data_load

### Restric Time Period ###
    REPT_data = {}
    for satellite, sat_data in REPT_data_raw.items():
        print(f'Restricting Time Period for satellite {satellite}...')
        REPT_data[satellite] = data_period(sat_data, start_date, stop_date)
    del REPT_data_raw

### Average fluxes within a minute ###
    for satellite, sat_data in REPT_data.items():
        print(f"Time Averaging Fluxes for satellite {satellite}...")
        REPT_data[satellite] = time_average(sat_data, satellite)

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
    alphaofK_filename = f"alphaofK_{extMag}.npz"
    alphaofK_save_path = os.path.join(base_save_folder, alphaofK_filename)
    if mode == 'save':
        alphaofK = {}
        for satellite, sat_data in REPT_data.items():
            print(f"Calculating Pitch Angle for satellite {satellite}...")
            alphaofK[satellite] = AlphaOfK(sat_data, K_set, extMag)

        # Save Data for later recall:
        print("Saving AlphaofK Data...")
        np.savez(alphaofK_save_path, **alphaofK)
        print("Data Saved \n")
    elif mode == 'load':
        # Load data from previous save
        alphaofK_load = np.load(alphaofK_save_path, allow_pickle=True)
        alphaofK = load_data(alphaofK_load)
        for satellite, sat_data in REPT_data.items():
            epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
            alphaofK[satellite] = pd.DataFrame(alphaofK[satellite], index=epoch_str, columns=np.atleast_1d(K_set))
        alphaofK_load.close()
        del alphaofK_load

    # Read in data from previous save
    save_path = os.path.join(base_save_folder, 'rept_data.npz')
    if mode == 'load':
        complete_load = np.load(save_path, allow_pickle=True)
        REPT_data = load_data(complete_load)
        complete_load.close()
        del complete_load

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
    for satellite, sat_data in REPT_data.items():
        print(f"Calculating PSD for satellite {satellite}")
        REPT_data[satellite]['PSD'] = find_psd(flux[satellite], energyofmualpha[satellite])

### Calculate L* ####
    new_save_path = os.path.join(base_save_folder, 'rept_data.npz')
    if mode == 'save':
        for satellite, sat_data in REPT_data.items():
            print(f"Calculating L* for satellite {satellite}...")
            REPT_data[satellite] = find_Lstar(sat_data, alphaofK[satellite], extMag='T89c')

        # Save Data for later recall:
        print("Saving REPT Data...")
        np.savez(new_save_path, **REPT_data)
        print("Data Saved \n")

#%% Plot PSD
from matplotlib import colors
k = 0.1
i_K = np.where(K_set == k)[0]
mu = 8000
i_mu = np.where(Mu_set == mu)[0]

fig, ax = plt.subplots(figsize=(16, 4))

colorscheme = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))
cmap = colors.ListedColormap(colorscheme)

# Logarithmic colorbar setup
min_val = np.nanmin(np.log10(1e-12))
max_val = np.nanmax(np.log10(1e-7))

for satellite, sat_data in REPT_data.items():
    psd_plot = REPT_data[satellite]['PSD'][k].values[:,i_mu].copy().flatten()
    psd_mask = (psd_plot > 0) & (psd_plot != np.nan)
    lstar_mask = sat_data['Lstar'][:,0]>0
    combined_mask = psd_mask & lstar_mask

    # Plotting, ignoring NaN values in the color
    scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], sat_data['Lstar'][combined_mask,i_K],
                        c=np.log10(psd_plot[combined_mask]), cmap=cmap, vmin=min_val, vmax=max_val)


ax.set_title(f"RBSP A&B REPT, K={k:.1f} $G^{{1/2}}R_E$, $\\mu$={mu:.0f} $MeV/G$", fontsize=textsize + 2)
ax.set_ylabel(r"L*", fontsize=textsize)
ax.tick_params(axis='both', labelsize=textsize, pad=10)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
# Force labels for first and last x-axis tick marks 
min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
ax.set_xlim(min_epoch, max_epoch)
ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
ax.set_ylim(3, 5.5)
ax.grid(True)

cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
tick_locations = np.arange(min_val, max_val + 1)
cbar.set_ticks(tick_locations)
cbar.set_label(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize)
cbar.ax.tick_params(labelsize=textsize)

plt.xticks(fontsize=textsize)
plt.subplots_adjust(top=0.82, right=0.95) 

plt.show()

#%% Create PSD Radial Profiles
sat_select = 'rbspb'
sat_data = REPT_data[sat_select]
k = 0.1

time_start  = dt.datetime(2012, 10, 9, 6, 00, 0)
time_stop   = dt.datetime(2012, 10, 9, 10, 00, 0)

# time_start  = dt.datetime(2017, 4, 23, 18, 45, 0)
# time_stop   = dt.datetime(2017, 4, 23, 22, 58, 0)

# time_start  = dt.datetime(2017, 4, 24, 17, 7, 0)
# time_stop   = dt.datetime(2017, 4, 24, 21, 35, 0)

# time_start  = dt.datetime(2017, 4, 25, 15, 30, 0)
# time_stop   = dt.datetime(2017, 4, 25, 19, 57, 0)

time_mask = (sat_data['Epoch'] >= time_start) & (sat_data['Epoch'] <= time_stop)

lstar_delta = 0.1
lstar_mask = (sat_data['Lstar'][:,0][time_mask]>0)
lstar_range = sat_data['Lstar'][:,0][time_mask][lstar_mask]
lstar_min = np.min(lstar_range[~np.isnan(lstar_range)])
lstar_max = np.max(lstar_range[~np.isnan(lstar_range)])
lstar_intervals = np.arange(np.floor(lstar_min / lstar_delta) * lstar_delta, np.ceil(lstar_max / lstar_delta) * lstar_delta + lstar_delta, lstar_delta)

# Initialize arrays to store averaged values.
averaged_lstar = np.zeros(len(lstar_intervals))
averaged_psd = np.zeros((len(lstar_intervals), len(sat_data['PSD'][k].values[:,0].flatten())))

fig, ax = plt.subplots(figsize=(6, 4.5))
color_set = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(Mu_set), dtype=int)]
color_set[3] = [0, 1, 1, 1]  # Teal

for mu_index in range(len(Mu_set)):
    for i, lstar_val in enumerate(lstar_intervals):
        # Find indices within the current Lstar interval and time range.
        lstar_start = lstar_val - 1/2 * lstar_delta
        lstar_end = lstar_val + 1/2 * lstar_delta
        interval_indices = np.where(time_mask & (sat_data['Lstar'][:,0] >= lstar_start) & (sat_data['Lstar'][:,0] < lstar_end))[0]
        
        # Calculate averages for the current Lstar interval
        averaged_psd[i, mu_index] = np.nanmean(sat_data['PSD'][k].values[interval_indices,mu_index].flatten())  # average along the first axis, ignoring NaNs.

    
    # Create a mask to filter out NaN values
    psd_mask = (averaged_psd[:,mu_index] > 0) & (averaged_psd[:,mu_index] != np.nan)
    
    # Apply the mask to both averaged_lstar and averaged_psd
    ax.plot(
        lstar_intervals[psd_mask],
        averaged_psd[psd_mask,mu_index],
        color=color_set[mu_index],
        linewidth=2,
        marker='o',
        markersize=4,
        label=f"{Mu_set[mu_index]:.0f}"
        )

ax.set_xlim(3, 5.5)
ax.set_xlabel(r"L*", fontsize=textsize - 2)
ax.set_ylim(1e-13, 1e-5)
ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize - 2)
plt.yscale('log')
ax.grid(True)

# Add legend
ax.legend(
    title=r"$\mu$ (MeV/G)",
    loc='center right',
    bbox_to_anchor=(1.25, 0.5),
    fontsize='small', #adjust legend fontsize
    title_fontsize='medium', #adjust legend title fontsize
    markerscale=0.7,
    handlelength=1
)

# Add K text to the plot
ax.text(0.02, 0.98, r"K = " + f"{K_set:.1f} $G^{{1/2}}R_E$", transform=ax.transAxes, fontsize=textsize-4, verticalalignment='top') #add the text

# Set the plot title to the time interval
title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
ax.set_title(title_str)

plt.tight_layout()
plt.show()
