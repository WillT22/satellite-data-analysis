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
from matplotlib import colors
import pandas as pd

import importlib
import GPS_PSD_func
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import (QinDenton_period, import_GPS, data_period, data_from_gps, load_data,
                            AlphaOfK, MuofEnergyAlpha, EnergyofMuAlpha, energy_spectra, find_psd, find_Lstar)
import Zhao2018_PAD_Model
importlib.reload(Zhao2018_PAD_Model)
from Zhao2018_PAD_Model import (import_Zhao_coeffs, find_Zhao_PAD_coeffs, create_PAD, PAD_Scale_Factor)

import time

#%% Global Variables
textsize = 16
Re = 6378.137 #Earth's Radius
Mu_set = np.array((2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G
K_set = np.array((0.1,1,2)) # R_E*G^(1/2)
mode = 'load' # 'save' or 'load'
storm_name = 'april2017storm'
plot_flux = True
plot_psd = True
plot_combined_psd = True
plot_radial = True

GPS_data_root = '/home/wzt0020/GPS_data/'
input_folder = os.path.join(GPS_data_root, storm_name)
base_save_folder = os.path.join(GPS_data_root, storm_name)
extMag = 'TS04'

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

start_time = time.perf_counter()

# Import
QD_storm_data = QinDenton_period(start_date, stop_date)
Zhao_coeffs = import_Zhao_coeffs()

#%% Main
if __name__ == '__main__':

### Load in data ###
    # Be mindful of ns60 and ns69 data as they have poorer fits and more noise
    raw_save_path = os.path.join(base_save_folder, 'raw_gps.npz')
    if mode == 'save':
        loaded_data = import_GPS(input_folder)
        # Save Data for later recall:
        print("Saving Raw GPS Data...")
        np.savez(raw_save_path, **loaded_data)
        print("Data Saved \n")
    elif mode == 'load':
        # Read in data from previous save
        raw_data_load = np.load(raw_save_path, allow_pickle=True)
        loaded_data = load_data(raw_data_load)
        raw_data_load.close()
        del raw_data_load
    
### Preprocessing ###        
    processed_save_path = os.path.join(base_save_folder, 'processed_gps.npz')
    if mode == 'save':
        # Restrict to time period
        storm_data_raw = {}
        for satellite, sat_data in loaded_data.items():
            print(f'Restricting Time Period for satellite {satellite}', end='\r')
            storm_data_raw[satellite] = data_period(sat_data, start_date, stop_date)
        del loaded_data

        # Limit to relevant Lshells, convert satellite position from spherical GEO to GSM and extract relevant data
        # (Takes a few minutes)
        print('Processing Data for each Satellite...')
        storm_data = data_from_gps(storm_data_raw, Lshell=6)
        del storm_data_raw

        # Save Data for later recall:
        print("Saving Processed GPS Data...")
        np.savez(processed_save_path, **storm_data)
        print("Data Saved \n")

    elif mode == 'load':
        # Read in data from previous save
        storm_data_load = np.load(processed_save_path, allow_pickle=True)
        storm_data = load_data(storm_data_load)
        storm_data_load.close()
        del storm_data_load

### Find Pitch Angles ###
    alphaofK_filename = f"alphaofK_{extMag}.npz"
    alphaofK_save_path = os.path.join(base_save_folder, alphaofK_filename)
    if mode == 'save':
        # Find pitch angle corresponding to set K
        alphaofK = {}
        for satellite, sat_data in storm_data.items():
            print(f"Calculating Pitch Angle for satellite {satellite}")
            alphaofK[satellite] = AlphaOfK(sat_data, K_set, extMag=extMag)

        # Save Data for later recall:
        print("Saving AlphaofK Data...")
        np.savez(alphaofK_save_path, **alphaofK)
        print("Data Saved \n")
    elif mode == 'load':   
        # Load data from previous save
        alphaofK_load = np.load(alphaofK_save_path, allow_pickle=True)
        alphaofK = load_data(alphaofK_load)
        for satellite, sat_data in storm_data.items():
            epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
            alphaofK[satellite] = pd.DataFrame(alphaofK[satellite], index=epoch_str, columns=K_set)
        alphaofK_load.close()
        del alphaofK_load
    
### Find Energies from Mu and AlphaofK ###
    # Find Mu spread of energy channels
    # muofenergyalpha, Mu_bounds = MuofEnergyAlpha(storm_data, alphaofK)
    '''
    Mu_set = np.unique(np.sort(np.concatenate(([Mu_bounds['Rounded'][0]], 
                       np.logspace(np.log10(Mu_bounds['Rounded'][0]),
                                   np.log10(math.ceil(Mu_bounds['Rounded'][1]/2000)*2000), 
                                   math.ceil(Mu_bounds['Rounded'][1]/2000)-math.floor(Mu_bounds['Rounded'][0]/2000)+1)[1:]))))
    '''
   
    energyofmualpha = {}
    for satellite, sat_data in storm_data.items():
        print(f"Calculating Energy of Mu and Alpha for satellite {satellite}", end='\r')
        energyofmualpha[satellite] = EnergyofMuAlpha(sat_data, Mu_set, alphaofK[satellite])
    

### Find Flux at Set Energy ###
    flux_energyofmualpha = energy_spectra(storm_data, energyofmualpha)

### Find Flux at Set Pitch Angle ####
    #--- Extract Zhao Coefficients at each Epoch ---
    Zhao_epoch_coeffs_filename = f"Zhao_epoch_coeffs.npz"
    Zhao_epoch_coeffs_save_path = os.path.join(base_save_folder, Zhao_epoch_coeffs_filename)
    if mode == 'save':
        Zhao_epoch_coeffs = find_Zhao_PAD_coeffs(storm_data, QD_storm_data, energyofmualpha)

        # Save Data for later recall:
        print("Saving Zhao coefficients for each Epoch...")
        np.savez(Zhao_epoch_coeffs_save_path, **Zhao_epoch_coeffs)
        print("Data Saved \n")
    elif mode == 'load': 
        # Load data from previous save
        Zhao_epoch_coeffs_load = np.load(Zhao_epoch_coeffs_save_path, allow_pickle=True)
        Zhao_epoch_coeffs = load_data(Zhao_epoch_coeffs_load)
        Zhao_epoch_coeffs_load.close()
        del Zhao_epoch_coeffs_load
    
    #--- Find Scale Factor from alphaofK and PAD Model ---#
    scale_factor = PAD_Scale_Factor(storm_data,Zhao_epoch_coeffs,alphaofK)

### Find Flux at Set Pitch Angle and Energy ###
    flux = {}
    for satellite, sat_data in storm_data.items():
        flux[satellite] = {}
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        for i_K, K_value in enumerate(K_set):
            flux[satellite][K_value] = flux_energyofmualpha[satellite][K_value].values * scale_factor[satellite][K_value].values
            flux[satellite][K_value] = pd.DataFrame(flux[satellite][K_value], index = epoch_str, columns=Mu_set)

### Calculate PSD ###
    psd = {}
    for satellite, sat_data in storm_data.items():
        print(f"Calculating PSD for satellite {satellite}", end='\r')
        psd[satellite] = find_psd(flux[satellite], energyofmualpha[satellite])

### Calculate Lstar ###)
    complete_filename = f"storm_data_T89c.npz"
    complete_save_path = os.path.join(base_save_folder, complete_filename)
    if mode == 'save':
        # ns57 has the first few values > Lshell = 6
        for satellite, sat_data in storm_data.items():
            print(f"Calculating L* for satellite {satellite}")
            storm_data[satellite] = find_Lstar(sat_data, alphaofK[satellite], extMag=extMag)

        # Save Data for later recall:
        print("Saving Processed GPS Data...")
        np.savez(complete_save_path, **storm_data )
        print("Data Saved \n")
    elif mode == 'load': 
        # Read in data from previous save
        complete_load = np.load(complete_save_path, allow_pickle=True)
        storm_data = load_data(complete_load)
        complete_load.close()
        del complete_load

### Execution time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    def format_runtime(elapsed_time):
        # Calculate whole hours
        hours = int(elapsed_time // 3600)
        # Calculate remaining minutes
        minutes = int((elapsed_time % 3600) // 60)
        # Calculate remaining seconds (including decimals)
        seconds = elapsed_time % 60
        
        return f"Script runtime: {hours}h {minutes}m {seconds:.2f}s"

    print(format_runtime(elapsed_time))

#%% Plot Flux
if plot_flux==True:
    energy = 5.0 # MeV
    energy_channels = storm_data['ns62']['Energy_Channels']
    i_energy = np.argmin(np.abs(energy_channels - energy))

    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1e2))
    max_val = np.nanmax(np.log10(1e5))

    fig, ax = plt.subplots(figsize=(16, 4))
    for satellite, sat_data in storm_data.items():
        flux_plot = sat_data['electron_diff_flux'][:,i_energy]
        flux_mask = (flux_plot > 0) & (flux_plot != np.nan)
        # Plotting, ignoring NaN values in the color
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[flux_mask], sat_data[f'L_LGM_{extMag}IGRF'][flux_mask],
                            c=np.log10(flux_plot[flux_mask]), vmin=min_val, vmax=max_val)

    ax.set_title(f"GPS CXD, {energy} $MeV$ Electron Differential Flux", fontsize=textsize + 2)
    ax.set_ylabel(r"McIlwain L", fontsize=textsize)
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
    cbar.set_label(label = r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)

    plt.show()


#%% Plot PSD
if plot_psd==True:
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    fig, ax = plt.subplots(figsize=(16, 4))

    colorscheme = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))
    cmap = colors.ListedColormap(colorscheme)

    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1e-9))
    max_val = np.nanmax(np.log10(1e-5))

    for satellite, sat_data in storm_data.items():
        psd_plot = psd[satellite][k].values[:,i_mu].copy().flatten()
        psd_mask = (psd_plot > 0) & (psd_plot != np.nan)
        # Plotting, ignoring NaN values in the color
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[psd_mask], sat_data['Lstar'][psd_mask,i_K],
                            c=np.log10(psd_plot[psd_mask]), cmap=cmap, vmin=min_val, vmax=max_val)


    ax.set_title(f"GPS CXD, K={k:.1f} $G^{{1/2}}R_E$, $\\mu$={mu:.0f} $MeV/G$", fontsize=textsize + 2)
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

#%% Plot PSD with REPT data
if plot_combined_psd==True:
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1e-9))
    max_val = np.nanmax(np.log10(1e-5))

    save_path = os.path.join(f'/home/wzt0020/REPT_data/{storm_name}/', f'rept_data_{extMag}.npz')
    complete_load = np.load(save_path, allow_pickle=True)
    REPT_data = load_data(complete_load)
    complete_load.close()
    del complete_load

    fig, ax = plt.subplots(figsize=(16, 4))

    colorscheme = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))
    cmap = colors.ListedColormap(colorscheme)

    for satellite, sat_data in storm_data.items():
        psd_plot = psd[satellite][k].values[:,i_mu].copy().flatten()
        psd_mask = (psd_plot > 0) & (psd_plot != np.nan)
        # Plotting, ignoring NaN values in the color
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[psd_mask], sat_data['Lstar'][psd_mask,i_K],
                            c=np.log10(psd_plot[psd_mask]), cmap=cmap, vmin=min_val, vmax=max_val)

    for satellite, sat_data in REPT_data.items():
        psd_plot = REPT_data[satellite]['PSD'][k].values[:,i_mu].copy().flatten()
        psd_mask = (psd_plot > 0) & (psd_plot != np.nan)
        lstar_mask = sat_data['Lstar'][:,0]>0
        combined_mask = psd_mask & lstar_mask
        # Plotting, ignoring NaN values in the color
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], sat_data['Lstar'][combined_mask,i_K],
                            c=np.log10(psd_plot[combined_mask]), cmap=cmap, vmin=min_val, vmax=max_val)



    ax.set_title(f"GPS CXD & RBSP REPT, K={k:.1f} $G^{{1/2}}R_E$, $\\mu$={mu:.0f} $MeV/G$", fontsize=textsize + 2)
    ax.set_ylabel(r"L*", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    # Force labels for first and last x-axis tick marks 
    min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    ax.set_ylim(3, 6)
    ax.grid(True)

    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    tick_locations = np.arange(min_val, max_val + 1)
    cbar.set_ticks(tick_locations)
    cbar.set_label(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)

    plt.show()

#%% Plot PSD Radial Profile with REPT data
if plot_radial==True:
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]
    gps_scale = 10

    min_val = np.nanmin(1e-12)
    max_val = np.nanmax(1e-4)

    REPT_data_root = '/home/wzt0020/REPT_data/'
    save_path = os.path.join(REPT_data_root, storm_name, f'rept_data_{extMag}.npz')
    complete_load = np.load(save_path, allow_pickle=True)
    REPT_data = load_data(complete_load)
    complete_load.close()
    del complete_load

    # time_start  = dt.datetime(2017, 4, 23, 19, 30, 0)
    # time_stop   = dt.datetime(2017, 4, 23, 23, 00, 0)
        
    # time_start  = dt.datetime(2017, 4, 24, 17, 7, 0)
    # time_stop   = dt.datetime(2017, 4, 24, 21, 35, 0)
    
    time_start  = dt.datetime(2017, 4, 25, 16, 30, 0)
    time_stop   = dt.datetime(2017, 4, 25, 21, 00, 0)

    # time_start  = dt.datetime(2017, 4, 21, 0, 0, 0)
    # time_stop   = dt.datetime(2017, 4, 26, 0, 0, 0)

    # time_start  = dt.datetime(2018, 8, 25, 2, 30, 0)
    # time_stop   = dt.datetime(2018, 8, 25, 5, 30, 0)

    # time_start  = dt.datetime(2018, 8, 26, 23, 30, 0)
    # time_stop   = dt.datetime(2018, 8, 27, 2, 0, 0)

    # Convert Epoch_A and Epoch_B to NumPy arrays of datetimes
    Epoch_B_np = np.array(REPT_data['rbspb']['Epoch'].UTC)

    # Define Lstar delta
    lstar_delta = 0.1

    # Generate Lstar interval boundaries within the time range.
    time_range = Epoch_B_np[(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
    lstar_range = REPT_data['rbspb']['Lstar'][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop),i_K]
    psd_range = REPT_data['rbspb']['PSD'][k].values[:,i_mu][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]

    fig, ax = plt.subplots(figsize=(12, 8))
    colormap_name = 'viridis_r' # Good choice for continuous data
    cmap = plt.cm.get_cmap(colormap_name)

    import matplotlib.dates as mdates
    time_range_timestamps = mdates.date2num(time_range)

    vmin = mdates.date2num(time_start) #- dt.timedelta(minutes=(time_start.minute % 30))
    vmax = mdates.date2num(time_stop ) #+ dt.timedelta(minutes=30 - (time_stop.minute % 30))
    norm = colors.Normalize(vmin=vmin,
                            vmax=vmax)

    # Apply the mask to both averaged_lstar and averaged_psd
    scatter_plot = ax.scatter(
        lstar_range,
        psd_range,
        c=time_range_timestamps, # Color by Epoch datetime objects
        cmap=cmap,
        norm=norm,
        marker='o')

    cbar = fig.colorbar(scatter_plot, ax=ax, orientation='horizontal', pad=0.15)
    cbar.set_label('Time (UTC)', fontsize=textsize)
    cbar.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    cbar.ax.tick_params(labelsize=textsize-2)

    rounded_dt_obj = time_start + dt.timedelta(minutes=10 - (time_start.minute % 10))
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
    color_set = plt.cm.get_cmap('viridis_r')(np.linspace(0, 1, 256))[np.linspace(0, 255, len(half_hours), dtype=int)]

    sat_valid = np.zeros((len(storm_data),len(half_hours)),dtype=bool)
    avg_Lstar = np.zeros((len(storm_data),len(half_hours)))
    avg_psd = np.zeros((len(storm_data),len(half_hours)))
    for i_hh, half_hour in enumerate(half_hours):
        for satellite, sat_data in storm_data.items():
            time_mask = (sat_data['Epoch'].UTC >= (half_hour-dt.timedelta(minutes=15))) & (sat_data['Epoch'].UTC < (half_hour+dt.timedelta(minutes=15)))
            if sum(time_mask) == 0:
                continue

            nearest_time = np.zeros(sum(time_mask),dtype=int)
            for i_epoch, epoch in enumerate(sat_data['Epoch'].UTC[time_mask]):
                nearest_time[i_epoch] = np.argmin(np.abs(REPT_data['rbspb']['Epoch'].UTC-epoch))

            MLT_mask = ((sat_data['MLT'][time_mask] >= (REPT_data['rbspb']['MLT'][nearest_time]-3)) 
                        & (sat_data['MLT'][time_mask] < (REPT_data['rbspb']['MLT'][nearest_time]+3)))
            valid_mask = MLT_mask
            
            if np.sum(valid_mask)>0:
                i_sat = list(storm_data.keys()).index(satellite)
                psd_masked = psd[satellite][k].values[time_mask,i_mu][valid_mask]
                psd_masked = psd_masked[~np.isnan(psd_masked)]
                if len(psd_masked) > 0:
                    sat_valid[i_sat,i_hh] = True
                    avg_Lstar[i_sat,i_hh] = np.average(sat_data['Lstar'][time_mask,i_K][valid_mask])
                    avg_psd[i_sat,i_hh] = np.exp(np.nanmean(np.log(psd_masked)))*gps_scale
        color_for_point = cmap(norm(mdates.date2num(half_hour)))
        if np.sum(sat_valid[:,i_hh]) != 0:
            ax.plot(avg_Lstar[sat_valid[:,i_hh], i_hh], avg_psd[sat_valid[:,i_hh], i_hh],
                        marker='*', markersize=12,
                        color=color_for_point, # Use the calculated color
                        label=half_hour.strftime("%d-%m-%Y %H:%M")) # Label for each star
    sat_valid = pd.DataFrame(sat_valid,index=list(storm_data.keys()),columns=half_hours)

    ax.tick_params(axis='both', labelsize=textsize, pad=10)

    ax.set_xlim(3, 6)
    ax.set_xlabel(r"L*", fontsize=textsize)
    ax.set_ylim(min_val, max_val)
    ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize)
    plt.yscale('log')
    ax.grid(True)

    # Add K and Mu text to the plot
    ax.text(0.02, 0.98, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$", transform=ax.transAxes, fontsize=textsize-2, verticalalignment='top') #add the text

    import matplotlib.lines as mlines
    handle_rbspb = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=10, label='RBSP-B') # Use a generic color/marker for circles
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None',
                                markersize=12, label='GPS') # Use a generic color/marker for stars
    # Create the first legend (for RBSP-B and GPS)
    legend1 = ax.legend(handles=[handle_rbspb, handle_gps],
                        title = 'Satellite',
                        title_fontsize = textsize-2,
                        loc='upper right',
                        bbox_to_anchor=(1.15, 1.0),
                        handlelength=1,
                        fontsize=textsize-4)
    # Add the first legend to the axes
    ax.add_artist(legend1)
    # Add legend
    ax.legend(
        title=r"Time",
        title_fontsize = textsize-2,
        loc='center right',
        bbox_to_anchor=(1.25, 0.5),
        markerscale=0.7,
        handlelength=1,
        fontsize=textsize-4
    )
    # Set the plot title to the time interval
    title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title_str, fontsize = textsize)
    plt.show()
