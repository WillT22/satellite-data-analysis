#%% Importing relevant libraries
import os
import glob
import sys
import datetime as dt
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_script_dir)
import numpy as np
import scipy.constants as sc
import matplotlib.dates as mdates
import matplotlib.ticker
import matplotlib.colors as colors
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pandas as pd

import importlib
import GPS_PSD_func
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import (QinDenton_period, load_data, data_period, AlphaOfK, find_Loss_Cone, find_local90PA, EnergyofMuAlpha, find_psd, find_McIlwain_L, find_Lstar)
import REPT_PSD_func
importlib.reload(REPT_PSD_func)
from REPT_PSD_func import (process_l3_data, time_average, find_mag, Average_FluxbyPA, Interp_Flux)

import time

#%% Global Variables
textsize = 16
Re = 6378.137 #Earth's Radius
Mu_set = np.array((2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G
K_set = np.array((0.1,1,2)) # R_E*G^(1/2)
mode = 'load' # 'save' or 'load'
storm_name = 'latefeb2019storm' # 'april2017storm', 'aug2018storm', 'oct2012storm', 'latefeb2019storm', 'may2019storm', 'sep2019storm'
plot_flux = True
plot_flux_all = True
plot_energies = False
plot_psd = True
plot_radial = False
plot_radial_Lstar = False
PAD_calculate = False

REPT_data_root = '/home/wzt0020/sat_data_analysis/REPT_data/'
input_folder = os.path.join(REPT_data_root, storm_name)
base_save_folder = os.path.join(REPT_data_root, storm_name)
extMag = 'TS04' # External Magnetic Field Model: 'T89c', 'TS04', NOT 'TS07'

if storm_name == 'april2017storm':
    start_date  = dt.datetime(2017, 4, 21, 00, 00, 0)
    stop_date   = dt.datetime(2017, 4, 26, 00, 00, 0)

elif storm_name == 'aug2018storm':
    start_date = dt.datetime(2018, 8, 25, 0, 0, 0)
    stop_date = dt.datetime(2018, 8, 28, 0, 0, 0)

elif storm_name == 'oct2012storm':
    start_date  = dt.datetime(2012, 10, 7, 00, 00, 0)
    stop_date   = dt.datetime(2012, 10, 11, 00, 00, 0)

elif storm_name == 'latefeb2019storm':
    start_date  = dt.datetime(2019, 2, 27, 00, 00, 0)
    stop_date   = dt.datetime(2019, 3, 4, 00, 00, 0)

elif storm_name == 'may2019storm':
    start_date  = dt.datetime(2019, 5, 10, 00, 00, 0)
    stop_date   = dt.datetime(2019, 5, 17, 00, 00, 0)

elif storm_name == 'sep2019storm':
    start_date  = dt.datetime(2019, 8, 31, 00, 00, 0)
    stop_date   = dt.datetime(2019, 9, 5, 00, 00, 0)

# Conversions
# electron mass in MeV is (m_e [kg] * c^2 [m^2/s^2]) [J] / (sc.eV [J/eV] * 10^6 [eV/MeV])
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6) # this is m_0*c^2
# b_satellite and b_equator are in Gauss: 1 G = 10^5 nT

start_time = time.perf_counter()

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
        file_paths_l3_A = glob.glob(input_folder + "/rbspa*[!r]*.cdf") 
        file_paths_l3_B = glob.glob(input_folder + "/rbspb*[!r]*.cdf")
        
        REPT_data_raw = {}
        if len(file_paths_l3_A) != 0:
            REPT_data_raw['rbspa'] = process_l3_data(file_paths_l3_A)
        if len(file_paths_l3_B) != 0:
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
    #del REPT_data_raw

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

        ### Find Loss Cone and Equatorial B ###
        for satellite, sat_data in REPT_data.items():
            print(f"Calculating Equatorial B-field for satellite {satellite}...")
            REPT_data[satellite]['b_min'], REPT_data[satellite]['b_footpoint'], REPT_data[satellite]['loss_cone'] = find_Loss_Cone(sat_data, extMag=extMag)
    
        ### Determine local 90 degree pitch angle ###
        for satellite, sat_data in REPT_data.items():
            print(f"Finding Local 90 degree Pitch angle for {satellite}...")
            sat_data['local90PA'] = find_local90PA(sat_data)

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
    save_path = os.path.join(base_save_folder, f'rept_data_{extMag}.npz')
    if mode == 'load':
        complete_load = np.load(save_path, allow_pickle=True)
        REPT_data = load_data(complete_load)
        complete_load.close()
        del complete_load
 
### Find Energy for set Mu and Alpha ###
    energyofmualpha = {}
    energyofmualpha_filename = f"energyofmualpha_{extMag}.npz"
    energyofmualpha_save_path = os.path.join(base_save_folder, energyofmualpha_filename)
    for satellite, sat_data in REPT_data.items():
        print(f"Calculating Energy of Mu and Alpha for satellite {satellite}")
        energyofmualpha[satellite] = EnergyofMuAlpha(sat_data, Mu_set, alphaofK[satellite])

    if mode == 'save':
        # Save Data for later recall:
        print("Saving REPT Data...")
        np.savez(energyofmualpha_save_path, **energyofmualpha)
        print("Data Saved \n")

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

### Calculate L ####
    if mode == 'save':
        for satellite, sat_data in REPT_data.items():
            print(f"Calculating L for satellite {satellite}...")
            REPT_data[satellite] = find_McIlwain_L(sat_data, alphaofK[satellite], extMag=extMag)

        # Save Data for later recall:
        print("Saving REPT Data...")
        np.savez(save_path, **REPT_data)
        print("Data Saved \n")
        
### Calculate L* ####
    if mode == 'save':
        for satellite, sat_data in REPT_data.items():
            print(f"Calculating L* for satellite {satellite}...")
            REPT_data[satellite] = find_Lstar(sat_data, alphaofK[satellite], extMag=extMag)

        # Save Data for later recall:
        print("Saving REPT Data...")
        np.savez(save_path, **REPT_data)
        print("Data Saved \n")

### Calculate PAD model (if necessary) ###
    if PAD_calculate == True:
        PAD_filename = f"REPT_PAD_model_{extMag}.npz"
        PAD_scale_filename = f"REPT_PAD_scale_{extMag}.npz"
        PAD_save_path = os.path.join(base_save_folder, PAD_filename)
        PAD_scale_save_path = os.path.join(base_save_folder, PAD_scale_filename)
        if mode == 'save':
            from Zhao_2018_PAD_Model import (import_Zhao_coeffs, find_Zhao_PAD_coeffs, create_PAD, PAD_Scale_Factor)
            
            Zhao_coeffs = import_Zhao_coeffs()

            Zhao_coeffs_REPT = {}
            REPT_PAD_Model = {}
            REPT_PAD_scale = {}
            REPT_PAD_Int = {}
            for satellite, sat_data in REPT_data.items():
                print(f"Calculating PAD model for satellite {satellite}...")
                Zhao_coeffs_REPT[satellite] = find_Zhao_PAD_coeffs(sat_data, QD_storm_data, energyofmualpha[satellite], extMag)
                REPT_PAD_Model[satellite] = create_PAD(sat_data, Zhao_coeffs_REPT[satellite], alphaofK[satellite])
                REPT_PAD_scale[satellite], REPT_PAD_Int[satellite] = PAD_Scale_Factor(sat_data, Zhao_coeffs_REPT[satellite], alphaofK[satellite])
            print(f"PAD model calculation completed.")

            # Save Data for later recall:
            print("\nSaving REPT PAD Model Data...")
            np.savez(PAD_save_path, **REPT_PAD_Model )
            print("Data Saved \n")

            print("\nSaving REPT PAD Scale Data...")
            np.savez(PAD_scale_save_path, **REPT_PAD_scale )
            print("Data Saved \n")

        if mode == 'load':
            # Load data from previous save
            PAD_model_load = np.load(PAD_save_path, allow_pickle=True)
            REPT_PAD_Model = load_data(PAD_model_load)
            PAD_model_load.close()
            del PAD_model_load
            PAD_scale_load = np.load(PAD_scale_save_path, allow_pickle=True)
            REPT_PAD_scale = load_data(PAD_scale_load)
            PAD_scale_load.close()
            del PAD_scale_load


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

#%% Plot Flux for one energy channel
if plot_flux==True:
    energy = 2.1 # MeV
    energy_channels = REPT_data[list(REPT_data.keys())[0]]['Energy_Channels']
    i_energy = np.argmin(np.abs(energy_channels - energy))
    energy = energy_channels[i_energy] # use exact energy from REPT channels

    time_start  = start_date
    time_stop   = stop_date

    time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0)
    time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0)

    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1e3))
    max_val = np.nanmax(np.log10(1e4))

    if extMag == 'T89c':
        plot_extMag = 'T89'
    else:
        plot_extMag = extMag

    fig, ax = plt.subplots(figsize=(16, 4))
    for satellite, sat_data in REPT_data.items():
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        flux_slice = sat_data['FEDU_averaged'][:,:,i_energy]
        flux_temp_mask = np.where(flux_slice >= 0, flux_slice, np.nan)
        flux_plot = np.nanmean(flux_temp_mask, axis=1)/2
        flux_mask = (flux_plot > 0) & (flux_plot != np.nan)
        combined_mask = flux_mask & sat_iepoch_mask
        # Plotting, ignoring NaN values in the color
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], sat_data[f'L_LGM_{plot_extMag}IGRF'][combined_mask],
                            c=np.log10(flux_plot[combined_mask]), vmin=min_val, vmax=max_val)

    ax.set_title(f"RBSP A&B REPT, {energy} MeV Electron Differential Flux", fontsize=textsize + 2)
    ax.set_ylabel(r"McIlwain L", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    # Force labels for first and last x-axis tick marks 
    min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(time_start, time_stop)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    ax.set_ylim(3, 6.5)
    ax.grid(True)

    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    cbar.set_label(label = r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)

    plt.show()

#%% Plot flux for all energy channels
if plot_flux_all==True:
    energy_channels = REPT_data[list(REPT_data.keys())[0]]['Energy_Channels']
    energy_channels = energy_channels[energy_channels<4]

    time_start  = start_date
    time_stop   = stop_date

    # time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0)
    # time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0)

    if extMag == 'T89c':
        plot_extMag = 'T89'
    else:
        plot_extMag = extMag

    fig, axes = plt.subplots(len(energy_channels),1,figsize=(20, 10),sharex=True,sharey=False)
    for i_energy, energy in enumerate(energy_channels):
        ax = axes[i_energy]
        for satellite, sat_data in REPT_data.items():
            sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
            flux_slice = sat_data['FEDU_averaged'][:,:,i_energy]
            flux_temp_mask = np.where(flux_slice >= 0, flux_slice, np.nan)
            flux_plot = np.nanmean(flux_temp_mask, axis=1)/2
            flux_mask = (flux_plot > 0) & (flux_plot != np.nan)
            combined_mask = flux_mask & sat_iepoch_mask
            vmax = np.ceil(max(np.log10(flux_plot[combined_mask])))
            # Plotting, ignoring NaN values in the color
            scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], sat_data[f'L_LGM_{plot_extMag}IGRF'][combined_mask],
                                c=np.log10(flux_plot[combined_mask]), vmin=0, vmax=vmax)

        ax.set_title(f"{energy:.2f} MeV", fontsize=textsize+2)
        # Force labels for first and last x-axis tick marks 
        min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
        max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
        ax.set_xlim(min_epoch, max_epoch)
        ax.tick_params(axis='both', labelsize=textsize, pad=5)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.set_ylim(3, 6.5)
        ax.grid(True)

        cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
        if vmax > 5:
            cbar.locator = matplotlib.ticker.MultipleLocator(2)
        else:
            cbar.locator = matplotlib.ticker.MultipleLocator(1)
        cbar.ax.tick_params(labelsize=textsize)

    # Force labels for first and last x-axis tick marks 
    # Set X-axis limits and labels only for the bottom row
    if i_energy >= len(energy_channels)-1:
        ax.set_xlabel('Time (UTC)', fontsize=textsize,labelpad=2)
        ax.set_xlim(time_start, time_stop)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        ax.tick_params(axis='x', labelsize=textsize, pad=10)

    fig.text(0.96, 0.5, r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', 
         fontsize=textsize, rotation='vertical', va='center')

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(right=0.95, hspace=0.2)
    fig.suptitle(f'RBSP REPT Differential Flux {time_start.strftime('%Y-%m-%d %H')} to {time_stop.strftime('%Y-%m-%d %H')}', fontsize=textsize + 4, y=0.94)
    plt.show()

#%% Plot energies corresponding to each time as probes pass through Lstars
if plot_energies==True:
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    time_start  = start_date
    time_stop   = stop_date

    # time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0)
    # time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0)

    colormap_name = 'viridis'
    cmap = plt.cm.get_cmap(colormap_name)
    vmin = mdates.date2num(time_start)
    vmax = mdates.date2num(time_stop)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 9))
    for satellite, sat_data in REPT_data.items():
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        energy_plot = energyofmualpha[satellite][k].values[:,i_mu].copy().flatten()
        energy_mask = (energy_plot > 0) & (energy_plot != np.nan)
        lstar_mask = (sat_data['Lstar'][:,i_K] > 0).flatten()
        combined_mask = sat_iepoch_mask & energy_mask & lstar_mask
        # Plotting, ignoring NaN values in the color
        scatter_plot = ax.scatter(sat_data['Lstar'][combined_mask,i_K], energy_plot[combined_mask], 
                               c=mdates.date2num(sat_data['Epoch'].UTC[combined_mask]), cmap=cmap, vmin=vmin, vmax=vmax)
    
    cbar = fig.colorbar(scatter_plot, ax=ax, orientation='horizontal', pad=0.11)
    cbar.set_label('Time (UTC)', fontsize=textsize)
    cbar.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    cbar.ax.tick_params(labelsize=textsize-2)

    # Add K and Mu text to the plot
    ax.text(0.5, 0.92, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$", transform=ax.transAxes, fontsize=textsize) #add the text
    
    ax.set_xlim(3.8,5.2)
    ax.set_ylim(1,3.5)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlabel(r"L*", fontsize=textsize)
    ax.set_ylabel(r"Energy (MeV)", fontsize=textsize)
    ax.grid(True)
    

#%% Plot PSD
if plot_psd==True:
    from matplotlib import colors
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    fig, ax = plt.subplots(figsize=(16, 4))

    colorscheme = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))
    cmap = colors.ListedColormap(colorscheme)

    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1e-12))
    max_val = np.nanmax(np.log10(1e-5))

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
    min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
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

#%% Create PSD Radial Profiles without Lstar Averaging
if plot_radial==True:
    sat_select = 'rbspb'
    sat_data = REPT_data[sat_select]
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    min_val = np.nanmin(1e-11)
    max_val = np.nanmax(1e-5)

    time_start  = start_date
    time_stop   = stop_date

    time_start = dt.datetime(start_date.year, 5, 11, 0, 0, 0)
    time_stop = dt.datetime(stop_date.year, 5, 12, 6, 0, 0)

    # Convert Epoch_A and Epoch_B to NumPy arrays of datetimes
    Epoch_B_np = np.array(REPT_data[sat_select]['Epoch'].UTC)

    # Generate Lstar interval boundaries within the time range.
    time_mask = (Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)
    time_range = Epoch_B_np[time_mask]

    fig, ax = plt.subplots(figsize=(14, 9))
    colormap_name = 'viridis'
    cmap = plt.cm.get_cmap(colormap_name)

    import matplotlib.dates as mdates
    time_range_timestamps = mdates.date2num(time_range)

    vmin = mdates.date2num(time_start) #- dt.timedelta(minutes=(time_start.minute % 30))
    vmax = mdates.date2num(time_stop ) #+ dt.timedelta(minutes=30 - (time_stop.minute % 30))
    norm = colors.Normalize(vmin=vmin,
                            vmax=vmax)

    # Apply the mask to both averaged_lstar and averaged_psd
    scatter_plot = ax.scatter(
        sat_data['Lstar'][time_mask,i_K],
        sat_data['PSD'][k].values[:,i_mu][time_mask],
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

    ax.tick_params(axis='both', labelsize=textsize, pad=10)

    ax.set_xlim(3.4, 5)
    ax.set_xlabel(r"L*", fontsize=textsize)
    ax.set_ylim(min_val, max_val)
    ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize)
    plt.yscale('log')
    ax.grid(True)

    # Add K and Mu text to the plot
    ax.text(0.02, 0.98, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$", transform=ax.transAxes, fontsize=textsize-2, verticalalignment='top') #add the text

    if sat_select == 'rbspa':
        rbsp_label = 'RBSP-A'
    elif sat_select == 'rbspb':
        rbsp_label = 'RBSP-B'
    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=10, label=rbsp_label) # Use a generic color/marker for circles
    # Create the first legend (for RBSP-B and GPS)
    ax.legend(handles=[handle_rbsp],
                title = 'Satellite',
                title_fontsize = textsize-2,
                loc='upper right',
                bbox_to_anchor=(1.15, 1.0),
                handlelength=1,
                fontsize=textsize-4)

    # Set the plot title to the time interval
    title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title_str, fontsize = textsize)
    plt.show()

#%% Create PSD Radial Profiles with Lstar Averaging
if plot_radial_Lstar==True:
    sat_select = 'rbspa'
    sat_data = REPT_data[sat_select]
    k = 0.1

    time_start  = start_date
    time_stop   = stop_date

    time_start = dt.datetime(start_date.year, 5, 11, 0, 0, 0)
    time_stop = dt.datetime(stop_date.year, 5, 12, 0, 0, 0)

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
    ax.text(0.02, 0.98, r"K = " + f"{k:.1f} $G^{{1/2}}R_E$", transform=ax.transAxes, fontsize=textsize-4, verticalalignment='top') #add the text

    # Set the plot title to the time interval
    title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title_str)

    plt.tight_layout()
    plt.show()
