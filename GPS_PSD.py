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
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib import colors
import pandas as pd

import importlib
import GPS_PSD_func
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import (QinDenton_period, import_GPS, data_period, data_from_gps, load_data,
                            AlphaOfK, MuofEnergyAlpha, EnergyofMuAlpha, energy_spectra, find_psd, find_Lstar)
import Zhao_2018_PAD_Model
importlib.reload(Zhao_2018_PAD_Model)
from Zhao_2018_PAD_Model import (import_Zhao_coeffs, find_Zhao_PAD_coeffs, create_PAD, PAD_Scale_Factor)

import time

#%% Global Variables
textsize = 16
Re = 6378.137 #Earth's Radius
Mu_set = np.array((2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G
K_set = np.array((0.1,1,2)) # R_E*G^(1/2)
mode = 'load' # 'save' or 'load'
storm_name = 'sep2019storm' # 'april2017storm', 'aug2018storm', 'oct2012storm', 'may2019storm', 'sep2019storm'
plot_flux = False
plot_processes_flux = False
plot_psd = False
plot_combined_psd = True
plot_energies = False
PAD_calculate = False
plot_PAD = True
plot_radial = True

GPS_data_root = '/home/wzt0020/sat_data_analysis/GPS_data/'
input_folder = os.path.join(GPS_data_root, storm_name)
base_save_folder = os.path.join(GPS_data_root, storm_name)
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

elif storm_name == 'may2019storm':
    start_date  = dt.datetime(2019, 5, 10, 00, 00, 0)
    stop_date   = dt.datetime(2019, 5, 17, 00, 00, 0)

elif storm_name == 'sep2019storm':
    start_date  = dt.datetime(2019, 8, 31, 00, 00, 0)
    stop_date   = dt.datetime(2019, 9, 2, 00, 00, 0)

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
    energyofmualpha_filename = f"energyofmualpha_{extMag}.npz"
    energyofmualpha_save_path = os.path.join(base_save_folder, energyofmualpha_filename)
    for satellite, sat_data in storm_data.items():
        print(f"Calculating Energy of Mu and Alpha for satellite {satellite}", end='\r')
        energyofmualpha[satellite] = EnergyofMuAlpha(sat_data, Mu_set, alphaofK[satellite])
    
    if mode == 'save':
        # Save Data for later recall:
        print("Saving REPT Data...")
        np.savez(energyofmualpha_save_path, **energyofmualpha)
        print("Data Saved \n")

### Find Flux at Set Energy ###
    flux_energyofmualpha = energy_spectra(storm_data, energyofmualpha)

### Find Flux at Set Pitch Angle ####
    #--- Extract Zhao Coefficients at each Epoch ---#
    Zhao_epoch_coeffs_filename = f"Zhao_epoch_coeffs_{extMag}.npz"
    Zhao_epoch_coeffs_save_path = os.path.join(base_save_folder, Zhao_epoch_coeffs_filename)
    if mode == 'save':
        Zhao_epoch_coeffs = {}
        for satellite, sat_data in storm_data.items():
            print(f"    Finding PAD coefficients for satellite {satellite}", end='\r')
            Zhao_epoch_coeffs[satellite] = find_Zhao_PAD_coeffs(sat_data, QD_storm_data, energyofmualpha[satellite], alphaofK[satellite], extMag)

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
    
    #---Find PAD Model shapes---#
    PAD_filename = f"PAD_model_{extMag}.npz"
    PAD_save_path = os.path.join(base_save_folder, PAD_filename)
    if PAD_calculate == True:    
        if mode == 'save':
            PAD_models = {}
            for satellite, sat_data in storm_data.items():
                print(f"    Modeling PAD for satellite {satellite}", end='\r')
                PAD_models[satellite] = create_PAD(sat_data, Zhao_epoch_coeffs[satellite], alphaofK[satellite])

            # Save Data for later recall:
            print("\nSaving GPS PAD Model Data...")
            np.savez(PAD_save_path, **PAD_models )
            print("Data Saved \n")

        elif mode == 'load': 
            # Read in data from previous save
            PAD_models_load = np.load(PAD_save_path, allow_pickle=True)
            PAD_models = load_data(PAD_models_load)
            PAD_models_load.close()
            del PAD_models_load

    #--- Find Scale Factor from alphaofK and PAD Model ---#
    scale_factor = {}
    PAD_int = {}
    for satellite, sat_data in storm_data.items():
        print(f"    Calculating Scale Factor for satellite {satellite}", end='\r')
        scale_factor[satellite], PAD_int[satellite] = PAD_Scale_Factor(sat_data, Zhao_epoch_coeffs[satellite], alphaofK[satellite]) # nans either come from alphaofK or dividing by zero (integral)

### Find Flux at Set Pitch Angle and Energy ###
    flux = {}
    for satellite, sat_data in storm_data.items():
        flux[satellite] = {}
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        for i_K, K_value in enumerate(K_set):
            flux[satellite][K_value] = flux_energyofmualpha[satellite][K_value].values * 2*2*np.pi * scale_factor[satellite][K_value].values
            flux[satellite][K_value] = pd.DataFrame(flux[satellite][K_value], index = epoch_str, columns=Mu_set)

### Calculate PSD ###
    psd = {}
    for satellite, sat_data in storm_data.items():
        print(f"Calculating PSD for satellite {satellite}", end='\r')
        psd[satellite] = find_psd(flux[satellite], energyofmualpha[satellite])

### Calculate Lstar ###
    complete_filename = f"storm_data_{extMag}.npz"
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

#%% Plot GPS Flux
if plot_flux==True:
    energy = 2.1 # MeV
    energy_channels = storm_data[list(storm_data.keys())[0]]['Energy_Channels']
    i_energy = np.argmin(np.abs(energy_channels - energy))

    k = 0.1
    i_K = np.where(K_set == k)[0]

    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1e2))
    max_val = np.nanmax(np.log10(1e6))

    if extMag == 'T89c':
        extMag_label = 'T89'
    else:
        extMag_label = extMag

    fig, ax = plt.subplots(figsize=(16, 4))
    for satellite, sat_data in storm_data.items():     
        flux_plot = sat_data['electron_diff_flux'][:,i_energy]
        flux_mask = (flux_plot > 0) & (flux_plot != np.nan)

        scale_muindex = np.argmin(np.abs(energyofmualpha[satellite][k].values - energy), axis=1)
        # Plotting, ignoring NaN values in the color
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[flux_mask], sat_data[f'L_LGM_{extMag_label}IGRF'][flux_mask],
                            c=np.log10(flux_plot[flux_mask]), vmin=min_val, vmax=max_val)

    ax.set_title(f"GPS CXD, {energy} MeV Electron Differential Flux", fontsize=textsize + 2)
    ax.set_ylabel(r"McIlwain L", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    # Force labels for first and last x-axis tick marks 
    min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=24))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    ax.set_ylim(3, 6)
    ax.grid(True)

    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    cbar.set_label(label = r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)

    plt.show()

# %% Plot Processed Flux for REPT and GPS at K and Mu
if plot_processes_flux==True:
    k=0.1
    i_K = np.where(K_set == k)[0]
    mu=2000
    i_mu = np.where(Mu_set == mu)[0]
    time_start = dt.datetime(start_date.year, 8, 31, 0, 0, 0) # for sep2019storm
    time_stop = dt.datetime(stop_date.year, 9, 2, 0, 0, 0) # for sep2019storm

    min_val = np.nanmin(np.log10(1e3))
    max_val = np.nanmax(np.log10(1e7))

    # Import REPT Data
    save_path = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'rept_data_{extMag}.npz')
    complete_load = np.load(save_path, allow_pickle=True)
    REPT_data = load_data(complete_load)
    complete_load.close()
    del complete_load

    # Load in REPT PAD Models
    REPT_PAD_save_path = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'REPT_PAD_model_{extMag}.npz')
    PAD_model_load = np.load(REPT_PAD_save_path, allow_pickle=True)
    REPT_PAD_Models = load_data(PAD_model_load)
    PAD_model_load.close()
    del PAD_model_load

    # Load in REPT alphaofK
    alphaofK_REPT_save = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'alphaofK_{extMag}.npz')
    alphaofK_load = np.load(alphaofK_REPT_save, allow_pickle=True)
    REPT_alphaofK = load_data(alphaofK_load)
    for satellite, sat_data in REPT_data.items():
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        REPT_alphaofK[satellite] = pd.DataFrame(REPT_alphaofK[satellite], index=epoch_str, columns=K_set)
    alphaofK_load.close()
    del alphaofK_load

    # Load in REPT energyofmualpha
    energyofmualpha_REPT_save = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'energyofmualpha_{extMag}.npz')
    energyofmualpha_load = np.load(energyofmualpha_REPT_save, allow_pickle=True)
    REPT_energyofmualpha = load_data(energyofmualpha_load)
    energyofmualpha_load.close()
    del energyofmualpha_load

    # Calculate REPT scale
    REPT_Model_scale = {}
    for satellite, sat_data in REPT_data.items():
        REPT_PA_local = REPT_data[satellite]['Pitch_Angles']

        REPT_Model_scale[satellite] = np.zeros(len(sat_data['Epoch'].UTC))
        for iepoch, epoch in enumerate(sat_data['Epoch'].UTC):
            energy_at_epoch = REPT_energyofmualpha[satellite][k][mu].iloc[iepoch]
            nearest_REPT_ienergy = np.argmin(np.abs(REPT_data[satellite]['Energy_Channels'][0:6]-energy_at_epoch))
            
            REPT_PA_eq = np.rad2deg(np.asin(np.sqrt(np.sin(np.deg2rad(REPT_PA_local))**2*REPT_data[satellite]['b_min'][iepoch]/REPT_data[satellite]['b_satellite'][iepoch])))
            REPT_PA_eq = np.unique(np.concatenate((REPT_PA_eq, 180-REPT_PA_eq)))
            REPT_PA_local90 = REPT_PA_eq[len(REPT_PA_local)-1]
            REPT_PAD = REPT_data[satellite]['FEDU'][iepoch,:,nearest_REPT_ienergy]
            REPT_PAD = np.insert(REPT_PAD, len(REPT_PA_local), REPT_PAD[len(REPT_PA_local)-1])
            
            Model_PA = REPT_PAD_Models[satellite][k][mu]['pitch_angles'].values[iepoch, :]
            Model_closest_PA = np.argmin(np.abs(Model_PA-REPT_PA_local90))
            Model_closest_alpha = np.argmin(np.abs(Model_PA-REPT_alphaofK[satellite][k].iloc[iepoch]))
            Model_PAD = REPT_PAD_Models[satellite][k][mu]['Model'].values[iepoch, :]

            REPT_Model_scale[satellite][iepoch] = REPT_PAD[len(REPT_PA_local)-1]/Model_PAD[Model_closest_PA]*Model_PAD[Model_closest_alpha]

    fig, ax = plt.subplots(figsize=(16, 4))
    for satellite, sat_data in REPT_data.items():
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        scale_mask = (REPT_Model_scale[satellite] > 0) & (REPT_Model_scale[satellite] != np.nan)
        lstar_mask = sat_data['Lstar'][:,0]>3
        combined_mask = sat_iepoch_mask & scale_mask & lstar_mask
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], sat_data['Lstar'][combined_mask,i_K],
                            c=np.log10(REPT_Model_scale[satellite][combined_mask]), marker='o', s=50, zorder=2,
                            cmap='viridis', vmin=min_val, vmax=max_val)

    for satellite, sat_data in storm_data.items():
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        scale_mask = (flux[satellite][k][mu].values > 0) & (flux[satellite][k][mu].values != np.nan)
        lstar_mask = sat_data['Lstar'][:,0]>0
        combined_mask = sat_iepoch_mask & scale_mask & lstar_mask
        scatter_B = ax.scatter(sat_data['Epoch'].UTC[combined_mask], sat_data['Lstar'][combined_mask,i_K],
                            c=np.log10(flux[satellite][k][mu].values[combined_mask]), marker='*', s=80, alpha=0.7, zorder=1,
                            cmap='viridis', vmin=min_val, vmax=max_val)

    ax.set_xlabel('Time (UTC)', fontsize=textsize)
    ax.set_ylabel('L*', fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.4))
    ax.set_xlim(time_start, time_stop)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    ax.set_ylim(3.6, 5.2)

    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    cbar.set_label(label = r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    plt.title(f'Flux Comparison at K={k:.1f} G$^{{1/2}}R_E$, $\mu$={mu} MeV/G', fontsize=textsize)

    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=10, label='RBSP') # Use a generic color/marker for circles
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None',
                                markersize=12, label='GPS') # Use a generic color/marker for stars
    # Create the first legend (for RBSP-B and GPS)
    legend1 = ax.legend(handles=[handle_rbsp, handle_gps],
                    title = 'Satellite',
                    title_fontsize = textsize-2,
                    loc='upper right',
                    bbox_to_anchor=(1.2, 1.1),
                    handlelength=1,
                    fontsize=textsize-4)

    ax.grid(True)
    plt.show()
    
#%% Plot PSD
if plot_psd==True:
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 4000
    i_mu = np.where(Mu_set == mu)[0]

    if extMag == 'T89c':
        extMag_label = 'T89'
    else:
        extMag_label = extMag

    fig, ax = plt.subplots(figsize=(16, 4))

    colorscheme = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))
    cmap = colors.ListedColormap(colorscheme)

    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1e-12))
    max_val = np.nanmax(np.log10(1e-6))

    plot_data = {}
    plot_data['ns53'] = storm_data['ns53']
    plot_data['ns55'] = storm_data['ns55']

    for satellite, sat_data in plot_data.items():
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

#%% Plot Combined PSD with REPT data
if plot_combined_psd==True:
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 4000
    i_mu = np.where(Mu_set == mu)[0]

    time_start  = start_date
    time_stop   = stop_date

    time_start = dt.datetime(start_date.year, 5, 14, 0, 0, 0)
    time_stop = dt.datetime(stop_date.year, 5, 15, 0, 0, 0)

    # Logarithmic colorbar setup
    min_val = np.nanmin(np.log10(1e-12))
    max_val = np.nanmax(np.log10(1e-6))

    save_path = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'rept_data_{extMag}.npz')
    complete_load = np.load(save_path, allow_pickle=True)
    REPT_data = load_data(complete_load)
    complete_load.close()
    del complete_load

    fig, ax = plt.subplots(figsize=(16, 4))

    colorscheme = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.9, 256))
    cmap = colors.ListedColormap(colorscheme)

    for satellite, sat_data in REPT_data.items():
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        psd_plot = REPT_data[satellite]['PSD'][k].values[:,i_mu].copy().flatten()
        psd_mask = (psd_plot > 0) & (psd_plot != np.nan)
        lstar_mask = sat_data['Lstar'][:,0]>0
        combined_mask = psd_mask & lstar_mask & sat_iepoch_mask
        # Plotting, ignoring NaN values in the color
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], sat_data['Lstar'][combined_mask,i_K],
                            c=np.log10(psd_plot[combined_mask]), cmap=cmap, vmin=min_val, vmax=max_val, zorder=2)

    for satellite, sat_data in storm_data.items():
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        psd_plot = psd[satellite][k].values[:,i_mu].copy().flatten()
        psd_mask = (psd_plot > 0) & (psd_plot != np.nan)
        combined_mask = psd_mask & sat_iepoch_mask
        # Plotting, ignoring NaN values in the color
        scatter_B = ax.scatter(sat_data['Epoch'].UTC[combined_mask], sat_data['Lstar'][combined_mask,i_K], 
                            marker='*', s=80, alpha=0.7,
                            c=np.log10(psd_plot[combined_mask]), cmap=cmap, vmin=min_val, vmax=max_val, zorder=1)

    ax.set_title(f"RBSP REPT & GPS CXD, K={k:.1f} $G^{{1/2}}R_E$, $\\mu$={mu:.0f} $MeV/G$", fontsize=textsize + 2)
    ax.set_ylabel(r"L*", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.4))
    # Force labels for first and last x-axis tick marks 
    #min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    #max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(time_start, time_stop)   #ax.set_xlim(min_epoch, max_epoch)
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
    ax.set_ylim(3.6, 5.4)
    ax.grid(True)

    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, format=matplotlib.ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    tick_locations = np.arange(min_val, max_val + 1)
    cbar.set_ticks(tick_locations)
    cbar.set_label(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                            markersize=10, label='RBSP') # Use a generic color/marker for circles
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None',
                            markersize=12, label='GPS') # Use a generic color/marker for stars
    # Create the first legend (for RBSP-B and GPS)
    legend1 = ax.legend(handles=[handle_rbsp, handle_gps],
                    title = 'Satellite',
                    title_fontsize = textsize-2,
                    loc='upper right',
                    bbox_to_anchor=(1.2, 1.1),
                    handlelength=1,
                    fontsize=textsize-4)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)

    plt.show()


#%% Plot Energies corresponding to Mu and K across L*
if plot_energies==True:
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    time_start  = start_date
    time_stop   = stop_date

    time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0)
    time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0)

    colormap_name = 'viridis'
    cmap = plt.cm.get_cmap(colormap_name)
    vmin = mdates.date2num(time_start)
    vmax = mdates.date2num(time_stop)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 9))
    for satellite, sat_data in storm_data.items():
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        energy_plot = energyofmualpha[satellite][k].values[:,i_mu].copy().flatten()
        energy_mask = (energy_plot > 0) & (energy_plot != np.nan)
        combined_mask = energy_mask & sat_iepoch_mask
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


#%% Plot PAD comparison
if plot_PAD==True: 
    time_select = dt.datetime(start_date.year, 8, 31, 8, 30, 0)
    sat_select = 'rbspa'
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]

    # Find REPT PAD
    if sat_select == 'rbspa':
        rbsp_label = 'RBSP-A'
    elif sat_select == 'rbspb':
        rbsp_label = 'RBSP-B'

    if extMag == 'T89c':
        extMag_label = 'T89'
    else:
        extMag_label = extMag

    save_path = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'rept_data_{extMag}.npz')
    complete_load = np.load(save_path, allow_pickle=True)
    REPT_data = load_data(complete_load)
    complete_load.close()
    del complete_load

    energyofmualpha_REPT_save = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'energyofmualpha_{extMag}.npz')
    complete_load = np.load(energyofmualpha_REPT_save, allow_pickle=True)
    REPT_energyofmualpha = load_data(complete_load)
    complete_load.close()
    del complete_load

    alphaofK_REPT_save = os.path.join(f'/home/wzt0020/sat_data_analysis/REPT_data/{storm_name}/', f'alphaofK_{extMag}.npz')
    alphaofK_load = np.load(alphaofK_REPT_save, allow_pickle=True)
    REPT_alphaofK = load_data(alphaofK_load)
    for satellite, sat_data in REPT_data.items():
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        REPT_alphaofK[satellite] = pd.DataFrame(REPT_alphaofK[satellite], index=epoch_str, columns=K_set)
    alphaofK_load.close()
    del alphaofK_load

    nearest_it_REPT = np.argmin(np.abs(REPT_data[sat_select]['Epoch'].UTC-time_select))
    nearest_time_REPT = REPT_data[sat_select]['Epoch'].UTC[nearest_it_REPT]

    DST_it_time = np.argmin(np.abs(QD_storm_data['DateTime']-nearest_time_REPT))
    DST_at_time = QD_storm_data['Dst'][DST_it_time]
    print(f'DST at time {nearest_time_REPT.strftime('%Y-%m-%d %H:%M')} is {DST_at_time} nT')
    if DST_at_time > -20:
        i_dst = 'Dst > -20 nT'
    elif DST_at_time <= -20 and DST_at_time > -50:
        i_dst = '-50 nT < Dst < -20 nT'
    elif DST_at_time <= -50:
        i_dst = 'Dst < -50 nT'

    energy_at_mutime = REPT_energyofmualpha[sat_select][k][mu].iloc[nearest_it_REPT]
    nearest_REPT_ienergy = np.argmin(np.abs(REPT_data[sat_select]['Energy_Channels'][0:6]-energy_at_mutime))
    nearest_REPT_energy = REPT_data[sat_select]['Energy_Channels'][nearest_REPT_ienergy]

    MLT_bins = Zhao_coeffs[nearest_REPT_energy][i_dst]['c2']['MLT_values']
    MLT_ref = REPT_data[sat_select]['MLT'][nearest_it_REPT]
    MLT_ibin = np.argmin(np.abs(MLT_bins-MLT_ref))
    MLT_bin = MLT_bins[MLT_ibin]

    L_bins = Zhao_coeffs[nearest_REPT_energy][i_dst]['c2']['L_values']
    L_ref = REPT_data[sat_select][f'L_LGM_{extMag_label}IGRF'][nearest_it_REPT]
    L_ibin = np.argmin(np.abs(L_bins-L_ref))
    L_bin = L_bins[L_ibin]

    print(f'{sat_select} found at MLT = {MLT_ref:.2f} and L = {L_ref:.2f} with Energy = {nearest_REPT_energy:.2f}')

    REPT_PA_local = REPT_data[sat_select]['Pitch_Angles']
    # NOTE: Sometimes, the measured local Bfield is smaller than the theoretically determined Bmin, causing invalid values in the conversion
    REPT_PA_eq = np.rad2deg(np.asin(np.sqrt(np.sin(np.deg2rad(REPT_PA_local))**2*REPT_data[sat_select]['b_min'][nearest_it_REPT]/REPT_data[sat_select]['b_satellite'][nearest_it_REPT])))
    REPT_PA_eq = np.unique(np.concatenate((REPT_PA_eq, 180-REPT_PA_eq)))
    REPT_PA_local90 = REPT_PA_eq[len(REPT_PA_local)-1]
    REPT_PAD = REPT_data[sat_select]['FEDU'][nearest_it_REPT,:,nearest_REPT_ienergy]
    REPT_PAD = np.insert(REPT_PAD, len(REPT_PA_local), REPT_PAD[len(REPT_PA_local)-1])

    # Extract only time period of interest
    REPT_data_epoch = {}
    for key, key_data in REPT_data[sat_select].items():
        if key == 'Energy_Channels' or key == 'Pitch_Angles':
            REPT_data_epoch[key] = REPT_data[sat_select][key]
        elif key == 'FEDU' or key == 'FEDU_averaged':
            REPT_data_epoch[key] = REPT_data[sat_select][key][nearest_it_REPT,:,:]
        elif key == 'Mu_calc':
            continue
        elif key == 'PSD':
            REPT_data_epoch[key] = {}
            REPT_data_epoch[key][k] = pd.DataFrame(REPT_data[sat_select]['PSD'][k][mu].values[nearest_it_REPT], index=[nearest_time_REPT], columns=[mu])
        elif key == 'Lstar':
            REPT_data_epoch[key] = REPT_data[sat_select][key][nearest_it_REPT,i_K]
        else:
            REPT_data_epoch[key] = REPT_data[sat_select][key][nearest_it_REPT]

    REPT_energyofmualpha_val = REPT_energyofmualpha[sat_select][k][mu].iloc[nearest_it_REPT]
    REPT_energyofmualpha_epoch = {}
    REPT_energyofmualpha_epoch[sat_select] = {}
    REPT_energyofmualpha_epoch[sat_select][k] = pd.DataFrame(REPT_energyofmualpha_val,index=[str(nearest_time_REPT)],columns=[mu])
    Zhao_coeffs_REPT = {}
    Zhao_coeffs_REPT[sat_select] = find_Zhao_PAD_coeffs(REPT_data_epoch, QD_storm_data, REPT_energyofmualpha_epoch[sat_select], REPT_alphaofK[sat_select], extMag)
    
    REPT_alpha_val = REPT_alphaofK[sat_select][k].iloc[nearest_it_REPT]
    REPT_alphaofK_epoch = {}
    REPT_alphaofK_epoch[sat_select] = pd.DataFrame(REPT_alpha_val,index=[str(nearest_time_REPT)],columns=[k])
    PAD_models_REPT = {}
    PAD_models_REPT[sat_select] = create_PAD(REPT_data_epoch, Zhao_coeffs_REPT[sat_select], REPT_alphaofK_epoch[sat_select])

    Model_PA = PAD_models_REPT[sat_select][k][mu]['pitch_angles'].values[0]
    Model_closest_PA = np.argmin(np.abs(Model_PA-REPT_PA_local90))
    Model_PAD = PAD_models_REPT[sat_select][k][mu]['Model'].values[0]
    Model_PAD_scale = REPT_PAD[len(REPT_PA_local)-1]/Model_PAD[Model_closest_PA]

    time_window = dt.timedelta(minutes=30)
    time_lower_bound = time_select - time_window
    time_upper_bound = time_select + time_window

    Model_GPS_PA = {}
    Model_GPS_PAD = {}
    near_time_GPS_index = {}
    near_time_GPS = {}
    for satellite, sat_data in storm_data.items():
        time_range_mask = (sat_data['Epoch'].UTC >= time_lower_bound) & (sat_data['Epoch'].UTC <= time_upper_bound)
        near_time_GPS_index_sat = np.where(time_range_mask)[0]
        Model_GPS_PA[satellite] = []
        Model_GPS_PAD[satellite] = []
        near_time_GPS_index[satellite] = []
        near_time_GPS[satellite] = []
        
        for ii, i_time in enumerate(near_time_GPS_index_sat):
            near_time_GPS_temp = sat_data['Epoch'].UTC[i_time]

            current_E = energyofmualpha[satellite][k][mu].iloc[i_time]
            current_L = sat_data[f'L_LGM_{extMag_label}IGRF'][i_time]
            current_L_ibin = np.argmin(np.abs(L_bins-current_L))
            current_MLT = sat_data['MLT'][i_time]
            current_MLT_ibin = int(((np.atleast_1d(sat_data['MLT'])[i_time] + 1) % 24) // 2)
            
            e_mask = np.argmin(np.abs(REPT_data[sat_select]['Energy_Channels'][0:6]-current_E))==nearest_REPT_ienergy
            l_shell_mask = current_L_ibin == L_ibin
            mlt_mask = current_MLT_ibin == MLT_ibin
            
            if e_mask and l_shell_mask and mlt_mask:
                near_time_GPS_index[satellite].append(i_time)
                near_time_GPS[satellite].append(near_time_GPS_temp)

                pa_array = PAD_models[satellite][k][mu]['pitch_angles'].values[i_time, :]
                Model_GPS_PA[satellite].append(pa_array)
                
                pad_array = PAD_models[satellite][k][mu]['Model'].values[i_time, :]
                Model_GPS_PAD[satellite].append(pad_array)
                print(f'GPS satellite {satellite} found at time {near_time_GPS_temp.strftime('%Y-%m-%d %H:%M')} in MLT = {current_MLT} and L = {current_L} with Energy = {current_E:.2f}')
        
        Model_GPS_PA[satellite] = np.array(Model_GPS_PA[satellite])
        Model_GPS_PA = {key: value for key, value in Model_GPS_PA.items() if value.size > 0}
        Model_GPS_PAD[satellite] = np.array(Model_GPS_PAD[satellite])
        Model_GPS_PAD = {key: value for key, value in Model_GPS_PAD.items() if value.size > 0}
    if not Model_GPS_PAD:
        print(f'No GPS satellites found at this time in MLT = {MLT_bin} and L = {L_bin}')

    # # Local Pitch Angle (REPT) Plot
    # fig, ax = plt.subplots(figsize=(10, 9))

    # REPT_Flux_plot = ax.scatter(np.sort(np.concatenate((REPT_PA_local, 180-REPT_PA_local)))[REPT_PAD>0],REPT_PAD[REPT_PAD>0],label=rbsp_label,zorder=3,color='black',marker='+',s=200)
    # REPT_PA_local_plot = np.rad2deg(np.asin(np.sqrt(np.sin(np.deg2rad(Model_PA[Model_PA<=REPT_PA_local90]))**2*REPT_data[sat_select]['b_satellite'][nearest_it_REPT]/REPT_data[sat_select]['b_min'][nearest_it_REPT])))
    # REPT_PA_local_plot = np.sort(np.concatenate((REPT_PA_local_plot, 180-REPT_PA_local_plot)))
    # REPT_PAD_Model_plot = np.concatenate((Model_PAD[Model_PA<=REPT_PA_local90],Model_PAD[Model_PA>=(180-REPT_PA_local90)]))
    # Model_PAD_scale_local = np.mean((REPT_PAD[len(REPT_PA_local)-1],REPT_PAD[len(REPT_PA_local)]))/REPT_PAD_Model_plot[Model_closest_PA]
    # REPT_PAD_plot = ax.plot(REPT_PA_local_plot,REPT_PAD_Model_plot*Model_PAD_scale_local,label='RBSP Model',zorder=2,linewidth=4,alpha=0.7,linestyle='solid')
    
    # ax.text(0.8, 0.96, r"E = " + f"{nearest_REPT_energy:.1f} " + r"$MeV$", transform=ax.transAxes, fontsize=textsize) #add the text

    # ax.legend(fontsize=textsize-4)

    # ax.set_xlim(0,180)
    # ax.set_ylim(10**np.floor(np.log10(np.nanmin(Model_PAD*Model_PAD_scale))),10**np.ceil(np.log10(np.nanmax(Model_PAD*Model_PAD_scale))))
    # plt.yscale('log')
    # ax.tick_params(axis='both', labelsize=textsize, pad=10)
    # ax.set_xlabel(r"Local Pitch Angle (degrees)", fontsize=textsize)
    # ax.set_ylabel(r'Directional Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
    # ax.grid(True)

    # title_str = f"Time: {time_select.strftime('%Y-%m-%d %H:%M')}"
    # ax.set_title(title_str, fontsize = textsize)


    fig, ax = plt.subplots(figsize=(9, 9))

    #REPT_Flux_plot = ax.scatter(REPT_PA_eq[REPT_PAD>0],REPT_PAD[REPT_PAD>0],label=rbsp_label,zorder=3,color='black',marker='+',s=200)
    #REPT_PAD_plot = ax.plot(Model_PA,Model_PAD*Model_PAD_scale,label='RBSP Model',zorder=2,linewidth=4,alpha=0.7,linestyle='solid')

    GPS_model_scale = {}
    GPS_local90 = {}
    GPS_loss_cone = {}
    GPS_alpha = {}
    model_scale_ratio = {}
    plot_first = False
    for satellite, GPS_PAD in Model_GPS_PAD.items():
        GPS_model_scale[satellite] = []
        GPS_local90[satellite] = []
        GPS_loss_cone[satellite] = []
        GPS_alpha[satellite] = []
        model_scale_ratio[satellite] = []
        if near_time_GPS_index[satellite]:
            for i, i_time in enumerate(near_time_GPS_index[satellite]):
                if plot_first == False:
                    plot_first = True
                    GPS_model_scale_epoch = flux[satellite][k][mu].values[i_time]
                    GPS_model_scale[satellite].append(GPS_model_scale_epoch)
                    model_scale_ratio[satellite].append(Model_PAD_scale/GPS_model_scale_epoch)
                    #GPS_PAD_plot = ax.plot(Model_GPS_PA[satellite][i],GPS_PAD[i]*GPS_model_scale_epoch,label=satellite,zorder=1,alpha=0.7,linewidth='3',linestyle='dotted')
                    
                    GPS_loss_cone_temp = storm_data[satellite]['loss_cone'][i_time]
                    GPS_loss_cone[satellite].append(GPS_loss_cone_temp)
                    local_loss_cone = np.rad2deg(np.arcsin(np.sqrt(np.sin(np.deg2rad(GPS_loss_cone_temp))**2 * (storm_data[satellite]['b_satellite'][i_time] / storm_data[satellite]['b_min'][i_time]))))

                    GPS_local90_temp = storm_data[satellite]['local90PA'][i_time]
                    GPS_local90[satellite].append(GPS_local90_temp)
                

                    low_mask = Model_GPS_PA[satellite][i] <= GPS_local90_temp
                    low_eq_pa = Model_GPS_PA[satellite][i][low_mask]
                    local_pa_rad = np.arcsin(np.sqrt(np.sin(np.deg2rad(low_eq_pa))**2 * (storm_data[satellite]['b_satellite'][i_time] / storm_data[satellite]['b_min'][i_time])))
                    local_pa_deg = np.rad2deg(local_pa_rad)
                    GPS_PA_local_plot = np.concatenate((local_pa_deg, 180 - local_pa_deg))
                    low_flux = GPS_PAD[i][low_mask]
                    GPS_PAD_Model_plot = np.concatenate((low_flux, low_flux))
                    sort_idxs = np.argsort(GPS_PA_local_plot)
                    GPS_PA_local_plot = GPS_PA_local_plot[sort_idxs]
                    GPS_PAD_Model_plot = GPS_PAD_Model_plot[sort_idxs]
                    nan_mask = ~np.isnan(GPS_PA_local_plot)
                    GPS_PA_local_plot = GPS_PA_local_plot[nan_mask]
                    GPS_PAD_Model_plot = GPS_PAD_Model_plot[nan_mask]
                    GPS_PAD_plot = ax.plot(GPS_PA_local_plot,GPS_PAD_Model_plot*GPS_model_scale_epoch,
                                label=satellite,zorder=1,alpha=0.7,linewidth=3,linestyle='dotted')
                    GPS_PAD_plot = ax.fill_between(GPS_PA_local_plot,1e-1,GPS_PAD_Model_plot*GPS_model_scale_epoch,
                                                   where = (GPS_PA_local_plot > local_loss_cone) & (GPS_PA_local_plot < 180-local_loss_cone),
                                                   label=satellite,zorder=1,alpha=0.7,linewidth=3,linestyle='dotted')

                    GPS_alpha_temp = alphaofK[satellite][k].iloc[i_time]
                    GPS_alpha[satellite].append(GPS_alpha_temp)
                
                    #current_color = GPS_PAD_plot[0].get_color()
                    GPS_loss_cone_plot = ax.vlines(x=local_loss_cone,ymin=0,ymax=1e8,color=current_color,linestyle='-.')
                    GPS_loss_cone_plot = ax.vlines(x=180-local_loss_cone,ymin=0,ymax=1e8,color=current_color,linestyle='-.')
                    #GPS_local90_plot = ax.vlines(x=GPS_local90_temp,ymin=0,ymax=1e8,color=current_color,linestyle='-')
                    #GPS_alpha_plot = ax.vlines(x=GPS_alpha_temp,ymin=0,ymax=1e8,color=current_color,linestyle='--')


    # Add K and Mu text to the plot
    ax.text(0.48, 0.96, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$", transform=ax.transAxes, fontsize=textsize) #add the text

    LEGEND_GRAY = [0.6, 0.6, 0.6]
    handle_loss_cone = mlines.Line2D([], [], color=LEGEND_GRAY, linestyle='-', 
                                linewidth=2, label='GPS Loss Cone')
    # handle_local90 = mlines.Line2D([], [], color=LEGEND_GRAY, linestyle='-', 
    #                             linewidth=2, label='GPS Local 90')
    # handle_alpha_k = mlines.Line2D([], [], color=LEGEND_GRAY, linestyle='--', 
    #                             linewidth=2, label=r'PA at K=' + f'{k:.1f}')

    existing_handles, existing_labels = ax.get_legend_handles_labels()
    new_handles = existing_handles + [handle_loss_cone]#, handle_local90, handle_alpha_k]
    new_labels = existing_labels + [handle_loss_cone.get_label()]#, handle_local90.get_label(), handle_alpha_k.get_label()]

    ax.legend(new_handles, new_labels, fontsize=textsize-4, loc='lower center')

    ax.set_xlim(0,180)
    #ax.set_ylim(10**np.floor(np.log10(np.nanmin(Model_PAD*Model_PAD_scale))),10**np.ceil(np.log10(np.nanmax(Model_PAD*Model_PAD_scale))))
    ax.set_ylim(1e2,1e4)
    plt.yscale('log')
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlabel(r"Local Pitch Angle (degrees)", fontsize=textsize)
    ax.set_ylabel(r'Directional Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
    ax.grid(True)

    title_str = f"Time: {time_select.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title_str, fontsize = textsize)
    plt.show()   




#%% Plot PSD Radial Profile with REPT data
if plot_radial==True:
    sat_select = 'rbspa'
    k = 0.1
    i_K = np.where(K_set == k)[0]
    mu = 2000
    i_mu = np.where(Mu_set == mu)[0]
    gps_scale = 1 # default = 1
    MLT_range = 3 # hours, default = 3 which corresponds to +-1.5 hours
    lstar_delta = 0.1 # default = 0.1
    time_delta = 30 # minutes, default = 30

    min_val = np.nanmin(1e-12)
    max_val = np.nanmax(1e-7)

    REPT_data_root = '/home/wzt0020/sat_data_analysis/REPT_data/'
    save_path = os.path.join(REPT_data_root, storm_name, f'rept_data_{extMag}.npz')
    complete_load = np.load(save_path, allow_pickle=True)
    REPT_data = load_data(complete_load)
    complete_load.close()
    del complete_load

    time_start  = start_date
    time_stop   = stop_date

    time_start = dt.datetime(start_date.year, 5, 14, 0, 0, 0)
    time_stop = dt.datetime(stop_date.year, 5, 15, 0, 0, 0)

    # time_start = dt.datetime(start_date.year, 8, 31, 8, 0, 0) # for sep2019storm
    # time_stop = dt.datetime(stop_date.year, 8, 31, 20, 0, 0) # for sep2019storm

    # time_start = dt.datetime(start_date.year, 8, 26, 0, 0, 0) # for aug2018storm
    # time_stop = dt.datetime(stop_date.year, 8, 27, 0, 0, 0) # for aug2018storm

    gps_time_start  = time_start
    gps_time_stop   = time_stop

    gps_time_start = dt.datetime(start_date.year, 5, 14, 0, 0, 0)
    gps_time_stop = dt.datetime(stop_date.year, 5, 15, 0, 0, 0)

    # gps_time_start = dt.datetime(start_date.year, 8, 31, 14, 0, 0) # for sep2019storm
    # gps_time_stop = dt.datetime(stop_date.year, 8, 31, 14, 0, 0) # for sep2019storm

    # gps_time_start = dt.datetime(start_date.year, 8, 26, 6, 40, 0) # for aug2018storm
    # gps_time_stop = dt.datetime(stop_date.year, 8, 26, 13, 0, 0) # for aug2018storm

    temp_data = []
    for satellite, sat_data in storm_data.items():
        # 1. Apply Time Mask
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= time_start) & (sat_data['Epoch'].UTC <= time_stop)
        
        # 2. Extract Data for the current satellite
        sat_epoch = sat_data['Epoch'].UTC[sat_iepoch_mask]
        sat_MLT = sat_data['MLT'][sat_iepoch_mask]
        sat_Lstar = sat_data['Lstar'][sat_iepoch_mask, i_K].flatten() # Flatten Lstar to a 1D array
        
        # ASSUMPTION: 'psd' is available and correctly indexed by satellite/k/i_mu
        # We apply the mask to the psd data, and flatten it for consistency
        sat_PSD = psd[satellite][k].values[sat_iepoch_mask, i_mu].flatten()
        
        # 3. Create a satellite name array for this block
        sat_name_array = np.full(len(sat_epoch), satellite, dtype='<U10') # Use a 10-character string dtype

        # 4. Combine arrays column-wise (Epoch, Name, Lstar, MLT, PSD)
        # Filter out any data where Lstar or PSD is NaN (optional, but good practice for clean data)
        valid_mask = ~np.isnan(sat_Lstar) & ~np.isnan(sat_PSD)

        # Use a structured array to hold mixed data types
        combined_satellite_data = np.vstack((
            sat_epoch[valid_mask],
            sat_name_array[valid_mask],
            sat_Lstar[valid_mask],
            sat_MLT[valid_mask],
            sat_PSD[valid_mask]
        )).T # Transpose to make it an N x 5 matrix (N rows, 5 columns)

        temp_data.append(combined_satellite_data)

    GPS_plot_data = np.concatenate(temp_data, axis=0)
    GPS_plot_data = GPS_plot_data[GPS_plot_data[:, 0].argsort()]

    nearest_time = np.zeros(len(GPS_plot_data),dtype=int)
    MLT_mask = np.zeros(len(GPS_plot_data),dtype=bool)
    for i_epoch, epoch in enumerate(GPS_plot_data[:,0]):
        nearest_time[i_epoch] = np.argmin(np.abs(REPT_data[sat_select]['Epoch'].UTC-GPS_plot_data[i_epoch,0]))

        MLT_ref = REPT_data[sat_select]['MLT'][nearest_time[i_epoch]]
        MLT_gps = GPS_plot_data[:,3][i_epoch]
        mlt_diff = np.minimum(np.abs(MLT_ref - MLT_gps), 24-np.abs(MLT_ref - MLT_gps))
        MLT_mask[i_epoch] = (mlt_diff <= MLT_range/2)

    # Convert Epoch to numerical time for plotting (RBSP)
    Epoch_np = np.array(REPT_data[sat_select]['Epoch'].UTC)
    time_mask_REPT = (Epoch_np >= time_start) & (Epoch_np <= time_stop)
    time_range_REPT = Epoch_np[time_mask_REPT]
    time_intervals_REPT = np.arange(time_start, time_stop+dt.timedelta(minutes=time_delta), dt.timedelta(minutes=time_delta)).astype(dt.datetime)
    
    time_range_num = mdates.date2num(time_range_REPT)
    sort_indices = np.argsort(time_range_num)
    time_range_REPT_sorted = time_range_REPT[sort_indices]
    time_range_num_sorted = time_range_num[sort_indices]

    lstar_range = REPT_data[sat_select]['Lstar'][time_mask_REPT, i_K].flatten() # Flatten Lstar for scatter plot
    lstar_min = np.min(lstar_range[~np.isnan(lstar_range) & ~(lstar_range==-1.0e31)])
    lstar_max = np.max(lstar_range[~np.isnan(lstar_range)])
    lstar_intervals = np.arange(np.floor(lstar_min / lstar_delta) * lstar_delta, np.ceil(lstar_max / lstar_delta) * lstar_delta + lstar_delta, lstar_delta)
    psd_range = REPT_data[sat_select]['PSD'][k].values[:, i_mu].flatten()[time_mask_REPT] # Flatten PSD

    lstar_range_sorted = lstar_range[sort_indices]
    psd_range_sorted = psd_range[sort_indices]

    time_intervals_GPS = np.arange(gps_time_start, gps_time_stop+dt.timedelta(minutes=time_delta), dt.timedelta(minutes=time_delta)).astype(dt.datetime)
    avg_psd = np.zeros((len(time_intervals_GPS), len(lstar_intervals))) * np.nan
    plotted_GPS_data = np.empty((len(time_intervals_GPS), len(lstar_intervals)), dtype=object)
    for i_time, time_int in enumerate(time_intervals_GPS):
        time_mask_GPS = (GPS_plot_data[:,0] >= (time_int - dt.timedelta(minutes=time_delta/2))) & (GPS_plot_data[:,0] < (time_int + dt.timedelta(minutes=time_delta/2)))
        for i_lstar, lstar_val in enumerate(lstar_intervals):
            lstar_mask = (GPS_plot_data[:,2] >= (lstar_val - lstar_delta/2)) & (GPS_plot_data[:,2] < (lstar_val + lstar_delta/2))
            combined_mask = time_mask_GPS & lstar_mask & MLT_mask
            if np.sum(combined_mask) > 0:
                plotted_GPS_data[i_time, i_lstar] = GPS_plot_data[combined_mask]
                psd_data = GPS_plot_data[:,4][combined_mask].astype(float)
                # Filter out NaNs and then append to the collection list
                psd_data = psd_data[~np.isnan(psd_data)]
                if len(psd_data) > 0:
                    avg_psd[i_time,i_lstar] = np.nanmean(psd_data)*gps_scale
    plotted_GPS_data = pd.DataFrame(plotted_GPS_data, index=time_intervals_GPS, columns=lstar_intervals)

    fig, ax = plt.subplots(figsize=(14, 9))
    colormap_name = 'viridis'
    cmap = plt.cm.get_cmap(colormap_name)
    vmin = mdates.date2num(time_start)
    vmax = mdates.date2num(time_stop)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

     # Apply the mask to both averaged_lstar and averaged_psd
    scatter_plot = ax.scatter(
        lstar_range_sorted,
        psd_range_sorted,
        c=time_range_num_sorted, # Color by Epoch datetime objects
        cmap=cmap,
        norm=norm,
        marker='o')
    
    # scatter_plot = ax.scatter(
    #     GPS_plot_data[MLT_mask,2],
    #     GPS_plot_data[MLT_mask,4],
    #     c=mdates.date2num(GPS_plot_data[MLT_mask,0]), # Color by Epoch datetime objects
    #     cmap=cmap,
    #     norm=norm,
    #     marker='*',
    #     s=80)

    cbar = fig.colorbar(scatter_plot, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Time (UTC)', fontsize=textsize)
    cbar.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    cbar.ax.tick_params(labelsize=textsize-2)

    for i_time, time_int in enumerate(time_intervals_GPS):
        if (sum(~np.isnan(avg_psd[i_time,:]))> 0):
            ax.plot(lstar_intervals, avg_psd[i_time,:],
                    marker='*', markersize=12,
                    color=cmap(norm(mdates.date2num(time_int))), # Use the calculated color
                    label=time_int.strftime("%d-%m-%Y %H:%M")) # Label for each star


    ax.tick_params(axis='both', labelsize=textsize, pad=10)

    ax.set_xlim(3.4, 5.0) # ax.set_xlim(3.8, 5.2)
    ax.set_xlabel(r"L*", fontsize=textsize)
    ax.set_ylim(min_val, max_val)
    ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize)
    plt.yscale('log')
    ax.grid(True)

    # Add K and Mu text to the plot
    ax.text(0.02, 0.98, r"K = " + f"{k:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{mu:.0f}" + r" $MeV/G$", transform=ax.transAxes, fontsize=textsize-2, verticalalignment='top') #add the text

    import matplotlib.lines as mlines
    if sat_select == 'rbspa':
        rbsp_label = 'RBSP-A'
    elif sat_select == 'rbspb':
        rbsp_label = 'RBSP-B'
    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=10, label=rbsp_label) # Use a generic color/marker for circles
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None',
                                markersize=12, label='GPS') # Use a generic color/marker for stars
    # Create the first legend (for RBSP-B and GPS)
    legend1 = ax.legend(handles=[handle_rbsp, handle_gps],
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
        title=r"Time (UTC)",
        title_fontsize = textsize-2,
        loc='center right',
        bbox_to_anchor=(1.225,0.4),
        markerscale=0.7,
        handlelength=1,
        fontsize=textsize-4
    )
    # Set the plot title to the time interval
    title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title_str, fontsize = textsize)
    plt.show()

