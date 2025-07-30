import numpy as np
import matplotlib.pyplot as plt
# Update geomagnetic index and leapsecond data
import spacepy.toolbox
#spacepy.toolbox.update(all=True)

#%% Single file example
#data = dm.readJSONheadedASCII("/home/will/GPS_data/april2017storm/ns60/ns60_170416_v1.10.ascii")
# This is how to see the full tree
#data.tree(verbose=True, attrs=True)
# This is how to see the attributes of one element of the data tree specifically
#data['L_shell'].attrs

#%% Importing all data files
import os
import sys
import glob
import spacepy.datamodel as dm
input_folder = "/home/will/GPS_data/april2017storm/"

#sat_raw = dm.readJSONheadedASCII("/home/will/GPS_data/april2017storm/ns59/ns59_170416_v1.10.ascii")
#sat_raw.tree(verbose=True, attrs=True)
#sat_raw['electron_diff_flux'].attrs

def process_GPS_data(input_folder):
    """
    Processes GPS data files from a specified input folder.
    It expects data organized in satellite-specific subdirectories (e.g., ns60, ns63).
    For each satellite, it finds all .ascii files, sorts them by date,
    and then attempts to read the *list* of sorted files into a SpaceData object.

    Args:
        input_folder (str): The absolute path to the main directory containing
                            satellite data subfolders.

    Returns:
        dict: A dictionary where keys are satellite names (from folder names)
              and values are SpaceData objects loaded from the *list* of files.
              Returns an empty dictionary if no data is found.
    """
    loaded_data = {} # Initialize an empty dictionary to store loaded data.
    print(f"Starting to process files in: {input_folder}\n")

    # Use os.walk to traverse the directory tree.
    # 'root' is the current directory path (e.g., "/home/will/GPS_data/april2017storm/").
    # 'dirnames' is a list of subdirectories in the current 'root' (e.g., ['ns60', 'ns63']).
    # '_' (underscore) is used as a throwaway variable for 'filenames' as it's not used directly here.
    for (root, satnames, _) in os.walk(input_folder):
        # Sort satellite names in numerical order
        sorted_satnames = sorted(satnames, key=lambda s: int(s[2:]))
        # Iterate over each satellite subdirectory name found in the current 'root'.
        for satname in sorted_satnames:
            # Construct the full path to the current satellite's directory.
            sat_dir_path = os.path.join(root, satname)
            print(f"Reading in satellite {satname}")
            # Use glob.glob to find all files matching "ns*.ascii" pattern
            # directly within the current satellite's directory.
            sat_filenames = glob.glob(sat_dir_path + "/ns*ascii")
            # Sort the collected filenames by their date (YYMMDD) component.
            sorted_sat_filenames = sorted(sat_filenames, 
                key=lambda filepath: os.path.basename(filepath).split('_v')[0].split('_')[-1])
            # Attempt to read all sorted files for the current satellite into a single SpaceData object.
            # dm.readJSONheadedASCII can accept a list of file paths.
            loaded_data[satname] = dm.readJSONheadedASCII(sorted_sat_filenames)
    print("Data Loaded \n")    
    return loaded_data

#%% Limit data to selected time period
import datetime as dt
def data_period(data, start_date, stop_date):
    print("Identifying Relevant Time Period...")
    start_object = dt.datetime.strptime(start_date, "%m/%d/%Y")
    start_year = float(start_object.year)
    start_day = float(start_object.timetuple().tm_yday)
    
    stop_object = dt.datetime.strptime(stop_date, "%m/%d/%Y")
    stop_year = float(stop_object.year)
    stop_day = float(stop_object.timetuple().tm_yday)

    time_restricted_data = {}
    for satellite, sat_data in data.items():
        year_mask = (sat_data['year'] >= start_year) & (sat_data['year'] <= stop_year)
        day_mask = (sat_data['decimal_day'] >= start_day) & (sat_data['decimal_day'] <= stop_day)
        time_mask = year_mask & day_mask
        for item, item_data in data[satellite].items():
            if satellite not in time_restricted_data:
                time_restricted_data[satellite] = {}
            time_restricted_data[satellite][item] = item_data[time_mask]
    print("Relevant Time Period Identified \n")
    return time_restricted_data

#%% Steve's date conversion function
import datetime as dt
import spacepy.time as spt
def ticks_from_gps(data, use_astropy=False):
    '''Get a Ticktock from the year and decimal day in GPS time

    Notes
    -----
    1 - The decimal day is given as "GPS time" which is offset
    from UTC by the number of leapseconds since 1980.
    2 - The timestamps correspond to the midpoints of the integration
    intervals
    '''
    print('Converting Time for each Satellite...')
    for satellite, sat_data in data.items():
        year = sat_data['year']
        decday = sat_data['decimal_day']
        intyear = year.astype(int)
        datearray = spt.doy2date(intyear, decday, dtobj=True, flAns=True)
        # this is GPS time, so needs to be adjusted by leap seconds
        GPS0 = dt.datetime(1980, 1, 6)  # Zero epoch for GPS seconds system
        gpsoffset = datearray - GPS0
        gpsseconds = [tt.total_seconds() for tt in gpsoffset]
        data[satellite]['Time'] = spt.Ticktock(gpsseconds, dtype='GPS')
    print('Satellite Times Converted \n')
    return data

# %% 
# Load in data
loaded_data = process_GPS_data(input_folder)

# Restrict to time period
start_date  = "04/21/2017"
stop_date   = "04/26/2017" # exclusive, end of the last day you want to see
storm_data = data_period(loaded_data, start_date, stop_date)

# Convert satellite time to Ticktock object
storm_data = ticks_from_gps(storm_data)

# %% Read in and process REPT Data
analysis_functions_folder = "/home/will/satellite-data-analysis/IRBEM"
sys.path.append(analysis_functions_folder)
import importlib
import analysis_functions
importlib.reload(analysis_functions)
from analysis_functions import process_l2_data

folder_path_l2 = "/mnt/box/Multipoint_Box/REPT_Data/April 2017 Storm/l2/"
if not os.path.exists(folder_path_l2):
    raise FileNotFoundError(f"Error: Folder path not found: {folder_path_l2}")
    
# Get all CDF file paths in the folder
file_paths_l2_A = glob.glob(folder_path_l2 + "rbspa*[!r]*.cdf") 
file_paths_l2_B = glob.glob(folder_path_l2 + "rbspb*[!r]*.cdf") 

# Read in data from RBSP CDF files
print("Processing Flux Data:")
Epoch_A, Position_A, L_A, MLT_A, FESA_A, energy_channels_A = process_l2_data(file_paths_l2_A)
FESA_A = np.where(FESA_A == -1e+31, 0, FESA_A)
Epoch_B, Position_B, L_B, MLT_B, FESA_B, energy_channels_B = process_l2_data(file_paths_l2_B)
FESA_B = np.where(FESA_B == -1e+31, 0, FESA_B)
    
# Handle cases where only A or B data is present (check which lists are not empty)
if not Epoch_A and not FESA_A:
    print("No RBSPA data found in the folder.")
if not Epoch_B and not FESA_B:
    print("No RBSPB data found in the folder.")
        
# Find the earliest and latest Epoch values
if Epoch_A and Epoch_B: 
    min_epoch = min(min(Epoch_A), min(Epoch_B))
    max_epoch = max(max(Epoch_A), max(Epoch_B))
else:
    # Handle cases where either Epoch_A or Epoch_B is empty
    if Epoch_A:
        min_epoch = min(Epoch_A)
        max_epoch = max(Epoch_A)
    elif Epoch_B:
        min_epoch = min(Epoch_B)
        max_epoch = max(Epoch_B)
        
# %% Plot Flux from REPT l2 for 4.2 MeV channel
#from spacepy import pycdf
#cdf_data = pycdf.CDF(file_paths_l2_A[0])
#print(cdf_data)

print("Plotting Data:")
# Create a new colormap object
import matplotlib
from matplotlib import colors
import math
textsize = 16

# Create the figure with subplots
fig, ax = plt.subplots(figsize=(16, 4))

if FESA_A is not None:
    subplot_A = ax.scatter(Epoch_A, L_A, c=FESA_A[:, 4], norm=colors.LogNorm())
    # Set colorbar limits to 5 orders of magnitude
    _, vmax_A = subplot_A.get_clim() 
if FESA_B is not None:
    subplot_B = ax.scatter(Epoch_B, L_B, c=FESA_B[:, 4], norm=colors.LogNorm())
    # Set colorbar limits to 5 orders of magnitude
    _, vmax_B = subplot_B.get_clim() 

# Add labels and title
ax.set_ylabel('McIlwain L', fontsize=textsize)
ax.set_title(f'RBSP REPT {energy_channels_A[4]:.2f} MeV Electron Spin-Averaged Flux', fontsize=textsize)
# Force labels for first and last x-axis tick marks 
min_epoch_plot = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.floor((min_epoch - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
max_epoch_plot = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.ceil((max_epoch - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
ax.set_xlim(min_epoch_plot, max_epoch_plot) 
# Set time labels every 12 hours
ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12) )
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H')) 
ax.tick_params(axis='both', which='major', labelsize=textsize, pad=10)
ax.set_yticks(np.arange(2, 8, 1))  # Set ticks from 2 to 7 with interval 1
ax.set_ylim(2, 7)
ax.grid(True)
  
cbar = plt.colorbar(subplot_A, ax=ax, fraction=0.03, pad=0.01)  # Adjust shrink as needed
vmax = 10**math.ceil(math.log10(max(vmax_A,vmax_B)))
vmin = vmax/10**4
subplot_A.set_clim(vmin, vmax) 
subplot_B.set_clim(vmin, vmax) 
cbar.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), num=5))
# Flux is in (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)
cbar.set_label(label = r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
cbar.ax.tick_params(labelsize=textsize)

# Show the plot
plt.show()

#%% Plot Flux for GPS data
fig, ax = plt.subplots(figsize=(16, 4))

for satname, satdata in storm_data.items():
    eflux = np.array(satdata['electron_diff_flux'])
    mask_eflux = eflux > -1.0
    mask_eflux_4 = mask_eflux[:,10]

    Epoch_A_np = np.array(Epoch_A)
    Epoch_B_np = np.array(Epoch_B)
    mask_mlt = np.zeros_like(satdata['local_time'], dtype=bool)
    for index, time in enumerate(satdata['Time'].UTC[mask_eflux_4]):
        '''
        time_diffs_A = np.abs(Epoch_A_np - time)
        closest_index_A = np.argmin(time_diffs_A)
        if satdata['local_time'][index] >= MLT_A[closest_index_A]-1.5 and satdata['local_time'][index] <= MLT_A[closest_index_A]+1.5:
            mask_mlt[index] = True
        else:
            time_diffs_B = np.abs(Epoch_B_np - time)
            closest_index_B = np.argmin(time_diffs_B)
            if satdata['local_time'][index] >= MLT_B[closest_index_B]-1.5 and satdata['local_time'][index] <= MLT_B[closest_index_B]+1.5:
                mask_mlt[index] = True
        '''
    mask_combined = mask_eflux_4 #& mask_mlt
    '''
    if mask_combined.any():
        print(satname)
        print(sum(mask_combined))
    '''
    subplot_satname = ax.scatter(satdata['Time'].UTC[mask_combined], satdata['L_shell'][mask_combined], c=eflux[mask_combined,10], norm=colors.LogNorm())
    subplot_satname.set_clim(vmin, vmax) 

# Force labels for first and last x-axis tick marks 
min_epoch_plot = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.floor((min_epoch - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
max_epoch_plot = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.ceil((max_epoch - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
ax.set_xlim(min_epoch_plot, max_epoch_plot) 
# Set time labels every 12 hours
ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12) )
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H')) 
ax.tick_params(axis='both', which='major', labelsize=textsize, pad=10)
ax.set_yticks(np.arange(2, 8, 1))  # Set ticks from 2 to 7 with interval 1
ax.set_ylim(2, 7)
ax.grid(True)

cbar = plt.colorbar(subplot_A, ax=ax, fraction=0.03, pad=0.01)  # Adjust shrink as needed
cbar.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), num=5))
# Flux is in (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)
cbar.set_label(label = r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
cbar.ax.tick_params(labelsize=textsize)

# Add Labels
ax.set_ylabel('McIlwain L', fontsize=textsize)
ax.set_title(f'GPS CXD {storm_data['ns59']['electron_diff_flux_energy'][0][10]:.2f} MeV Electron Differential Flux', fontsize=textsize)

# Show the plot
plt.show()

# %% Combined Plot
# Create the figure with subplots
fig, ax = plt.subplots(figsize=(20, 4))

if FESA_A is not None:
    subplot_A = ax.scatter(Epoch_A, L_A, c=FESA_A[:, 4], norm=colors.LogNorm())
    # Set colorbar limits to 5 orders of magnitude
    _, vmax_A = subplot_A.get_clim() 
if FESA_B is not None:
    subplot_B = ax.scatter(Epoch_B, L_B, c=FESA_B[:, 4], norm=colors.LogNorm())
    # Set colorbar limits to 5 orders of magnitude
    _, vmax_B = subplot_B.get_clim() 

# Add labels and title
ax.set_xlabel('UTC', fontsize=textsize)
ax.set_ylabel('L*', fontsize=textsize)
ax.set_title(f'RBSP REPT {energy_channels_A[4]:.2f} MeV Electron Spin-Averaged Flux', fontsize=textsize)
# Force labels for first and last x-axis tick marks 
min_epoch_plot = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.floor((min_epoch - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
max_epoch_plot = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.ceil((max_epoch - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
ax.set_xlim(min_epoch_plot, max_epoch_plot) 
# Set time labels every 12 hours
ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12) )
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H')) 
ax.tick_params(axis='both', which='major', labelsize=textsize)
ax.set_yticks(np.arange(2, 8, 1))  # Set ticks from 2 to 7 with interval 1
ax.set_ylim(2, 7)
ax.grid(True)
  
cbar = plt.colorbar(subplot_A, ax=ax, shrink=0.9)  # Adjust shrink as needed
vmax = 10**math.ceil(math.log10(max(vmax_A,vmax_B)))
vmin = vmax/10**4
subplot_A.set_clim(vmin, vmax) 
subplot_B.set_clim(vmin, vmax) 

for satname, satdata in storm_data.items():
    eflux = np.array(satdata['electron_diff_flux'])
    mask_eflux = eflux > -1.0
    mask_eflux_4 = mask_eflux[:,10]

    Epoch_A_np = np.array(Epoch_A)
    Epoch_B_np = np.array(Epoch_B)
    mask_mlt = np.zeros_like(satdata['local_time'], dtype=bool)
    for index, time in enumerate(satdata['Time'].UTC[mask_eflux_4]):
        time_diffs_A = np.abs(Epoch_A_np - time)
        closest_index_A = np.argmin(time_diffs_A)
        if satdata['local_time'][index] >= MLT_A[closest_index_A]-1.5 and satdata['local_time'][index] <= MLT_A[closest_index_A]+1.5:
            mask_mlt[index] = True
        else:
            time_diffs_B = np.abs(Epoch_B_np - time)
            closest_index_B = np.argmin(time_diffs_B)
            if satdata['local_time'][index] >= MLT_B[closest_index_B]-1.5 and satdata['local_time'][index] <= MLT_B[closest_index_B]+1.5:
                mask_mlt[index] = True
    mask_combined = mask_eflux_4 & mask_mlt

    subplot_satname = ax.scatter(satdata['Time'].UTC[mask_combined], satdata['L_shell'][mask_combined], c=eflux[mask_combined,10], norm=colors.LogNorm())
    subplot_satname.set_clim(vmin, vmax) 

cbar.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), num=5))
# Flux is in (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)
cbar.set_label(label = r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
cbar.ax.tick_params(labelsize=textsize)

# Show the plot
plt.show()

# %% Plot Radial profiles from REPT
def load_data(npzfile):
    print(f"Loading {npzfile}")
    loaded_data = {}
    for satellite, sat_data in npzfile.items():
        loaded_data[satellite] = {}
        if isinstance(sat_data, np.ndarray):
            if sat_data.ndim == 0 and sat_data.dtype == object:
                temp_inner_dict = sat_data.item()
                for item, item_data in temp_inner_dict.items():
                    loaded_data[satellite][item] = item_data
            elif sat_data.ndim > 0:
                loaded_data[satellite] = sat_data
        elif isinstance(sat_data, pd.DataFrame):
            loaded_data[satellite] = sat_data
    print("Data Loaded \n")
    return loaded_data

REPTB_load = np.load('/mnt/box/Multipoint_Box/REPT_Data/plot_data.npz', allow_pickle=True)
REPTB_data = load_data(REPTB_load)
REPTB_load.close()

Mu_set = np.array((4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G
K_set = np.array(0.1) # R_E*G^(1/2)

#time_start  = dt.datetime(2017, 4, 23, 18, 45, 0)
#time_stop   = dt.datetime(2017, 4, 23, 22, 58, 0)

#time_start  = dt.datetime(2017, 4, 24, 17, 7, 0)
#time_stop   = dt.datetime(2017, 4, 24, 21, 35, 0)

time_start  = dt.datetime(2017, 4, 25, 15, 30, 0)
time_stop   = dt.datetime(2017, 4, 25, 19, 57, 0)

# Convert Epoch_A and Epoch_B to NumPy arrays of datetimes
Epoch_B_np = np.array(REPTB_data['Epoch_B_averaged'])

# Define Lstar delta
lstar_delta = 0.1

# Generate Lstar interval boundaries within the time range.
time_range = Epoch_B_np[(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
lstar_range = REPTB_data['Lstar_B_set'][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
psd_range = REPTB_data['psd_B'][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
lstar_min = np.min(lstar_range[~np.isnan(lstar_range)])
lstar_max = np.max(lstar_range[~np.isnan(lstar_range)])
lstar_intervals = np.arange(np.floor(lstar_min / lstar_delta) * lstar_delta, np.ceil(lstar_max / lstar_delta) * lstar_delta + lstar_delta, lstar_delta)

energy_range = REPTB_data['energy_B_set'][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
interpa_range = REPTB_data['FEDU_B_interpa'][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]
interpaE_range = REPTB_data['FEDU_B_interpaE'][(Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop)]

# Initialize arrays to store averaged values.
averaged_lstar = np.zeros(len(lstar_intervals))
averaged_psd = np.zeros((len(lstar_intervals), REPTB_data['psd_B'].shape[1]))

fig, ax = plt.subplots(figsize=(6, 4.5))
color_set = plt.cm.get_cmap('nipy_spectral')(np.linspace(0, 0.875, 256))[np.linspace(0, 255, len(Mu_set), dtype=int)]
color_set[3] = [0, 1, 1, 1]  # Teal


for mu_index in range(len(Mu_set)):
    # Iterate through each Lstar interval.
    for i, lstar_val in enumerate(lstar_intervals):
        # Find indices within the current Lstar interval and time range.
        lstar_start = lstar_val - 1/2 * lstar_delta
        lstar_end = lstar_val + 1/2 * lstar_delta
        interval_indices = np.where((Epoch_B_np >= time_start) & (Epoch_B_np <= time_stop) & (REPTB_data['Lstar_B_set'] >= lstar_start) & (REPTB_data['Lstar_B_set'] < lstar_end))[0]           
        
        # Calculate averages for the current Lstar interval
        averaged_psd[i, mu_index] = np.nanmean(REPTB_data['psd_B'][interval_indices, mu_index])  # average along the first axis, ignoring NaNs.

    # Create a mask to filter out NaN values
    nan_mask = ~np.isnan(averaged_psd[:, mu_index])
    
    # Apply the mask to both averaged_lstar and averaged_psd
    ax.plot(
        lstar_intervals[nan_mask],
        averaged_psd[nan_mask, mu_index],
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

# %% Compare calculated pitch angles
REPTB_load = np.load('/mnt/box/Multipoint_Box/REPT_Data/plot_data.npz', allow_pickle=True)
REPTB_data1 = load_data(REPTB_load)
REPTB_load.close()

REPTB_load = np.load('/mnt/box/Multipoint_Box/REPT_Data/plot_data2.npz', allow_pickle=True)
REPTB_data2 = load_data(REPTB_load)
REPTB_load.close()

fig, ax = plt.subplots(figsize=(14, 2))
ax.scatter(REPTB_data1['Epoch_B_averaged'], REPTB_data1['alpha_B_set'])
ax.set_xlabel('UTC', fontsize=textsize)
ax.set_ylabel(r"Pitch Angle", fontsize=textsize)
ax.tick_params(axis='both', which='major', labelsize=textsize)
ax.set_title('Interpolated Pitch Angle for K=0.1',fontsize=textsize)
ax.grid(True)

fig, ax = plt.subplots(figsize=(14, 2))
ax.scatter(REPTB_data2['Epoch_B_averaged'], REPTB_data2['alpha_B_set'])
ax.set_xlabel('UTC', fontsize=textsize)
ax.set_ylabel(r"Pitch Angle", fontsize=textsize)
ax.tick_params(axis='both', which='major', labelsize=textsize)
ax.set_title('Calculated Pitch Angle (IRBEM AlphaOfK) for K=0.1',fontsize=textsize)
ax.grid(True)

fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(REPTB_data1['alpha_B_set'], REPTB_data2['alpha_B_set'])
ax.set_xlabel(r"PA (Interpolated)", fontsize=textsize)
ax.set_ylabel(r"PA (AlphaOfK)", fontsize=textsize)
ax.tick_params(axis='both', which='major', labelsize=textsize)
ax.set(xlim=(40,90), ylim=(40,65))
ax.plot(np.linspace(0,100,1000),np.linspace(0,100,1000), color='black', linestyle = '--')
ax.set_aspect('equal')
ax.grid(True)

# %% Show local 90 PA distribution

