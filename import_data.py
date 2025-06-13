import numpy as np
import matplotlib.pyplot as plt
import spacepy.plot as splot
# Update geomagnetic index and leapsecond data
import spacepy.toolbox
#spacepy.toolbox.update(all=True)

#%% Single file example
#data = dm.readJSONheadedASCII("/home/will/GPS_data/april2017storm/ns60/ns60_170416_v1.10.ascii")
# This is how to see the full tree
#data.tree(verbose=True, attrs=True)
# This is how to see the attributes of one element of the data tree specifically
#data['local_time'].attrs

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
        if not use_astropy:
            data[satellite]['Time'] = spt.Ticktock(gpsseconds, dtype='GPS')
        else:
            import astropy.time
            data[satellite]['Time'] = astropy.time.Time(gpsseconds, format='gps')
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
from analysis_functions import time_average_FESA

folder_path_l2 = "/mnt/box/Multipoint_Box/REPT_Data/April 2017 Storm/l2/"
if not os.path.exists(folder_path_l2):
    raise FileNotFoundError(f"Error: Folder path not found: {folder_path_l2}")
    
# Get all CDF file paths in the folder
file_paths_l2_A = glob.glob(folder_path_l2 + "rbspa*[!r]*.cdf") 
file_paths_l2_B = glob.glob(folder_path_l2 + "rbspb*[!r]*.cdf") 

# Read in data from RBSP CDF files
print("Processing Flux Data:")
Epoch_A, Position_A, L_star_A, MLT_A, FESA_A, energy_channels_A = process_l2_data(file_paths_l2_A)
FESA_A = np.where(FESA_A == -1e+31, 0, FESA_A)
Epoch_B, Position_B, L_star_B, MLT_B, FESA_B, energy_channels_B = process_l2_data(file_paths_l2_B)
FESA_B = np.where(FESA_B == -1e+31, 0, FESA_B)
    
# Handle cases where only A or B data is present (check which lists are not empty)
if not Epoch_A and not FEDU_A:
    print("No RBSPA data found in the folder.")
if not Epoch_B and not FEDU_B:
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
fig, ax = plt.subplots(figsize=(20, 4))

if FESA_A is not None:
    subplot_A = ax.scatter(Epoch_A, L_star_A, c=FESA_A[:, 4], norm=colors.LogNorm())
    # Set colorbar limits to 5 orders of magnitude
    _, vmax_A = subplot_A.get_clim() 
if FESA_B is not None:
    subplot_B = ax.scatter(Epoch_B, L_star_B, c=FESA_B[:, 4], norm=colors.LogNorm())
    # Set colorbar limits to 5 orders of magnitude
    _, vmax_B = subplot_B.get_clim() 

# Add labels and title
ax.set_ylabel('L', fontsize=textsize)
ax.set_title(f'RBSP REPT {energy_channels_A[4]:.2f} MeV Electron Spin-Averaged Flux', fontsize=textsize)
# Force labels for first and last x-axis tick marks 
min_epoch_plot = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.floor((min_epoch - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
max_epoch_plot = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.ceil((max_epoch - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
ax.set_xlim(min_epoch, max_epoch) 
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
cbar.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), num=5))
# Flux is in (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)
cbar.set_label(label = r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
cbar.ax.tick_params(labelsize=textsize)

# Add x-axis label for last plot
ax.set_xlabel('UTC', fontsize=textsize)

# Show the plot
plt.show()

#%% Plot Flux for GPS data
fig, ax = plt.subplots(figsize=(20, 4))

for satname, satdata in storm_data.items():
    eflux = np.array(satdata['electron_diff_flux'])
    mask_eflux = eflux > -1.0
    mask_eflux_4 = mask_eflux[:,10]

    Epoch_A_np = np.array(Epoch_A)
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

# Force labels for first and last x-axis tick marks 
min_epoch_plot = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.floor((min_epoch - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
max_epoch_plot = dt.datetime(1970, 1, 1) + dt.timedelta(hours=math.ceil((max_epoch - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
ax.set_xlim(min_epoch, max_epoch) 
# Set time labels every 12 hours
ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12) )
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H')) 
ax.tick_params(axis='both', which='major', labelsize=textsize)
ax.set_yticks(np.arange(2, 8, 1))  # Set ticks from 2 to 7 with interval 1
ax.set_ylim(2, 7)
ax.grid(True)

cbar = plt.colorbar(subplot_A, ax=ax, shrink=0.9)  # Adjust shrink as needed
cbar.set_ticks(np.logspace(np.log10(vmin), np.log10(vmax), num=5))
# Flux is in (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)
cbar.set_label(label = r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
cbar.ax.tick_params(labelsize=textsize)

# Add x-axis label for last plot
ax.set_xlabel('UTC', fontsize=textsize)

# Show the plot
plt.show()

# %%
