import numpy as np
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
import glob
import spacepy.datamodel as dm
input_folder = "/home/will/GPS_data/april2017storm/"
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
    # Convert the loaded_data dictionary to a new dictionary.
    loaded_data = dict(loaded_data)  
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
        year_key = (sat_data['year'] >= start_year) & (sat_data['year'] <= stop_year)
        day_key = (sat_data['decimal_day'] >= start_day) & (sat_data['decimal_day'] <= stop_day)
        time_key = year_key & day_key
        for item, item_data in data[satellite].items():
            if satellite not in time_restricted_data:
                time_restricted_data[satellite] = {}
            time_restricted_data[satellite][item] = item_data[time_key]
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
stop_date   = "04/25/2017" # inclusive, last day you want to see data
storm_data = data_period(loaded_data, start_date, stop_date)

# Convert satellite time to Ticktock object
storm_data_ticks = ticks_from_gps(storm_data)