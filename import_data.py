import spacepy.datamodel as dm
import numpy as np
# Update geomagnetic index and leapsecond data
import spacepy.toolbox
#spacepy.toolbox.update(all=True)

#%% Single file example
data = dm.readJSONheadedASCII("/home/will/GPS_data/april2017storm/ns60/ns60_170416_v1.10.ascii")
# This is how to see the full tree
data.tree(verbose=True, attrs=True)
# This is how to see the attributes of one element of the data tree specifically
data['local_time'].attrs

#%% Importing all data files
import os
input_folder = "/home/will/GPS_data/april2017storm/"
loaded_data = {}
def process_GPS_data(input_folder):
    individual_file_data_by_satellite = {}
    for (root, satnames, _) in os.walk(input_folder):
        for satname in satnames:
            sat_dir_path = os.path.join(root, satname)
            print(f"Reading in satellite {satname}")
            for (_, _, sat_filenames) in os.walk(sat_dir_path):
                for sat_filename in sat_filenames:
                    sat_file_path = os.path.join(sat_dir_path, sat_filename)
                    print(f"    Reading in file {sat_filename}")
                    loaded_data_temp = dm.readJSONheadedASCII(sat_file_path)
                    if satname not in loaded_data:
                        loaded_data[satname] = []
                    loaded_data[satname].append(loaded_data_temp)
    return loaded_data

#%% Steve's date conversion function
import datetime as dt
import spacepy.time as spt
def ticks_from_gps(year, decday, use_astropy=False):
    '''Get a Ticktock from the year and decimal day in GPS time

    Notes
    -----
    1 - The decimal day is given as "GPS time" which is offset
    from UTC by the number of leapseconds since 1980.
    2 - The timestamps correspond to the midpoints of the integration
    intervals
    '''
    intyear = year.astype(int)
    datearray = spt.doy2date(intyear, decday, dtobj=True, flAns=True)
    # this is GPS time, so needs to be adjusted by leap seconds
    GPS0 = dt.datetime(1980, 1, 6)  # Zero epoch for GPS seconds system
    gpsoffset = datearray - GPS0
    gpsseconds = [tt.total_seconds() for tt in gpsoffset]
    if not use_astropy:
        return spt.Ticktock(gpsseconds, dtype='GPS')
    else:
        import astropy.time
        return astropy.time.Time(gpsseconds, format='gps')

# %%
# Load in data
loaded_data = process_GPS_data(input_folder)

#for sat_file, sat_data in loaded_data