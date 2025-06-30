
#%% Importing all data files
import os
import sys
import datetime as dt
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_script_dir)
import numpy as np

import GPS_PSD_func
import importlib
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import import_GPS, data_period, data_from_gps

#%% Preprocessing
if __name__ == '__main__':
    """ 
    # Load in data
    input_folder = "/home/will/GPS_data/april2017storm/"
    loaded_data = import_GPS(input_folder)

    # Restrict to time period
    start_date  = "04/21/2017"
    stop_date   = "04/26/2017" # exclusive, end of the last day you want to see
    storm_data_raw = data_period(loaded_data, start_date, stop_date)

    # Convert satellite time to Ticktock object
    # and position from spherical GEO to GSM
    storm_data = data_from_gps(storm_data_raw)

    # Save Data for later recall:
    print("Saving Data")
    np.savez('/home/will/GPS_data/april2017storm/processed_gps.npz', **storm_data)
    """
    # Read in data from previous save
    print("Loading Saved Data")
    storm_data_load = np.load('/home/will/GPS_data/april2017storm/processed_gps.npz', allow_pickle=True)
    storm_data = {}
    for satellite, sat_data in storm_data_load.items():
        storm_data[satellite] = {}
        temp_inner_dict = sat_data.item()
        for item, item_data in temp_inner_dict.items():
            storm_data[satellite][item] = item_data