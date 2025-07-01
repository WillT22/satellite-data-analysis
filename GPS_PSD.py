
#%% Importing all data files
import os
import sys
import datetime as dt
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_script_dir)
import numpy as np
import scipy.constants as sc

import GPS_PSD_func
import importlib
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import (import_GPS, data_period, QD_data_period, data_from_gps, load_preprocessed,
                            find_local90PA, AlphaOfK)

# Initialize global variables
textsize = 16
Re = 6378.137 #Earth's Radius
Mu_set = np.array((4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G
K_set = 0.10 # R_E*G^(1/2)

# Conversions
# electron mass in MeV is (m_e [kg] * c^2 [m^2/s^2]) [J] / (sc.eV [J/eV] * 10^6 [eV/MeV])
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6)
# b_satellite and b_equator are in Gauss: 1 G = 10^5 nT

#%% Main
if __name__ == '__main__':

    # Load in data
    input_folder = "/home/will/GPS_data/april2017storm/"
    loaded_data = import_GPS(input_folder)

### Preprocessing

    # Restrict to time period
    start_date  = "04/21/2017"
    stop_date   = "04/26/2017" # exclusive, end of the last day you want to see
    storm_data_raw, QD_data = data_period(loaded_data, start_date, stop_date)

    # Convert satellite time to Ticktock object
    # and position from spherical GEO to GSM
    storm_data = data_from_gps(storm_data_raw)

    # Save Data for later recall:
    print("Saving Data")
    np.savez('/home/will/GPS_data/april2017storm/processed_gps.npz', **storm_data)

    # Read in data from previous save
    storm_data_load = np.load('/home/will/GPS_data/april2017storm/processed_gps.npz', allow_pickle=True)
    storm_data = load_preprocessed(storm_data_load)

    # Find local pitch angle
    storm_data = find_local90PA(storm_data)

    '''
    test_set = {}
    test_set['ns64'] = {}
    for item, item_data in storm_data['ns64'].items():
        test_set['ns64'][item] = item_data[0]
    '''

    # Find pitch angle corresponding to set K
    #storm_data = AlphaOfK(storm_data, K_set)
