#%% Initialize
import os
import glob
import spacepy.datamodel as dm
import datetime as dt
import spacepy.time as spt
from spacepy import coordinates as Coords
import numpy as np

from lgmpy import Lgm_Vector
import lgmpy.Lgm_Wrap as lgm_lib
from ctypes import pointer, c_int, c_double

#%% Import GPS data function
def import_GPS(input_folder):
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
    print(f"Starting to process files in: {input_folder}")

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
            print(f"    Reading in satellite {satname}")
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
    
    QD_folder = "/home/will/QinDenton/"
    QD_filenames = []
    current_date_object = dt.datetime.strptime(start_date, "%m/%d/%Y")
    while current_date_object <= dt.datetime.strptime(stop_date, "%m/%d/%Y"):
        print(f"    {current_date_object.date()}")
        QD_year = os.path.join(QD_folder, str(current_date_object.year),'5min')
        QD_filenames.append(os.path.join(QD_year, f"QinDenton_{current_date_object.strftime("%Y%m%d")}_5min.txt"))
        current_date_object += dt.timedelta(days=1)
    QD_data = dm.readJSONheadedASCII(QD_filenames)
    
    print("Relevant Time Period Identified \n")
    return time_restricted_data, QD_data

#%% Import Qin Denton data for selected period
def QD_data_period(QD_folder, start_date, stop_date):
    print("Identifying Relevant Time Period for Qin Denton...")
    start_date = dt.datetime.strptime(start_date, "%m/%d/%Y") 
    stop_date = dt.datetime.strptime(stop_date, "%m/%d/%Y")

    QD_filenames = []
    current_date = start_date
    while current_date <= stop_date:
        QD_year = os.path.join(QD_folder, str(current_date.year),'5min')
        QD_filenames.append(os.path.join(QD_year, f"QinDenton_{current_date.strftime("%Y%m%d")}_5min.txt"))
        current_date += dt.timedelta(days=1)
    QD_data = dm.readJSONheadedASCII(QD_filenames)
    return QD_data

#%% Modified from Steve's date conversion function
def data_from_gps(gps_data_in):
    gps_data_out = {}
    # Modified from Steve's date conversion function
    print('Converting Time for each Satellite...')
    for satellite, sat_data in gps_data_in.items():
        gps_data_out[satellite] = {}
        year = sat_data['year']
        decday = sat_data['decimal_day']
        intyear = year.astype(int)
        datearray = spt.doy2date(intyear, decday, dtobj=True, flAns=True)
        # this is GPS time, so needs to be adjusted by leap seconds
        GPS0 = dt.datetime(1980, 1, 6)  # Zero epoch for GPS seconds system
        gpsoffset = datearray - GPS0
        gpsseconds = [tt.total_seconds() for tt in gpsoffset]
        gps_data_out[satellite]['Epoch'] = spt.Ticktock(gpsseconds, dtype='GPS')
    print('Satellite Times Converted \n')

    print('Converting Position for each Satellite...')
    for satellite, sat_data in gps_data_in.items():
        print(f"    Converting Position for satellite {satellite}")
        R = sat_data['Rad_Re']
        Lat = np.deg2rad(sat_data['Geographic_Latitude'])
        Lon = np.deg2rad(sat_data['Geographic_Longitude'])

        position_init = Coords.Coords(np.column_stack((R,np.pi-Lat,Lon)),'GEO','sph')
        position_init.ticks = gps_data_out[satellite]['Epoch']
        gps_data_out[satellite]['Position'] = position_init.convert('GSM','car')
    print('Satellite Positions Converted \n')

    # Extract relevant data
    chosen_vars = ['local_time',
                   'b_satellite','b_equator',
                   'L_LGM_T89IGRF', 'L_LGM_TS04IGRF',
                   'electron_diff_flux_energy','electron_diff_flux', 'efitpars']
    for satellite, sat_data in gps_data_in.items():
        for var_name in chosen_vars:
            if var_name == 'local_time':
                gps_data_out[satellite]['MLT'] = gps_data_in[satellite][var_name]
            elif var_name == 'electron_diff_flux_energy':
                gps_data_out[satellite]['Energy_Channels'] = gps_data_in[satellite][var_name][0]
            else:
                gps_data_out[satellite][var_name] = gps_data_in[satellite][var_name]
    return gps_data_out

#%% Load preprocessed data from file
def load_preprocessed(npzfile):
    print("Loading Preprocessed Data...")
    loaded_data = {}
    for satellite, sat_data in npzfile.items():
        loaded_data[satellite] = {}
        temp_inner_dict = sat_data.item()
        for item, item_data in temp_inner_dict.items():
            loaded_data[satellite][item] = item_data
    print("Preprocessed Data Loaded \n")
    return loaded_data

#%% Find local pitch angle
def find_local90PA(gps_data_in):
    gps_data_out = gps_data_in
    for satellite, sat_data in gps_data_in.items():
        Beq = sat_data['b_equator']
        Bsat = sat_data['b_satellite']
        mask = np.where(Beq > 0) and np.where(Bsat > 0)
        gps_data_out[satellite]['local90PA'] = np.zeros_like(Beq)
        gps_data_out[satellite]['local90PA'].fill(np.nan)
        gps_data_out[satellite]['local90PA'][mask] = np.rad2deg(np.arcsin(np.sqrt(Beq[mask] / Bsat[mask])))
    return(gps_data_out)

#%% Convert TickTock to Lgm_DateTime
def ticktock_to_Lgm_DateTime(ticktock):
    dt_obj = ticktock.UTC[0]
    lgm_dt = lgm_lib.Lgm_DateTime()
    # Date (YYYYMMDD)
    lgm_dt.Date = dt_obj.year * 10000 + dt_obj.month * 100 + dt_obj.day
    # Time (decimal hours)
    lgm_dt.Time = float(dt_obj.hour + 
                               dt_obj.minute / 60.0 + 
                               dt_obj.second / 3600.0 + 
                               dt_obj.microsecond / 3600000000.0)
    
    # Other fields (populate as needed, based on datetime object attributes)
    lgm_dt.Year = dt_obj.year
    lgm_dt.Month = dt_obj.month
    lgm_dt.Day = dt_obj.day
    lgm_dt.Hour = dt_obj.hour
    lgm_dt.Minute = dt_obj.minute
    lgm_dt.Second = float(dt_obj.second + dt_obj.microsecond / 1e6) # Seconds with microseconds
    lgm_dt.Doy = dt_obj.timetuple().tm_yday # Day of Year

    lgm_dt.TimeSystem = lgm_lib.LGM_TIME_SYS_UTC

    return(lgm_dt)

#%% Find pitch angle corresponding to set K

def AlphaOfK(gps_data_in, K_set, qindenton_data, ext_mag_model = 'T89'):
    gps_data_out = gps_data_in
    alphaofK = {}

    MagInfo = lgm_lib.Lgm_InitMagInfo()
    IntMagModel = c_int(lgm_lib.__dict__[f"LGM_IGRF"])
    ExtMagModel = c_int(lgm_lib.__dict__[f"LGM_EXTMODEL_{ext_mag_model}"])
    lgm_lib.Lgm_Set_MagModel(IntMagModel, ExtMagModel, MagInfo)

    K_set_c = c_double(K_set)

    for satellite, sat_data in gps_data_in.items():
        gps_data_out[satellite]['AlphaofK'] = np.zeros_like(sat_data['b_satellite'])
        gps_data_out[satellite]['AlphaofK'].fill(np.nan)
        for i, epoch in enumerate(sat_data['Epoch']):
            current_time = ticktock_to_Lgm_DateTime(epoch)
            current_vec = Lgm_Vector.Lgm_Vector(*sat_data['Position'][i].data[0])
            lgm_lib.Lgm_Setup_AlphaOfK(current_time, current_vec, MagInfo)
            gps_data_out[satellite]['AlphaofK'][i] = lgm_lib.Lgm_AlphaOfK(K_set_c, MagInfo)
            lgm_lib.Lgm_TearDown_AlphaOfK(MagInfo)
    return(gps_data_out)

