#%% Initialize
import os
import glob
import spacepy.datamodel as dm
import datetime as dt
import spacepy.time as spt
from spacepy import coordinates as Coords
import numpy as np
import scipy.constants as sc

from lgmpy import Lgm_Vector
import lgmpy.Lgm_Wrap as lgm_lib
from ctypes import c_int, c_double

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
    # Modified from Steve's date conversion function
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
        data[satellite]['Epoch'] = spt.Ticktock(gpsseconds, dtype='GPS')
    print('Satellite Times Converted \n')
    
    print("Identifying Relevant Time Period...")
    start_object = dt.datetime.strptime(start_date, "%m/%d/%Y")
    stop_object = dt.datetime.strptime(stop_date, "%m/%d/%Y")

    time_restricted_data = {}
    for satellite, sat_data in data.items():
        time_mask = (sat_data['Epoch'].UTC >= start_object) & (sat_data['Epoch'].UTC <= stop_object)
        for item, item_data in data[satellite].items():
            if satellite not in time_restricted_data:
                time_restricted_data[satellite] = {}
            time_restricted_data[satellite][item] = item_data[time_mask]
    
    print("Relevant Time Period Identified \n")
    return time_restricted_data

#%% Extract QinDenton data for the time period
def QinDenton_period(start_date, stop_date): 
    print('Loading QinDenton Data...')
    QD_folder = "/home/will/QinDenton/"
    QD_filenames = []
    current_date_object = dt.datetime.strptime(start_date, "%m/%d/%Y")
    while current_date_object <= dt.datetime.strptime(stop_date, "%m/%d/%Y"):
        QD_year = os.path.join(QD_folder, str(current_date_object.year),'5min')
        QD_filenames.append(os.path.join(QD_year, f"QinDenton_{current_date_object.strftime("%Y%m%d")}_5min.txt"))
        current_date_object += dt.timedelta(days=1)
    global QD_data
    QD_data = dm.readJSONheadedASCII(QD_filenames)
    # Convert DateTime format to datetime
    datetime_format = "%Y-%m-%dT%H:%M:%S"
    for dti, datetime in enumerate(QD_data['DateTime']):
        QD_data['DateTime'][dti] = dt.datetime.strptime(datetime, datetime_format)
    print("QinDenton Data Loaded \n")
    return QD_data


#%% Modified from Steve's date conversion function
def data_from_gps(time_restricted_data):
    gps_data_out = {}

    print('Converting Position for each Satellite...')
    for satellite, sat_data in time_restricted_data.items():
        print(f"    Converting Position for satellite {satellite}")
        gps_data_out[satellite] = {}
        R = sat_data['Rad_Re']
        Lat = np.deg2rad(sat_data['Geographic_Latitude'])
        Lon = np.deg2rad(sat_data['Geographic_Longitude'])

        position_init = Coords.Coords(np.column_stack((R,np.pi-Lat,Lon)),'GEO','sph')
        position_init.ticks = sat_data['Epoch']
        gps_data_out[satellite]['Position'] = position_init.convert('GSM','car')
    print('Satellite Positions Converted \n')

    # Extract relevant data
    chosen_vars = ['Epoch', 'local_time',
                   'b_satellite','b_equator',
                   'L_LGM_T89IGRF', 'L_LGM_TS04IGRF',
                   'electron_diff_flux_energy','electron_diff_flux', 'efitpars']
    for satellite, sat_data in time_restricted_data.items():
        for var_name in chosen_vars:
            if var_name == 'local_time':
                gps_data_out[satellite]['MLT'] = time_restricted_data[satellite][var_name]
            elif var_name == 'electron_diff_flux_energy':
                gps_data_out[satellite]['Energy_Channels'] = time_restricted_data[satellite][var_name][0]
            else:
                gps_data_out[satellite][var_name] = time_restricted_data[satellite][var_name]
    return gps_data_out

#%% Load preprocessed data from file
def load_data(npzfile):
    print(f"Loading {npzfile} Data...")
    loaded_data = {}
    for satellite, sat_data in npzfile.items():
        loaded_data[satellite] = {}
        temp_inner_dict = sat_data.item()
        for item, item_data in temp_inner_dict.items():
            loaded_data[satellite][item] = item_data
    print("Data Loaded \n")
    return loaded_data

#%% Find local pitch angle
def find_local90PA(gps_data):
    local90PA = {}
    for satellite, sat_data in gps_data.items():
        local90PA[satellite] = {}
        Beq = sat_data['b_equator']
        Bsat = sat_data['b_satellite']
        mask = np.where(Beq > 0) and np.where(Bsat > 0)
        local90PA[satellite]['Epoch'] = sat_data['Epoch']
        local90PA[satellite] = np.zeros_like(Beq)
        local90PA[satellite].fill(np.nan)
        local90PA[satellite][mask] = np.rad2deg(np.arcsin(np.sqrt(Beq[mask] / Bsat[mask])))
    return local90PA

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

    return lgm_dt 

#%% Set magnetic field model coefficients to closest time of QinDenton data
def QD_inform_MagInfo(time, MagInfo):
    # Round down to the nearest 5 minutes
    time_dt = time.UTC[0] # Assumes time is a spacepy TickTock object
    minutes_to_subtract = time_dt.minute % 5
    rounded_dt = time_dt - dt.timedelta(
        minutes=minutes_to_subtract,
        seconds=time_dt.second,
        microseconds=time_dt.microsecond
    )
    time_index = int(np.where(QD_data['DateTime']==rounded_dt)[0])
    # Map each variable from QD_data to MagInfo
    MagInfo.contents.By    = c_double(QD_data['ByIMF'][time_index])
    MagInfo.contents.Bz    = c_double(QD_data['BzIMF'][time_index])
    MagInfo.contents.P     = c_double(QD_data['Pdyn'][time_index])
    MagInfo.contents.G1    = c_double(QD_data['G'][time_index][0])
    MagInfo.contents.G2    = c_double(QD_data['G'][time_index][1])
    MagInfo.contents.G3    = c_double(QD_data['G'][time_index][2])
    MagInfo.contents.Kp    = c_int(round(QD_data['Kp'][time_index]))
    MagInfo.contents.fKp   = c_double(QD_data['Kp'][time_index])
    MagInfo.contents.Dst   = c_double(QD_data['Dst'][time_index])
    MagInfo.contents.W[0]  = c_double(QD_data['W'][time_index][0])
    MagInfo.contents.W[1]  = c_double(QD_data['W'][time_index][1])
    MagInfo.contents.W[2]  = c_double(QD_data['W'][time_index][2])
    MagInfo.contents.W[3]  = c_double(QD_data['W'][time_index][3])
    MagInfo.contents.W[4]  = c_double(QD_data['W'][time_index][4])
    MagInfo.contents.W[5]  = c_double(QD_data['W'][time_index][5])
    return

#%% Find pitch angle corresponding to set K
def AlphaOfK(gps_data, K_set, ext_mag_model = 'T89'):
    print('Calculating Pitch Angle for given Ks...')
    alphaofK = {}
    
    MagInfo = lgm_lib.Lgm_InitMagInfo()
    IntMagModel = c_int(lgm_lib.__dict__[f"LGM_IGRF"])
    ExtMagModel = c_int(lgm_lib.__dict__[f"LGM_EXTMODEL_{ext_mag_model}"])
    lgm_lib.Lgm_Set_MagModel(IntMagModel, ExtMagModel, MagInfo)

    K_set_c = c_double(K_set)

    for satellite, sat_data in gps_data.items():
        print(f"    Calculating Alpha for satellite {satellite}")
        alphaofK[satellite] = {}
        alphaofK[satellite]['Epoch'] = sat_data['Epoch']
        alphaofK[satellite]['AlphaofK'] = np.zeros_like(sat_data['b_satellite'])
        alphaofK[satellite]['AlphaofK'].fill(np.nan)
        for i, epoch in enumerate(sat_data['Epoch']):
            current_time = ticktock_to_Lgm_DateTime(epoch)
            current_vec = Lgm_Vector.Lgm_Vector(*sat_data['Position'][i].data[0])
            QD_inform_MagInfo(epoch, MagInfo)
            lgm_lib.Lgm_Setup_AlphaOfK(current_time, current_vec, MagInfo)
            alphaofK[satellite]['AlphaofK'][i] = lgm_lib.Lgm_AlphaOfK(K_set_c, MagInfo)
            lgm_lib.Lgm_TearDown_AlphaOfK(MagInfo)
    print('Pitch Angles Calculated \n')
    return alphaofK

#%% Calculate energy from set mu and alpha:
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6) # this is m_0*c^2
def EnergyofMu(gps_data, Mu_set, AlphaofK):
    print('Calculating Energies for given Mus and Alphas...')
    energyofMu = {}
    # Convert Mu_set to a NumPy array if it's a single value
    Mu_set = np.atleast_1d(Mu_set)

    for satellite, sat_data in gps_data.items():
        energyofMu[satellite] = {}
        energyofMu[satellite]['Epoch'] = sat_data['Epoch']
        energyofMu[satellite]['AlphaofK'] = AlphaofK[satellite]['AlphaofK']
        # Convert Alpha_set to radians
        alpha_rad = np.radians(AlphaofK[satellite]['AlphaofK'])

        # Calculate sin^2(Alpha)
        sin_squared_alpha = np.sin(alpha_rad)**2

        energyofMu[satellite]['EnergyofMu'] = np.zeros((AlphaofK[satellite]['AlphaofK'].shape[0], Mu_set.shape[0]))  # Initialize the output array

        for i, mu in enumerate(Mu_set):
            # Reminder, GPS Bfield data is in Gauss
            energyofMu[satellite]['EnergyofMu'][:, i] = np.sqrt(2 * E0 * mu * (sat_data['b_satellite']) / sin_squared_alpha + E0**2) - E0
    print('Energies Calculated')
    return energyofMu