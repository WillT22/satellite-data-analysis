#%% Initialize
import os
import glob
import spacepy.datamodel as dm
import datetime as dt
import spacepy.time as spt
from spacepy import coordinates as Coords
import numpy as np
import scipy
import scipy.constants as sc
import math
import pandas as pd

from lgmpy import Lgm_Vector
import lgmpy.Lgm_Wrap as lgm_lib
from ctypes import c_int, c_double, pointer

E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6) # this is m_0*c^2

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
    '''
    Processes and filters satellite data by a specified time period.
    Args:
    data (dict): A nested dictionary where keys are satellite names, and values
                 are dictionaries containing data arrays (e.g., 'year', 'decimal_day',
                 and other data variables). This is typically the output from 'import_GPS'.
    start_date (str): The start date for filtering, in "MM/DD/YYYY" format.
    stop_date (str): The stop date for filtering (exclusive), in "MM/DD/YYYY" format.

    '''
    # --- Phase 1: Convert Year/Decimal_Day to SpacePy Ticktock Epoch ---
    # Modified from Steve's date conversion function
    print('Converting Time for each Satellite...')

    # Iterate through each satellite and its associated data in the input 'data' dictionary.
    # 'data' is modified in-place to add the 'Epoch' key.
    for satellite, sat_data in data.items():
        # Extract 'year' and 'decimal_day' NumPy arrays for the current satellite.
        year = sat_data['year']
        decday = sat_data['decimal_day']

        # Convert the 'year' array to integer type, as doy2date expects integer years.
        intyear = year.astype(int)

        # Convert Day-of-Year (Doy) and Year to datetime objects using spacepy.time.doy2date.
        datearray = spt.doy2date(intyear, decday, dtobj=True, flAns=True)
        # --- Adjusting for GPS Time Offset ---
        # this is GPS time, so needs to be adjusted by leap seconds
        GPS0 = dt.datetime(1980, 1, 6)  # Zero epoch for GPS seconds system
        # Calculate the time difference between each datetime in 'datearray' and the GPS epoch.
        gpsoffset = datearray - GPS0
        # Convert each time difference object to total seconds.
        gpsseconds = [tt.total_seconds() for tt in gpsoffset]
        # Create a spacepy.time.Ticktock object using the GPS seconds.
        data[satellite]['Epoch'] = spt.Ticktock(gpsseconds, dtype='GPS')
    print('Satellite Times Converted \n')
    
    print("Identifying Relevant Time Period...")

    time_restricted_data = {}
    # Iterate through each satellite's data again.
    for satellite, sat_data in data.items():
        # Create a boolean mask for the 'Epoch' data (which is a Ticktock object) between the date bounds
        time_mask = (sat_data['Epoch'].UTC >= start_date) & (sat_data['Epoch'].UTC < stop_date)
        for item, item_data in data[satellite].items():
            # Initialize the satellite's dictionary in time_restricted_data if it doesn't exist
            if satellite not in time_restricted_data:
                time_restricted_data[satellite] = {}
            # Apply the time_mask and extract temporally relevant data
            time_restricted_data[satellite][item] = item_data[time_mask]
    
    print("Relevant Time Period Identified \n")
    return time_restricted_data

#%% Extract QinDenton data for the time period
def QinDenton_period(start_date, stop_date): 
    print('Loading QinDenton Data...')
    QD_folder = "/home/will/QinDenton/"
    QD_filenames = []
    current_date_object = start_date
    while current_date_object <= stop_date:
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

#%% Extract information for only relevant Lshells
def limit_Lshell(time_restricted_data, Lshell, intMag = 'IGRF', extMag = 'T89'):
    model_var = f"L_LGM_{extMag}{intMag}"
    lshell_time_restricted_data = {}
    for satellite, sat_data in time_restricted_data.items():
        lshell_time_restricted_data[satellite] = {}
        lshell_time_restricted_data[satellite] = sat_data[sat_data[model_var] <= Lshell]
    return lshell_time_restricted_data

#%% Set magnetic field model coefficients to closest time of QinDenton data
## Someday, replace this with Lgm_QinDenton in LANLGeoMag...
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

#%% Find local pitch angle
def find_local90PA(sat_data):
    local90PA = {}
    Beq = sat_data['b_equator']
    Bsat = sat_data['b_satellite']
    mask = np.where(Beq > 0) and np.where(Bsat > 0)
    local90PA = np.zeros_like(Beq)
    local90PA.fill(np.nan)
    local90PA[mask] = np.rad2deg(np.arcsin(np.sqrt(Beq[mask] / Bsat[mask])))
    return local90PA

#%% Find Loss Cone
def find_Loss_Cone(sat_data, height = 100, extMag='T89'):
    
    print(f'        Finding Loss Cone...')
    
    MagInfo = lgm_lib.Lgm_InitMagInfo()
    IntMagModel = c_int(lgm_lib.__dict__[f"LGM_IGRF"])
    ExtMagModel = c_int(lgm_lib.__dict__[f"LGM_EXTMODEL_{extMag}"])
    lgm_lib.Lgm_Set_MagModel(IntMagModel, ExtMagModel, MagInfo)

    b_min = np.zeros(len(sat_data['Epoch']))
    b_footpoint = np.zeros(len(sat_data['Epoch']))
    loss_cone = np.zeros(len(sat_data['Epoch']))
        
    for i_epoch, epoch in enumerate(sat_data['Epoch']):
        current_time = ticktock_to_Lgm_DateTime(epoch, MagInfo.contents.c)
        lgm_lib.Lgm_Set_Coord_Transforms(current_time.contents.Date, current_time.contents.Time, MagInfo.contents.c)
        current_vec = Lgm_Vector.Lgm_Vector(*sat_data['Position'][i_epoch].data[0])
        south_vec = Lgm_Vector.Lgm_Vector()
        north_vec = Lgm_Vector.Lgm_Vector()
        minB_vec = Lgm_Vector.Lgm_Vector()
        QD_inform_MagInfo(epoch, MagInfo)
                
        lgm_lib.Lgm_Trace(pointer(current_vec), pointer(south_vec), pointer(north_vec), pointer(minB_vec),
                            height, 0.01, 1e-7, MagInfo)
                
        b_min[i_epoch] = MagInfo.contents.Bmin * 1e-5
        if sat_data['Position'][i_epoch].z >= 0:
            b_footpoint[i_epoch] = MagInfo.contents.Ellipsoid_Footprint_Bn * 1e-5
        else:
            b_footpoint[i_epoch] = MagInfo.contents.Ellipsoid_Footprint_Bs * 1e-5

    loss_cone = np.rad2deg(np.arcsin(np.sqrt(b_min/b_footpoint)))
    print(f'        Loss Cone Found')
    return b_min, b_footpoint, loss_cone

#%% Extract relevant information from time processed data
def data_from_gps(time_restricted_data, Lshell = [], intMag = 'IGRF', extMag = 'T89'):
    gps_data_out = {}
    model_var = f"L_LGM_{extMag}{intMag}"

    chosen_vars = ['Epoch', 'local_time',
                   'b_satellite','b_equator',
                   'L_LGM_T89IGRF', 'L_LGM_TS04IGRF',
                   'electron_diff_flux_energy','electron_diff_flux', 'efitpars']
    
    print('Processing Data for each Satellite...')
    for satellite, sat_data in time_restricted_data.items():
        print(f"    Processing Data for satellite {satellite}")
        gps_data_out[satellite] = {}
        if isinstance(Lshell, (int, float)):
            Lmask = sat_data[model_var] < Lshell
        elif not Lshell:
            Lmask = np.full(sat_data[model_var].shape, True, dtype=bool)
        else:
            print("Error: Lshell must be a scalar")

        efit_mask = ((np.max(np.log10(time_restricted_data[satellite]['model_counts_electron_fit'][:,0:5]/time_restricted_data[satellite]['electron_diff_flux'][:,0:5]),axis=1) <= 0.11) 
                                                | (np.sum(time_restricted_data[satellite]['electron_diff_flux'][:,0:5]==-1,axis=1)==0))
        mask = Lmask & efit_mask

        R = sat_data['Rad_Re'][mask]
        Lat = np.deg2rad(sat_data['Geographic_Latitude'][mask])
        Lon = np.deg2rad(sat_data['Geographic_Longitude'][mask])

        position_init = Coords.Coords(np.column_stack((R,np.pi-Lat,Lon)),'GEO','sph')
        position_init.ticks = sat_data['Epoch'][mask]
        gps_data_out[satellite]['Position'] = position_init.convert('GSM','car')

        for var_name in chosen_vars:
            if var_name == 'local_time':
                gps_data_out[satellite]['MLT'] = time_restricted_data[satellite][var_name][mask]
            elif var_name == 'electron_diff_flux_energy':
                gps_data_out[satellite]['Energy_Channels'] = time_restricted_data[satellite][var_name][0]
            else:
                gps_data_out[satellite][var_name] = time_restricted_data[satellite][var_name][mask]

        gps_data_out[satellite]['local90PA'] = find_local90PA(gps_data_out[satellite])
        gps_data_out[satellite]['b_min'], gps_data_out[satellite]['b_footpoint'], gps_data_out[satellite]['loss_cone'] = find_Loss_Cone(gps_data_out[satellite])
        
    print('Satellite Data Processed \n')

    return gps_data_out

#%% Load preprocessed data from file
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

#%% Convert TickTock to Lgm_DateTime
def ticktock_to_Lgm_DateTime(ticktock, c):
    dt_obj = ticktock.UTC[0]
    lgm_dt = lgm_lib.Lgm_DateTime_Create(dt_obj.year, dt_obj.month, dt_obj.day, 
                                dt_obj.hour+dt_obj.minute/60+dt_obj.second/3600, lgm_lib.LGM_TIME_SYS_UTC, c)
    return lgm_dt 

#%% Find pitch angle corresponding to set K
def AlphaOfK(gps_data, K_set, extMag = 'T89'):
    
    '''
    alphaofK is structured like:
        Satellite name
            Epoch: vector of TickTock time objects
            K_set: vector of set Ks used in calculation
            AlphaofK: Pandas DataFrame columns by time and indexed by K_set
    '''

    print('Calculating Pitch Angle for given Ks...')
    alphaofK = {}
    
    # Determine the size of the first dimension based on K_set's type
    if isinstance(K_set, (np.ndarray, list)):
        k_dim_size = len(K_set)
    else: # Assume it's a scalar (float, int) if not an array/list
        k_dim_size = 1

    MagInfo = lgm_lib.Lgm_InitMagInfo()
    IntMagModel = c_int(lgm_lib.__dict__[f"LGM_IGRF"])
    ExtMagModel = c_int(lgm_lib.__dict__[f"LGM_EXTMODEL_{extMag}"])
    lgm_lib.Lgm_Set_MagModel(IntMagModel, ExtMagModel, MagInfo)

    for satellite, sat_data in gps_data.items():
        print(f"    Calculating Alpha for satellite {satellite}")
        alphaofK[satellite] = np.zeros((len(sat_data['Epoch']),k_dim_size))
        alphaofK[satellite].fill(np.nan)
        for i_K in range(k_dim_size):
            if isinstance(K_set, (np.ndarray, list)):
                current_K_value = K_set[i_K]
            else:
                current_K_value = K_set
            for i_epoch, epoch in enumerate(sat_data['Epoch']):
                current_time = ticktock_to_Lgm_DateTime(epoch, MagInfo.contents.c)
                current_vec = Lgm_Vector.Lgm_Vector(*sat_data['Position'][i_epoch].data[0])
                QD_inform_MagInfo(epoch, MagInfo)
                lgm_lib.Lgm_Setup_AlphaOfK(current_time, current_vec, MagInfo)
                alphaofK[satellite][i_epoch,i_K] = lgm_lib.Lgm_AlphaOfK(current_K_value, MagInfo)
                lgm_lib.Lgm_TearDown_AlphaOfK(MagInfo)
        if k_dim_size > 1:
            epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
            alphaofK[satellite] = pd.DataFrame(alphaofK[satellite], index=epoch_str, columns=K_set)
    print('Pitch Angles Calculated \n')
    return alphaofK

#%% Calculate Mu from energy channels and set alpha:
def MuofEnergyAlpha(gps_data, alphaofK):
    
    '''
    muofenergyalpha is structured like:
        Satellite name
            Epoch: vector of TickTock time objects
            Energy_Channels: vector of instrument energy channels used in calculation
            MuofEnergyAlpha: dictionary separated by previously set K values
                each K index is a Pandas DataFrame columns by time and indexed by energy channel
    '''
    
    print('Calculating Mus for Energy Channels and Alphas...')
    muofenergyalpha = {}
    # Convert Mu_set to a NumPy array if it's a single value

    Mu_bounds = {}
    Mu_min = 1e10
    Mu_max = 0
    for satellite, sat_data in gps_data.items():
        Mu_bounds[satellite] = {}
        muofenergyalpha[satellite] = {}
        muofenergyalpha[satellite]['Epoch'] = sat_data['Epoch']
        muofenergyalpha[satellite]['Energy_Channels'] = sat_data['Energy_Channels']

        # Initialize the output array
        muofenergyalpha[satellite]['MuofEnergyAlpha'] = {}
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        
        # Determine the size of the first dimension based on K_set's type
        K_set = np.array(list(alphaofK[satellite].columns.tolist()), dtype=float)
        if isinstance(K_set, (np.ndarray, list)):
            k_dim_size = len(K_set)
        else: # Assume it's a scalar (float, int) if not an array/list
            k_dim_size = 1
        
        for i_K in range(k_dim_size):
            if isinstance(K_set, (np.ndarray, list)):
                K = K_set[i_K]
            else:
                K = K_set
            muofenergyalpha[satellite]['MuofEnergyAlpha'][K] = np.zeros((len(sat_data['Epoch']),sat_data['Energy_Channels'].shape[0]))
            # Convert Alpha_set to radians
            if isinstance(alphaofK[satellite], pd.DataFrame):
                alpha_rad = np.radians(alphaofK[satellite].values[:,i_K])
            else:
                alpha_rad = np.radians(alphaofK[satellite][:,i_K])
            # Calculate sin^2(Alpha)
            sin_squared_alpha = np.sin(alpha_rad)**2
            for ch, channel in enumerate(sat_data['Energy_Channels']):
                # Reminder, GPS Bfield data is in Gauss
                muofenergyalpha[satellite]['MuofEnergyAlpha'][K][:,ch] = (channel**2 + 2*channel*E0) * sin_squared_alpha / (2*E0*sat_data['b_equator'])

            Mu_bounds[satellite][K] = np.zeros(2)
            Mu_bounds[satellite][K][0] = np.min(muofenergyalpha[satellite]['MuofEnergyAlpha'][K][:,0])
            if Mu_bounds[satellite][K][0] < Mu_min:
                Mu_min = Mu_bounds[satellite][K][0]
            Mu_bounds[satellite][K][1] = np.max(muofenergyalpha[satellite]['MuofEnergyAlpha'][K][:,-1])
            if Mu_bounds[satellite][K][1] > Mu_max:
                Mu_max = Mu_bounds[satellite][K][1]
            muofenergyalpha[satellite]['MuofEnergyAlpha'][K] = pd.DataFrame(muofenergyalpha[satellite]['MuofEnergyAlpha'][K], index=epoch_str, columns=sat_data['Energy_Channels'])
    Mu_bounds['Total'] = np.array((Mu_min, Mu_max))

    magnitude_min = 10**math.floor(math.log10(Mu_bounds['Total'][0]))
    Mu_min_sci = Mu_bounds['Total'][0]/(magnitude_min/10)
    Mu_min_sci_round = math.floor(Mu_min_sci)
    Mu_min_round = Mu_min_sci_round * (magnitude_min/10)

    magnitude_max = 10**math.floor(math.log10(Mu_bounds['Total'][1]))
    Mu_max_sci = Mu_bounds['Total'][1]/(magnitude_max/10)
    Mu_max_sci_round = math.ceil(Mu_max_sci)
    Mu_max_round = Mu_max_sci_round * (magnitude_max/10)

    Mu_bounds['Rounded'] = np.array((Mu_min_round, Mu_max_round))
    
    print('Mus Calculated \n')
    return muofenergyalpha, Mu_bounds

#%% Calculate energy from set mu and alpha:
def EnergyofMuAlpha(gps_data, Mu_set, alphaofK):
    
    '''
    energyofmualpha is structured like:
        Satellite name
            Epoch: vector of TickTock time objects
            AlphaofK: Pandas DataFrame columns by time and indexed by K_set
            EnergyofMuAlpha: dictionary separated by previously set K values
                each K index is a Pandas DataFrame columns by time and indexed by Mu_set
    '''
    
    print('Calculating Energies for given Mus and Alphas...')
    energyofmualpha = {}
    # Convert Mu_set to a NumPy array if it's a single value
    Mu_set = np.atleast_1d(Mu_set)

    for satellite, sat_data in gps_data.items():
        energyofmualpha[satellite] = {}
        # Convert Alpha_set to radians
        alpha_rad = np.radians(alphaofK[satellite])

        # Calculate sin^2(Alpha)
        sin_squared_alpha = np.sin(alpha_rad)**2 
        
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        K_set = np.array(alphaofK[satellite].columns.tolist(), dtype=float)
        if isinstance(K_set, (np.ndarray, list)):
            k_dim_size = len(K_set)
        else: # Assume it's a scalar (float, int) if not an array/list
            k_dim_size = 1
        
        for i_K in range(k_dim_size):
            if isinstance(K_set, (np.ndarray, list)):
                K = K_set[i_K]
            else:
                K = K_set
            energyofmualpha[satellite][K] = np.zeros((len(sat_data['Epoch']),Mu_set.shape[0]))
            # Convert Alpha_set to radians
            if isinstance(alphaofK[satellite], pd.DataFrame):
                alpha_rad = np.radians(alphaofK[satellite].values[:,i_K])
            else:
                alpha_rad = np.radians(alphaofK[satellite][:,i_K])
            # Calculate sin^2(Alpha)
            sin_squared_alpha = np.sin(alpha_rad)**2
            for i_Mu, mu in enumerate(Mu_set):
                # Reminder, GPS Bfield data is in Gauss
                energyofmualpha[satellite][K][:,i_Mu] = np.sqrt(2 * E0 * mu * sat_data['b_equator'] / sin_squared_alpha + E0**2) - E0
            energyofmualpha[satellite][K] = pd.DataFrame(energyofmualpha[satellite][K], index=epoch_str, columns=Mu_set)
    print('Energies Calculated \n')
    return energyofmualpha

#%% Calculate Energy Spectra
def reletavistic_Maxwellian(energies, n, T): # Based on Maxwell-Juttner distribution from gps data readme
    c_cms = sc.c * 10**2
    p = np.sqrt((energies + E0)**2 - E0**2) / sc.c # reletavistic momentum in MeV/c
    K2 = scipy.special.kn(2, E0/T) # modified Bessel function of the second kind
    j_MJ = n * c_cms /(4*np.pi*T*K2*np.exp(E0/T)) * p**2*sc.c**2/E0**2 * np.exp(-energies/T)
    return j_MJ

def Gaussian(energies, n, mu, sigma):
    c_cms = sc.c * 10**2
    p = np.sqrt((energies + E0)**2 - E0**2) / sc.c # reletavistic momentum in MeV/c
    j_G = n * np.exp(-np.log(p*sc.c/mu)**2/(2*sigma**2))
    return j_G

def energy_spectra(gps_data, energyofMuAlpha):
    print('Calculating Energy Spectra for given Mus and Alphas...')
    j_CXD = {}
    for satellite, sat_data in gps_data.items():
        echannel_min = sat_data['Energy_Channels'][0]
        echannel_max = sat_data['Energy_Channels'][-1]
        
        j_CXD[satellite] = {}
        
        efitpars = sat_data['efitpars']
        n1      = efitpars[:,0]     # number density of MJ1
        T1      = efitpars[:,1]     # temperature of MJ1
        n2      = efitpars[:,2]     # number density of MJ2
        T2      = efitpars[:,3]     # temperature of MJ2
        n3      = efitpars[:,4]     # number density of MJ3
        T3      = efitpars[:,5]     # temperature of MJ3
        nG      = efitpars[:,6]     # number density of Gaussian
        muG     = efitpars[:,7]     # reletavistic momentum at Gaussian peak
        sigma   = efitpars[:,8]     # standard deviation of Gaussian

        energy_data = energyofMuAlpha[satellite]
        for K_val, K_data in energy_data.items():
            Mu_set = np.array(list(K_data.keys()), dtype=float)
            epoch_list = K_data[Mu_set[0]].index.tolist()
            j_CXD[satellite][K_val] = np.zeros((len(epoch_list),len(Mu_set)))
            for Mu_val, Mu_data in K_data.items():
                i_Mu = np.where(Mu_set == Mu_val)[0][0]

                energies = Mu_data.values

                # Do NOT extrapolate outside of energy channel range!
                energy_mask = (energies >= echannel_min) & (energies <= echannel_max)
                j_MJ1 = reletavistic_Maxwellian(energies,n1,T1)
                j_MJ2 = reletavistic_Maxwellian(energies,n2,T2)
                j_MJ3 = reletavistic_Maxwellian(energies,n3,T3)
                j_G = Gaussian(energies,nG,muG,sigma)

                j_CXD[satellite][K_val][energy_mask,i_Mu] = j_MJ1[energy_mask] + j_MJ2[energy_mask] + j_MJ3[energy_mask] + j_G[energy_mask]

            j_CXD[satellite][K_val] = pd.DataFrame(j_CXD[satellite][K_val], index=epoch_list, columns=Mu_set) 
    print('Energy Spectra Calculated\n')
    return j_CXD

#%% Transform from flux to PSD
# Define the relativistic energy conversion factor for an electron.
# This converts the rest mass energy of an electron (m_0*c^2) from Joules to MeV.
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6)

def find_psd(j_CXD, energyofMuAlpha):
    print('Calculating PSD for set Mus and Ks...')
    psd = {}
    for satellite, sat_flux in j_CXD.items():
        psd[satellite] = {}
        for K_val, K_data in sat_flux.items():
            Mu_set = np.array(K_data.columns.tolist(), dtype=float)
            epoch_list = K_data[Mu_set[0]].index.tolist()
            psd[satellite][K_val] = np.zeros((len(epoch_list),len(Mu_set)))
            psd[satellite][K_val].fill(np.nan)
            energy_data = energyofMuAlpha[satellite][K_val].values
            for i_Mu in range(len(Mu_set)):
                mask = ~((np.isnan(K_data.values[:,i_Mu])) | (K_data.values[:,i_Mu]==0))
                E_rel = energy_data[mask, i_Mu]**2 + 2 * energy_data[mask, i_Mu] * E0
                psd[satellite][K_val][mask, i_Mu] = K_data.values[mask,i_Mu] / E_rel * 1.66e-10 * 1e-3 * 200.3
            psd[satellite][K_val] = pd.DataFrame(psd[satellite][K_val], index=epoch_list, columns=Mu_set)
    print('PSD Calculated\n')
    return psd

#%% Calculate L_star
def find_Lstar(gps_data, alphaofK, intMag = 'IGRF', extMag = 'T89'):
    print('Finding L* for set Ks...')
    #MagEphemInfo = lgm_lib.Lgm_InitMagEphemInfo(1,1)
    LstarInfo = lgm_lib.InitLstarInfo(0)
    IntMagModel = c_int(lgm_lib.__dict__[f"LGM_IGRF"])
    ExtMagModel = c_int(lgm_lib.__dict__[f"LGM_EXTMODEL_{extMag}"])
    lgm_lib.Lgm_Set_MagModel(IntMagModel, ExtMagModel, LstarInfo.contents.mInfo)
    
    for satellite, sat_data in gps_data.items():
        print(f"    Calculating L* for satellite {satellite}")
        gps_data[satellite]['Lstar'] = np.zeros_like(alphaofK[satellite].values)
        K_set = np.array(list(alphaofK[satellite].columns.tolist()), dtype=float)
        for i_epoch, epoch in enumerate(sat_data['Epoch']):
            for i_K in range(len(K_set)):
                # Could possibly speed up with NewTimeLstarInfo
                current_time = ticktock_to_Lgm_DateTime(epoch, LstarInfo.contents.mInfo.contents.c)
                lgm_lib.Lgm_Set_Coord_Transforms(current_time.contents.Date, current_time.contents.Time, LstarInfo.contents.mInfo.contents.c)
                current_vec = Lgm_Vector.Lgm_Vector(*sat_data['Position'][i_epoch].data[0])
                QD_inform_MagInfo(epoch, LstarInfo.contents.mInfo)
                LstarInfo.contents.PitchAngle = c_double(alphaofK[satellite].values[i_epoch,i_K])
                LstarInfo.contents.mInfo.contents.Bm = 0
                lgm_lib.Lstar(pointer(current_vec), LstarInfo)
                gps_data[satellite]['Lstar'][[i_epoch,i_K]] = LstarInfo.contents.LS
    print('L* Found\n')
    return gps_data