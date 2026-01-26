#%% Initialize
import os
import sys
import spacepy.datamodel as dm
import datetime as dt
import spacepy.time as spt
from spacepy.time import Ticktock
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import scipy.constants as sc
import math
import pandas as pd

from lgmpy import Lgm_Vector
import lgmpy.Lgm_Wrap as lgm_lib
from ctypes import c_int, c_long, c_double, pointer

# Physical Constants
# Rest mass energy of an electron in MeV (m_0 * c^2)
global E0
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6)

#%% Load preprocessed data from file
def load_data(npzfile):
    """
    Reconstructs a dictionary structure from a loaded NumPy .npz file.
    Handles 0-d object arrays and DataFrames.

    Args:
        npzfile (NpzFile): The object returned by np.load().

    Returns:
        dict: Reconstructed dictionary of satellite data.
    """

    print(f"Loading {npzfile}")
    loaded_data = {}

    for satellite, sat_data in npzfile.items():
        loaded_data[satellite] = {}
        # Handle 0-d arrays that wrap dictionaries (common numpy save behavior for dicts)
        if isinstance(sat_data, np.ndarray):
            # Check if the array is 0-dimensional (scalar-like), which often happens
            # when a dictionary is saved directly into a numpy array without pickling.
            # This creates a 0-d array of dtype 'object' wrapping the dictionary.
            if sat_data.ndim == 0 and sat_data.dtype == object:

                # Extract the actual Python dictionary object from the 0-d numpy wrapper
                temp_inner_dict = sat_data.item()

                # Iterate through the extracted dictionary and store items in the loaded structure
                for item, item_data in temp_inner_dict.items():
                    loaded_data[satellite][item] = item_data
            
            # If the array has dimensions (is a standard data array), store it directly
            elif sat_data.ndim > 0:
                loaded_data[satellite] = sat_data
        elif isinstance(sat_data, pd.DataFrame):
            loaded_data[satellite] = sat_data

    return loaded_data

#%% Limit data to selected time period
def data_period(sat_data, start_date, stop_date):
    """
    Filters satellite data dictionary to a specific time range.

    Args:
        sat_data (dict): Satellite data dictionary.
        start_date (datetime): Start time (inclusive).
        stop_date (datetime): Stop time (exclusive).

    Returns:
        dict: Filtered dictionary containing only data within the time window.
    """
    
    if ('Epoch' in sat_data) == False:
        from GPS_PSD_func import convert_time
        sat_data = convert_time(sat_data)

    time_restricted_data = {}
    # Iterate through each satellite's data again.
    # Create a boolean mask for the 'Epoch' data (which is a Ticktock object) between the date bounds
    time_mask = (sat_data['Epoch'].UTC >= start_date) & (sat_data['Epoch'].UTC < stop_date)

    if np.sum(time_mask) == 0:
        print(f"No data within time period")

    # Filter Time Object
    filtered_time = sat_data['Epoch'].UTC[time_mask]
    time_restricted_data['Epoch'] = Ticktock(filtered_time, dtype='UTC')
    
    # Filter all other array data
    for item, item_data in sat_data.items():
        if item == 'Epoch': # Skip 'Epoch' as it's already handled above
            continue
        
        # Keep metadata items that shouldn't be sliced
        if item in ['Energy_Channels', 'Pitch_Angles']:
            time_restricted_data[item] = item_data
            continue
        
        # Slice arrays
        if isinstance(item_data, list):
            item_data = np.array(item_data)
        time_restricted_data[item] = item_data[time_mask]

    return time_restricted_data

#%% Extract QinDenton data for the time period
def QinDenton_period(start_date, stop_date): 
    """
    Loads Qin-Denton OMNI data (solar wind/geomagnetic indices) for the specified range.

    Args:
        start_date (datetime): Start date.
        stop_date (datetime): Stop date.

    Returns:
        SpaceData: Structure containing 'Dst', 'Kp', 'ByIMF', 'BzIMF', 'P', etc.
    """
    
    print('Loading QinDenton Data...')
    QD_folder = "/home/wzt0020/sat_data_analysis/QinDenton/"
    QD_filenames = []

    # Generate list of daily files
    current_date_object = start_date
    while current_date_object <= stop_date:
        QD_year = os.path.join(QD_folder, str(current_date_object.year),'5min')
        QD_filenames.append(os.path.join(QD_year, f"QinDenton_{current_date_object.strftime("%Y%m%d")}_5min.txt"))
        current_date_object += dt.timedelta(days=1)
    
    global QD_data
    QD_data = dm.readJSONheadedASCII(QD_filenames)
    
    # Parse DateTime strings to objects
    datetime_format = "%Y-%m-%dT%H:%M:%S"
    QD_data['DateTime'] = [dt.datetime.strptime(t, datetime_format) for t in QD_data['DateTime']]

    return QD_data

#%% Set magnetic field model coefficients to closest time of QinDenton data
## Someday, replace this with Lgm_QinDenton in LANLGeoMag...
def QD_inform_MagInfo(time_dt, MagInfo):
    """
    Updates the Lgm_MagModelInfo C-structure with solar wind/Dst conditions 
    from the global QD_data for a specific time.

    Args:
        time_dt (datetime): Current time.
        MagInfo (Lgm_MagModelInfo): Pointer to the C-structure to update.
    """

    # Round down to nearest 5 minutes to match QD resolution
    minutes_to_subtract = time_dt.minute % 5
    rounded_dt = time_dt - dt.timedelta(
        minutes=minutes_to_subtract,
        seconds=time_dt.second,
        microseconds=time_dt.microsecond
    )

    # Find index (Assumes QD_data is global)
    time_index = int(np.where(np.array(QD_data['DateTime']) == rounded_dt)[0][0])

    # Update C-Structure Fields
    MagInfo.contents.By    = c_double(QD_data['ByIMF'][time_index])
    MagInfo.contents.Bz    = c_double(QD_data['BzIMF'][time_index])
    MagInfo.contents.P     = c_double(QD_data['Pdyn'][time_index])
    MagInfo.contents.G1    = c_double(QD_data['G'][time_index][0])
    MagInfo.contents.G2    = c_double(QD_data['G'][time_index][1])
    MagInfo.contents.G3    = c_double(QD_data['G'][time_index][2])
    MagInfo.contents.Kp    = c_int(round(QD_data['Kp'][time_index]))
    MagInfo.contents.fKp   = c_double(QD_data['Kp'][time_index])
    MagInfo.contents.Dst   = c_double(QD_data['Dst'][time_index])
    
    # Update W parameters (0-5)
    for i in range(6):
        MagInfo.contents.W[i] = c_double(QD_data['W'][time_index][i])
    
    return

#%% Convert TickTock to Lgm_DateTime
def ticktock_to_Lgm_DateTime(ticktock, c):
    """Helper to create Lgm_DateTime C-struct from Python Ticktock/Datetime."""
    dt_obj = ticktock
    lgm_dt = lgm_lib.Lgm_DateTime_Create(dt_obj.year, dt_obj.month, dt_obj.day, 
                                dt_obj.hour+dt_obj.minute/60+dt_obj.second/3600, lgm_lib.LGM_TIME_SYS_UTC, c)
    return lgm_dt 

#%% Find Loss Cone
def find_Loss_Cone(sat_data, height = 100, extMag='T89c'):
    """
    Calculates the magnetic loss cone angle and equatorial field strength by tracing field lines.
    
    Args:
        sat_data (dict): Satellite data with Position and Epoch.
        height (float): Altitude in km to define the atmosphere (footpoint).
        extMag (str): Magnetic field model ('T89c', 'TS04').

    Returns:
        tuple: (b_min, P_min, b_footpoint, loss_cone) arrays.
    """

    # Initialize LGM info once    
    MagInfo = lgm_lib.Lgm_InitMagInfo()
    IntMagModel = c_int(lgm_lib.__dict__[f"LGM_IGRF"])
    ExtMagModel = c_int(lgm_lib.__dict__[f"LGM_EXTMODEL_{extMag}"])
    lgm_lib.Lgm_Set_MagModel(IntMagModel, ExtMagModel, MagInfo)

    # Pre-allocate output arrays
    n_points = len(sat_data['Epoch'])
    b_min = np.zeros(n_points)
    P_min = np.zeros((n_points, 3))
    b_footpoint = np.zeros(n_points)
        
    # Loop through epochs (Tracing is inherently iterative in this library)    
    for i_epoch, epoch in enumerate(sat_data['Epoch'].UTC):
        # Setup Time and Coords
        current_time = ticktock_to_Lgm_DateTime(epoch, MagInfo.contents.c)
        lgm_lib.Lgm_Set_Coord_Transforms(current_time.contents.Date, current_time.contents.Time, MagInfo.contents.c)
        
        # Setup Vectors
        current_vec = Lgm_Vector.Lgm_Vector(*sat_data['Position'][i_epoch].data[0])
        south_vec = Lgm_Vector.Lgm_Vector()
        north_vec = Lgm_Vector.Lgm_Vector()
        minB_vec = Lgm_Vector.Lgm_Vector()
        QD_inform_MagInfo(epoch, MagInfo)
                
        # Trace Field Line
        # Lgm_Trace populates minB_vec (equator) and footpoints (north/south)
        lgm_lib.Lgm_Trace(pointer(current_vec), pointer(south_vec), pointer(north_vec), pointer(minB_vec),
                            height, 0.01, 1e-7, MagInfo)
                
        # Extract Results
        b_min[i_epoch] = MagInfo.contents.Bmin * 1e-5 # nT -> Gauss
        P_min[i_epoch,:] = [MagInfo.contents.Pmin.x, MagInfo.contents.Pmin.y, MagInfo.contents.Pmin.z]

        # Determine Footpoint B based on Hemisphere
        if sat_data['Position'][i_epoch].z >= 0:
            b_footpoint[i_epoch] = MagInfo.contents.Ellipsoid_Footprint_Bn * 1e-5
        else:
            b_footpoint[i_epoch] = MagInfo.contents.Ellipsoid_Footprint_Bs * 1e-5

    # Calculate Loss Cone: alpha_LC = arcsin(sqrt(B_eq / B_foot))
    loss_cone = np.rad2deg(np.arcsin(np.sqrt(b_min/b_footpoint)))

    # Clean up memory
    lgm_lib.Lgm_FreeMagInfo(MagInfo)

    return b_min, P_min, b_footpoint, loss_cone

#%% Find local pitch angle
def find_local90PA(sat_data):
    """
    Calculates the Local Pitch Angle that corresponds to a 90-degree Equatorial Pitch Angle.
    Based on the First Adiabatic Invariant (Conservation of Magnetic Moment).

    Args:
        sat_data (dict): Must contain 'b_satellite' and 'b_min' (or 'b_equator').

    Returns:
        numpy.ndarray: Array of local pitch angles in degrees.
    """

    local90PA = {}
    Beq = sat_data['b_min']
    Bsat = sat_data['b_satellite']

    # Only calculate where data is valid (B > 0)
    mask = (Beq > 0) & (Bsat > 0)
    local90PA = np.full_like(Beq, np.nan)

    # Conservation of 1st Adiabatic Invariant: sin^2(alpha_loc)/B_loc = sin^2(alpha_eq)/B_eq
    # Solve for alpha_eq given alpha_loc = 90 degrees:
    # sin(alpha_eq) = sqrt(B_eq / B_loc)
    local90PA[mask] = np.rad2deg(np.arcsin(np.sqrt(Beq[mask] / Bsat[mask])))
    return local90PA

#%% Find pitch angle corresponding to set K
def AlphaOfK(sat_data, K_set, extMag = 'T89c'):
    """
    Calculates the Equatorial Pitch Angle (alpha) required to maintain a constant 
    Second Adiabatic Invariant (K) for each epoch.

    Args:
        sat_data (dict): Satellite data with Epoch and Position.
        K_set (array): List of K values (Re G^0.5).

    Returns:
        DataFrame: Index=Epoch, Columns=K values. Contains Alpha (degrees).
    """
    
    K_set = np.atleast_1d(K_set)

    # Initialize LGM structures
    MagInfo = lgm_lib.Lgm_InitMagInfo()
    IntMagModel = c_int(lgm_lib.__dict__[f"LGM_IGRF"])
    ExtMagModel = c_int(lgm_lib.__dict__[f"LGM_EXTMODEL_{extMag}"])
    lgm_lib.Lgm_Set_MagModel(IntMagModel, ExtMagModel, MagInfo)

    alphaofK = np.full((len(sat_data['Epoch']), len(K_set)), np.nan)
    print()
    for i_epoch, epoch in enumerate(sat_data['Epoch'].UTC):
        print(f"    Time Index: {i_epoch+1}/{len(sat_data['Epoch'])}", end='\r')
        
        # Update Time and Coords in C-struct
        current_time = ticktock_to_Lgm_DateTime(epoch, MagInfo.contents.c)
        lgm_lib.Lgm_Set_Coord_Transforms(current_time.contents.Date, current_time.contents.Time, MagInfo.contents.c)
        QD_inform_MagInfo(epoch, MagInfo)
        
        current_vec = Lgm_Vector.Lgm_Vector(*sat_data['Position'][i_epoch].data[0])

        if extMag == 'TS07': # Work in Progress... I need the coefficient files for this
            lgm_lib.Lgm_SetCoeffs_TS07(current_time.contents.Date, current_time.contents.Time, pointer(MagInfo.contents.TS07_Info))
        
        # Calculate Alpha for each K
        for i_K, K in enumerate(K_set): 
            # Setup returns validity flag   
            setup_val = lgm_lib.Lgm_Setup_AlphaOfK(current_time, current_vec, MagInfo)
            if setup_val != -5:
                # Lgm_AlphaOfK returns alpha in degrees
                alphaofK[i_epoch,i_K] = lgm_lib.Lgm_AlphaOfK(K, MagInfo)
            else:
                alphaofK[i_epoch,i_K] = np.nan

            lgm_lib.Lgm_TearDown_AlphaOfK(MagInfo)
    
    epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
    alphaofK = pd.DataFrame(alphaofK, index=epoch_str, columns=K_set)
    
    # Clean up memory
    lgm_lib.Lgm_FreeMagInfo(MagInfo)

    return alphaofK

#%% Calculate Mu from energy channels and set alpha:
def MuofEnergyAlpha(gps_data, alphaofK):
    """
    Calculates the First Adiabatic Invariant (Mu) for each energy channel 
    given the calculated equatorial pitch angles (alphaofK).

    Formula: Mu = E_perp / B_eq = (E^2 + 2*E*E0) * sin^2(alpha_eq) / (2*E0*B_eq)

    Returns:
        tuple: (muofenergyalpha dict, Mu_bounds dict)
    """
    
    print('Calculating Mus for Energy Channels and Alphas...')
    muofenergyalpha = {}
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
        
        for i_K, K in enumerate(K_set):
            # Extract alpha values (N_epochs, 1)
            if isinstance(alphaofK[satellite], pd.DataFrame):
                alpha_vals = np.radians(alphaofK[satellite].values[:,i_K])
            else:
                alpha_vals = np.radians(alphaofK[satellite][:,i_K])
            
            # Vectorized Calculation: (N_epochs, 1) broadcast against (N_channels,) requires reshaping
            alpha_rad = np.radians(alpha_vals)[:, np.newaxis] # Shape (N, 1)
            sin_sq_alpha = np.sin(alpha_rad)**2
            
            b_eq = sat_data['b_equator'][:, np.newaxis] # Shape (N, 1)
            energies = sat_data['Energy_Channels'][np.newaxis, :] # Shape (1, M)
            
            # Result Shape (N, M)
            mu_matrix = (energies**2 + 2*energies*E0) * sin_sq_alpha / (2*E0*b_eq)
            
            # Store
            muofenergyalpha[satellite]['MuofEnergyAlpha'][K] = pd.DataFrame(mu_matrix, index=epoch_str, columns=sat_data['Energy_Channels'])

            # Track min/max for bounds
            curr_min = np.nanmin(mu_matrix)
            curr_max = np.nanmax(mu_matrix)
            Mu_bounds[satellite][K] = np.array([curr_min, curr_max])
            
            if curr_min < Mu_min: Mu_min = curr_min
            if curr_max > Mu_max: Mu_max = curr_max

            Mu_bounds[satellite][K] = np.zeros(2)
            Mu_bounds[satellite][K][0] = np.min(muofenergyalpha[satellite]['MuofEnergyAlpha'][K][:,0])
            if Mu_bounds[satellite][K][0] < Mu_min:
                Mu_min = Mu_bounds[satellite][K][0]
            Mu_bounds[satellite][K][1] = np.max(muofenergyalpha[satellite]['MuofEnergyAlpha'][K][:,-1])
            if Mu_bounds[satellite][K][1] > Mu_max:
                Mu_max = Mu_bounds[satellite][K][1]
            muofenergyalpha[satellite]['MuofEnergyAlpha'][K] = pd.DataFrame(muofenergyalpha[satellite]['MuofEnergyAlpha'][K], index=epoch_str, columns=sat_data['Energy_Channels'])
    
    Mu_bounds['Total'] = np.array((Mu_min, Mu_max))
    
    # Calculate nice rounded bounds for display/binning
    magnitude_min = 10**math.floor(math.log10(Mu_min))
    Mu_min_round = math.floor(Mu_min / (magnitude_min/10)) * (magnitude_min/10)

    magnitude_max = 10**math.floor(math.log10(Mu_max))
    Mu_max_round = math.ceil(Mu_max / (magnitude_max/10)) * (magnitude_max/10)

    Mu_bounds['Rounded'] = np.array((Mu_min_round, Mu_max_round))

    return muofenergyalpha, Mu_bounds

#%% Calculate energy from set mu and alpha:
def EnergyofMuAlpha(sat_data, Mu_set, alphaofK):
    """
    Calculates the Energy required to maintain a constant Mu 
    at a specific equatorial pitch angle (alpha).
    Inverts the Mu formula to solve for Energy.

    Returns:
        dict: Keys are K values, Values are DataFrames of Energy(time, Mu).
    """
    
    # Convert to a NumPy array if it's a single value
    Mu_set = np.atleast_1d(Mu_set)
    energyofmualpha = {}

    epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
    K_set = np.array(alphaofK.columns.tolist(), dtype=float)
    
    for i_K, K in enumerate(K_set):
        # Extract alpha (N_epochs,)
        if isinstance(alphaofK, pd.DataFrame):
            alpha_rad = np.radians(alphaofK.values[:,i_K])
        else:
            alpha_rad = np.radians(alphaofK[:,i_K])

        # Calculate sin^2(alpha)
        sin_sq_alpha = np.sin(alpha_rad)**2

        # Prepare arrays for broadcasting
        # sin_sq_alpha: (N,) -> (N, 1)
        # b_min: (N,) -> (N, 1)
        # Mu_set: (M,) -> (1, M)
        sin_sq_col = sin_sq_alpha[:, np.newaxis]
        b_min_col = sat_data['b_min'][:, np.newaxis]
        mu_row = Mu_set[np.newaxis, :]

        # Formula: E = sqrt(2*E0*Mu*B_eq / sin^2(alpha) + E0^2) - E0
        energy_matrix = np.sqrt(2 * E0 * mu_row * b_min_col / sin_sq_col + E0**2) - E0
        
        energyofmualpha[K] = pd.DataFrame(energy_matrix, index=epoch_str, columns=Mu_set)

    return energyofmualpha

#%% Transform from flux to PSD
def find_psd(j_CXD, energyofMuAlpha):
    """
    Converts Differential Flux (j) to Phase Space Density (f).
    PSD = j / p^2
    """
    psd = {}

    for K_val, K_data in j_CXD.items():
        Mu_set = np.array(K_data.columns.tolist(), dtype=float)
        epoch_list = K_data[Mu_set[0]].index.tolist()

        psd_matrix = np.full(K_data.shape, np.nan)
        energy_data = energyofMuAlpha[K_val].values # Energies in MeV
        flux_data = K_data.values

        # E_rel = E^2 + 2*E*E0 (Relativistic energy term for momentum)
        # p^2 ~ E_rel (ignoring constants for now, scaling handled below)
        E_rel = energy_data**2 + 2 * energy_data * E0

                # Calculate only where data is valid
        mask = (flux_data > 0) & (~np.isnan(flux_data))
        
        # Constants conversion:
        # 1e-3: flux units (cm^-2) to (m^-2)
        # Standard PSD = j / p^2. 
        # Here: flux_data / E_rel * scaling_factors
        psd_matrix[mask] = (flux_data[mask] / E_rel[mask]) * 1.66e-10 * 1e-3 * 200.3
        
        psd[K_val] = pd.DataFrame(psd_matrix, index=epoch_list, columns=Mu_set)

    return psd

#%% Calculate McIlain L
def find_McIlwain_L(sat_data, intMag = 'IGRF', extMag = 'T89c'):
    """
    Calculates the McIlwain L-parameter using LANLGeoMag.
    """
    MagInfo = lgm_lib.Lgm_InitMagInfo()
    IntMagModel = c_int(lgm_lib.__dict__[f"LGM_IGRF"])
    ExtMagModel = c_int(lgm_lib.__dict__[f"LGM_EXTMODEL_{extMag}"])
    lgm_lib.Lgm_Set_MagModel(IntMagModel, ExtMagModel, MagInfo)
    
    n_epochs = sat_data['Epoch'].UTC.shape
    l_shell_arr = np.zeros((n_epochs))
    
    # C-pointers for output (reused in loop)
    I = c_double(0)
    Bm = c_double(0)
    M = c_double(0)

    print()
    for i_epoch, epoch in enumerate(sat_data['Epoch'].UTC):
        print(f"    Time Index: {i_epoch+1}/{len(sat_data['Epoch'])}", end='\r')

        # Update Time/Coordinate Transformations in C-struct
        current_time = ticktock_to_Lgm_DateTime(epoch, MagInfo.contents.c)
        lgm_lib.Lgm_Set_Coord_Transforms(current_time.contents.Date, current_time.contents.Time, MagInfo.contents.c)
        QD_inform_MagInfo(epoch, MagInfo)

        # Get Satellite Position (GSM)
        current_vec = Lgm_Vector.Lgm_Vector(*sat_data['Position'][i_epoch].data[0])   # for equatorial PA

        # Calculate L for each K (pitch angle column) at this epoch   
        l_shell_arr[i_epoch] = lgm_lib.Lgm_McIlwain_L(
            current_time.contents.Date, current_time.contents.Time, 
            current_vec, 90, 0, 
            pointer(I), pointer(Bm), pointer(M), MagInfo)

    extMag_label = 'T89' if extMag == 'T89c' else extMag                                          
    sat_data[f'L_LGM_{extMag_label}IGRF'] = l_shell_arr
    lgm_lib.Lgm_FreeMagInfo(MagInfo)

    return sat_data

#%% Calculate L_star
def find_Lstar(sat_data, alphaofK, intMag = 'IGRF', extMag = 'T89c'):
    """
    Calculates L* (Roederer L) using LANLGeoMag tracing.
    """
    # Initialize Lstar Info Structure
    LstarInfo = lgm_lib.InitLstarInfo(0)
    IntMagModel = c_int(lgm_lib.__dict__[f"LGM_IGRF"])
    ExtMagModel = c_int(lgm_lib.__dict__[f"LGM_EXTMODEL_{extMag}"])
    lgm_lib.Lgm_Set_MagModel(IntMagModel, ExtMagModel, LstarInfo.contents.mInfo)
    # variables are: tolerance, N Field lines, LstarInfo
    lgm_lib.Lgm_SetLstarTolerances(0, 10, LstarInfo) #3, 24 for decent quality, 0, 10 for quick
    
    # Extract K values (columns) from the alphaofK DataFrame
    K_set = np.array(list(alphaofK.columns.tolist()), dtype=float)
    # Initialize Lstar array with same shape as the pitch angle input (N_epochs x N_K)
    sat_data['Lstar'] = np.zeros_like(alphaofK.values)

    print()
    for i_epoch, epoch in enumerate(sat_data['Epoch'].UTC):
        print(f"    Time Index: {i_epoch+1}/{len(sat_data['Epoch'])}", end='\r')
        # Could possibly speed up with NewTimeLstarInfo
        
        # Update Time/Coordinate Transformations in C-struct
        current_time = ticktock_to_Lgm_DateTime(epoch, LstarInfo.contents.mInfo.contents.c)
        lgm_lib.Lgm_Set_Coord_Transforms(current_time.contents.Date, current_time.contents.Time, LstarInfo.contents.mInfo.contents.c)
        QD_inform_MagInfo(epoch, LstarInfo.contents.mInfo)

        # NOTE: Lstar uses local pitch angle as the input, so we can either
        #       1) Use satellite location and convert to local pitch angle
        #       2) Use Pmin and equatorial pitch angle output from AlphaOfK      
        current_vec = Lgm_Vector.Lgm_Vector(*sat_data['P_min'][i_epoch,:])   # for equatorial PA
        #current_vec = Lgm_Vector.Lgm_Vector(*sat_data['Position'].data[i_epoch]) # for local PA

        # b_local = sat_data['b_satellite'][i_epoch]
        # b_min = sat_data['b_min'][i_epoch]

        for i_K, K in enumerate(K_set):
            eq_pitch_angle = alphaofK.values[i_epoch,i_K] # this is equitorial pitch angle
            #local_pitch_angle = np.rad2deg(np.arcsin(np.sqrt(np.sin(np.deg2rad(eq_pitch_angle))**2*b_local/b_min)))
            
            # Set Pitch Angle in C-Struct (required for mirror point determination)
            LstarInfo.contents.PitchAngle = c_double(eq_pitch_angle)
            
            # Compute L* using the C library function
            # This traces the drift shell and integrates magnetic flux.
            lgm_lib.Lstar(pointer(current_vec), LstarInfo)
            
            # Extract result (LS property of the structure)
            sat_data['Lstar'][i_epoch, i_K] = LstarInfo.contents.LS

            #lgm_lib.FreeLstarInfo(LstarInfo)
    return sat_data