#%% Initialize
import os
import glob
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import datetime as dt
import spacepy.datamodel as dm
import spacepy.time as spt
from spacepy.time import Ticktock
from spacepy.coordinates import Coords
import scipy
import scipy.constants as sc
import pandas as pd

from all_PSD_func import (find_Loss_Cone, find_local90PA)

# Physical Constants
# Rest mass energy of an electron in MeV (m_0 * c^2)
global E0
E0 = sc.electron_mass * sc.c**2 / (sc.electron_volt * 1e6)

#%% Import GPS data function
def import_GPS(input_folder):
    """
    Loads and concatenates GPS data files (NS*.ascii) for all satellites in a directory tree.

    Args:
        input_folder (str): Root directory containing satellite subfolders (e.g., 'ns60').

    Returns:
        dict: Keys are satellite names, values are SpaceData objects with sorted time series data.
    """
    loaded_data = {} # Initialize an empty dictionary to store loaded data.
    print(f"Starting to process files in: {input_folder}")

    # Use os.walk to traverse the directory tree.
    # 'root' is the current directory path (e.g., "/home/wzt0020/GPS_data/april2017storm/").
    # 'dirnames' is a list of subdirectories in the current 'root' (e.g., ['ns60', 'ns63']).
    # '_' (underscore) is used as a throwaway variable for 'filenames' as it's not used directly here.
    for (root, satnames, _) in os.walk(input_folder):
        # Sort satellite names numerically (ns54 -> ns60) for consistent processing order
        sorted_satnames = sorted(satnames, key=lambda s: int(s[2:]))
        
        # Iterate over each satellite subdirectory name found in the current 'root'.
        for satname in sorted_satnames:
            # Construct the full path to the current satellite's directory.
            sat_dir_path = os.path.join(root, satname)
            print(f"    Reading in satellite {satname}", end='\r')
            
            # Use glob.glob to find all files matching "ns*.ascii" pattern directly within the current satellite's directory.
            sat_filenames = glob.glob(sat_dir_path + "/ns*ascii")
            
            # Sort the collected filenames by their date (YYMMDD) component.
            sorted_sat_filenames = sorted(sat_filenames, 
                key=lambda filepath: os.path.basename(filepath).split('_v')[0].split('_')[-1])
            
            if not sorted_sat_filenames:
                continue
            
            # Efficiently read all files at once into a single SpaceData object
            loaded_data[satname] = dm.readJSONheadedASCII(sorted_sat_filenames)

    print("Data Loaded \n")    
    return loaded_data

#%% Convert Time for GPS satellites
def convert_time(sat_data):
    """
    Converts GPS Year/Decimal Day arrays into SpacePy Ticktock objects (UTC).
    Handles GPS-to-UTC offset (leap seconds).

    Args:
        sat_data (dict): Dictionary with 'year' and 'decimal_day' arrays.

    Returns:
        dict: Updated sat_data with 'Epoch' key (Ticktock object).
    """
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
    sat_data['Epoch'] = Ticktock(gpsseconds, dtype='GPS')
    return sat_data

#%% Extract relevant information from time processed data
def data_from_gps(time_restricted_data, Lshell = [], intMag = 'IGRF', extMag = 'T89c'):
    """
    Extracts and structures specific GPS data variables for further analysis.
    Filters based on L-shell and electron flux quality flags.
    """
    
    gps_data_out = {}
    extMag_label = 'T89' if extMag == 'T89c' else extMag   
    model_var = f"L_LGM_{extMag_label}{intMag}"

    chosen_vars = ['Epoch', 'local_time',
                   'b_satellite','b_equator',
                   'L_LGM_T89IGRF', 'L_LGM_TS04IGRF',
                   'electron_diff_flux_energy','electron_diff_flux', 'efitpars']
    
    for satellite, sat_data in time_restricted_data.items():
        print(f"    Processing Data for satellite {satellite}", end='\r')
        gps_data_out[satellite] = {}

        # L-Shell Filter
        if isinstance(Lshell, (int, float)):
            Lmask = sat_data[model_var] <= Lshell
        elif not Lshell:
            Lmask = np.full(sat_data[model_var].shape, True, dtype=bool)
        else:
            print("Error: Lshell must be a scalar")
            continue

        # Data Quality Filter (Fitting residuals and bad data flags)
        # Filters out epochs where the spectral fit was poor or flux is -1
        efit_mask = ((np.max(np.log10(time_restricted_data[satellite]['model_counts_electron_fit'][:,0:5] / 
                                      time_restricted_data[satellite]['electron_diff_flux'][:,0:5]),axis=1) <= 0.11) 
                        | (np.sum(time_restricted_data[satellite]['electron_diff_flux'][:,0:5]==-1,axis=1)==0))
        
        mask = Lmask & efit_mask

        # Process Position (GEO -> GSM)
        R = sat_data['Rad_Re'][mask]
        Lat = sat_data['Geographic_Latitude'][mask]
        Lon = sat_data['Geographic_Longitude'][mask]

        # SpacePy Coords requires array of [R, Lat, Lon]
        position_init = Coords(np.column_stack((R,Lat,Lon)),'GEO','sph')
        position_init.ticks = sat_data['Epoch'][mask]
        gps_data_out[satellite]['Position'] = position_init.convert('GSM','car')

        # Extract other variables
        for var_name in chosen_vars:
            if var_name == 'local_time':
                gps_data_out[satellite]['MLT'] = time_restricted_data[satellite][var_name][mask]
            elif var_name == 'electron_diff_flux_energy':
                # Energy channels are constant, take first row
                gps_data_out[satellite]['Energy_Channels'] = time_restricted_data[satellite][var_name][0]
            else:
                gps_data_out[satellite][var_name] = time_restricted_data[satellite][var_name][mask]

        # Calculate Derived Magnetic Properties
        (gps_data_out[satellite]['b_min'], 
         gps_data_out[satellite]['P_min'], 
         gps_data_out[satellite]['b_footpoint'], 
         gps_data_out[satellite]['loss_cone']) = find_Loss_Cone(gps_data_out[satellite], extMag=extMag)
        gps_data_out[satellite]['local90PA'] = find_local90PA(gps_data_out[satellite])
    
    return gps_data_out

#%% Calculate Energy Spectra
def reletavistic_Maxwellian(energies, n, T): # Based on Maxwell-Juttner distribution from gps data readme
    """Calculates relativistic Maxwell-Juttner flux distribution."""
    # c in cm/s for flux units
    c_cms = sc.c * 10**2 
    # Relativistic momentum p in MeV/c
    p = np.sqrt((energies + E0)**2 - E0**2) / sc.c 
    # Bessel function K2
    K2 = scipy.special.kn(2, E0/T) # modified Bessel function of the second kind

    j_MJ = n * c_cms /(4*np.pi*T*K2*np.exp(E0/T)) * p**2*sc.c**2/E0**2 * np.exp(-energies/T)
    return j_MJ

def Gaussian(energies, n, mu, sigma):
    """Calculates Gaussian flux distribution in log-momentum space."""
    p = np.sqrt((energies + E0)**2 - E0**2) / sc.c # reletavistic momentum in MeV/c
    j_G = n * np.exp(-np.log(p*sc.c/mu)**2/(2*sigma**2))
    return j_G

def energy_spectra(sat_data, energy_data):
    """
    Calculates Phase Space Density (or Flux) based on the CXD fitted parameters 
    (3 Maxwellians + 1 Gaussian) for specified energies.

    Args:
        sat_data (dict): Contains 'efitpars' (fit coefficients).
        energy_data (dict): Nested dict {K: {Mu: Series_of_Energies}}.

    Returns:
        dict: Calculated flux/PSD values matching the structure of energy_data.
    """
    j_CXD = {}
    echannel_min = sat_data['Energy_Channels'][0]
    echannel_max = sat_data['Energy_Channels'][-1]
    
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

    for K_val, K_data in energy_data.items():
        Mu_set = np.array(list(K_data.keys()), dtype=float)
        # Assuming all Mu entries share the same index length
        if len(Mu_set) == 0: continue
        
        # Get index from the first Mu entry
        epoch_list = K_data[Mu_set[0]].index.tolist()
        
        # Pre-allocate result matrix (N_epochs, N_Mu)
        j_result = np.zeros((len(epoch_list), len(Mu_set)))

        for i_Mu, (Mu_val, Mu_series) in enumerate(K_data.items()):
            energies = Mu_series.values

            # Mask validity: Energy must be within instrument range
            energy_mask = (energies >= echannel_min) & (energies <= echannel_max)
            
            # Calculate spectral components
            j_MJ1 = reletavistic_Maxwellian(energies, n1, T1)
            j_MJ2 = reletavistic_Maxwellian(energies, n2, T2)
            j_MJ3 = reletavistic_Maxwellian(energies, n3, T3)
            j_G   = Gaussian(energies, nG, muG, sigma)

            # Sum components
            total_flux = j_MJ1 + j_MJ2 + j_MJ3 + j_G

            # Assign masked values
            # Using boolean indexing on 1D arrays
            j_result[energy_mask, i_Mu] = total_flux[energy_mask]
            # Outside mask remains 0 (or could initialize to NaN)

        j_CXD[K_val] = pd.DataFrame(j_result, index=epoch_list, columns=Mu_set) 
        
    return j_CXD