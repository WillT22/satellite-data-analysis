#%% Initialize
import re
import numpy as np
import pandas as pd
import datetime as dt
from scipy.special import legendre

import importlib
import GPS_PSD_func
importlib.reload(GPS_PSD_func)
from GPS_PSD_func import find_local90PA, find_Loss_Cone

#%%
global Zhao_median_filepath
Zhao_median_filepath = '/home/wzt0020/Zhao_2018_model_files/PAD_model_coeff_median.txt'

#%% Extract coefficients from Zhao_2018
# NOTE: MLT and L are midpoints of the bin!
def import_Zhao_coeffs():
    print("Importing Zhao Coefficients... ")
    global Zhao_coeffs
    Zhao_coeffs = {}

    # Regex patterns for different header lines
    energy_pattern = re.compile(r"E=(\d+\.?\d*|\.\d+)\s*(keV|MeV)")
    dst_gt_pattern = re.compile(r"Dst\s*([<>])\s*(-?\d+)\s*nT")
    dst_range_pattern = re.compile(r"(-?\d+)\s*nT\s*<\s*Dst\s*<\s*(-?\d+)\s*nT")
    coeff_pattern = re.compile(r"c(\d+)") # Captures the number after 'c'
    mlt_l_header_pattern = re.compile(r"MLT\\L,\s*(.*)") # Captures everything after "MLT\L,"

    f = open(Zhao_median_filepath, 'r')

    for line_num, line in enumerate(f, 1):
        stripped_line = line.strip()

        if not stripped_line:
            continue

        # --- Parse Energy Line ---
        match_energy = energy_pattern.match(stripped_line)
        if match_energy:
            current_energy_str = float(match_energy.group(1))
            unit_str = match_energy.group(2)
            if unit_str == 'keV':
                current_energy = current_energy_str/1000
            elif unit_str == 'MeV':
                current_energy = current_energy_str

            Zhao_coeffs[current_energy] = {}
            current_dst_range = None # Reset Dst when new Energy starts
            current_coeff_block = None # Reset Coeff when new Energy starts
            #print(f"Loading Energy: {current_energy_str} {unit_str}")
            continue

        # --- Parse Dst Range Line ---
        match_dst_gt = dst_gt_pattern.match(stripped_line)
        match_dst_range = dst_range_pattern.match(stripped_line)

        if match_dst_gt:
            dst_sign = match_dst_gt.group(1)
            dst_val = int(match_dst_gt.group(2))
            current_dst_range = f"Dst {dst_sign} {dst_val} nT"
            Zhao_coeffs[current_energy][current_dst_range] = {}
            current_coeff_block = None
            #print(f"    Found Dst Range: {current_dst_range}")
            continue
        elif match_dst_range:
            dst_low = int(match_dst_range.group(1))
            dst_high = int(match_dst_range.group(2))
            current_dst_range = f"{dst_low} nT < Dst < {dst_high} nT"
            Zhao_coeffs[current_energy][current_dst_range] = {}
            current_coeff_block = None
            #print(f"    Found Dst Range: {current_dst_range}")
            continue
                    
        # Ensure we are in a valid energy and Dst context
        if current_energy is None or current_dst_range is None:
            print(f"Warning: Skipping line {line_num} ('{stripped_line}') - No active Energy or Dst context.")
            continue

        # --- Parse Coefficient Line ---
        match_coeff = coeff_pattern.match(stripped_line)
        if match_coeff:
            current_coeff_block = f"c{match_coeff.group(1)}"
            Zhao_coeffs[current_energy][current_dst_range][current_coeff_block] = {}
            current_l_values = None
            current_mlt_values = []
            current_data_rows = []
            #print(f"        Found Coefficient Block: {current_coeff_block}")
            continue

        # Ensure we are in a valid coefficient context
        if current_coeff_block is None:
            print(f"Warning: Skipping line {line_num} ('{stripped_line}') - No active Coefficient context.")
            continue

        # --- Parse MLT\L Header Line ---
        # NOTE: for energies < 1MeV, L=1-6, for energies > 1MeV, L=3-6
        match_mlt_l_header = mlt_l_header_pattern.match(stripped_line)
        if match_mlt_l_header:
            l_values_str = match_mlt_l_header.group(1).replace(',', ' ').split()
            try:
                current_l_values = np.array([float(val) for val in l_values_str])
                Zhao_coeffs[current_energy][current_dst_range][current_coeff_block]['L_values'] = current_l_values
            except ValueError:
                print(f"Warning: Skipping line {line_num} ('{stripped_line}') - Malformed L_values header.")
                current_l_values = None # Invalidate L_values for this block
            continue

        # Ensure L_values header has been parsed for data rows
        if current_l_values is None:
            print(f"Warning: Skipping line {line_num} ('{stripped_line}') - No L_values header found for data.")
            continue

        # --- Parse Data Rows ---
        # Data rows start with MLT, then data values
        parts = stripped_line.replace(',', ' ').split() # Replace commas with spaces for splitting
        if len(parts) > 0: # Ensure line is not just empty after stripping
            try:
                mlt_val = float(parts[0])
                data_row_values = np.array([float(val) for val in parts[1:]])
                # Replace fill values with zeroes
                data_row_values[data_row_values == 1e31] = 0
                            
                # Data row must match the number of L_values
                if len(data_row_values) == len(current_l_values):
                    current_mlt_values.append(mlt_val)
                    current_data_rows.append(data_row_values)
                else:
                    print(f"Warning: Skipping line {line_num} ('{stripped_line}') - Data columns mismatch L_values count.")
            except ValueError:
                print(f"Warning: Skipping line {line_num} ('{stripped_line}') - Malformed data row.")
        else:
            print(f"Warning: Skipping line {line_num} ('{stripped_line}') - Unexpected empty line in data block.")

        # After loop, store the collected data for the last coefficient block
        # This handles the case where the file ends immediately after a data block.
        if current_energy is not None and current_dst_range is not None and \
            current_coeff_block is not None and current_mlt_values:
            # Create Pandas DataFrame with MLT as index and L as columns
            data_df = pd.DataFrame(current_data_rows, index=current_mlt_values, columns=current_l_values)
            Zhao_coeffs[current_energy][current_dst_range][current_coeff_block]['MLT_values'] = np.array(current_mlt_values)
            Zhao_coeffs[current_energy][current_dst_range][current_coeff_block]['data_matrix'] = data_df
    print("Zhao Coefficients Imported \n")
    return Zhao_coeffs

#%% Find PAD coefficients from Zhao 2018 model
def find_Zhao_PAD_coeffs(gps_data, QD_data, EnergyofMuAlpha):
    """
    Extracts Zhao coefficients for each epoch based on current conditions (Dst), set parameters (Energy), and satellite location (MLT, L)
    
    Args:
        gps_data (dict): Processed GPS data, typically containing 'Epoch', 'Energy_Channels',
                         'L_LGM_T89IGRF', and 'MLT' for each satellite.
        EnergyofMuAlpha (dict): Dictionary containing 'Mu_set' (Mu values) and
                                'EnergyofMuAlpha' (matrix of energy values for each Mu and Epoch).
                                This 'EnergyofMuAlpha' is assumed to be a 2D array: (len(Mu_set), len(Epoch)).

    Returns:
        dict: Zhao_epoch_coeffs - A nested dictionary with calculated coefficients.
              Structure: {satellite: {K_value: {Mu_value: pd.DataFrame}}
              Each DataFrame has Legendre coefficient numberas index and epoch strings as columns.                         
    """
    print("Extracting Zhao Coefficients for each Epoch...")
    Zhao_epoch_coeffs = {}
    # Define the full list of possible coefficient names
    full_coeff_list = ['c2','c4','c6','c8','c10']
    # Extract energy bins from the global Zhao_coeffs dictionary to serve as keys in the Zhao_epoch_coeff DataFrame
    energy_bins = np.array(list(Zhao_coeffs.keys()), dtype=float)
    # --- Outer Loop: Iterate through each satellite in the GPS data ---
    for satellite, sat_data in gps_data.items():
        print(f"    Extracting Coefficients for satellite {satellite}", end='\r')
        Zhao_epoch_coeffs[satellite] = {}
        # Get the matrix of energy values for each Mu and Epoch for the current satellite.
        sat_energyofmualpha = EnergyofMuAlpha[satellite]
        # Define the valid range for energy_value besed on instrument energy channels.
        echannel_min = sat_data['Energy_Channels'][0]
        echannel_max = sat_data['Energy_Channels'][-1]
        # Prepare a list of formatted epoch strings for the column headers of the final Pandas DataFrame for each Mu_value.
        epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
        
        # --- Loop 1: Iterate through each K value (K_val is the index, K_data is the energy data for that Mu) ---
        for K_val, K_data in sat_energyofmualpha.items():
            Zhao_epoch_coeffs[satellite][K_val] = {}
            K_data_values = K_data.values   
            # --- Loop 2: Iterate through each Mu value in Mu_set ---
            # Get the Mu_set array for this satellite for use as a key in Zhao_epoch_coeffs.
            Mu_set = np.array(list(EnergyofMuAlpha[satellite][K_val].columns.tolist()), dtype=float)
            for Mu_val in range(len(Mu_set)):
                Mu_value = Mu_set[Mu_val]
                # Initialize final coefficient matrix for each satellite, K, and Mu_set value
                Zhao_epoch_coeffs[satellite][K_val][Mu_value] = np.zeros((len(sat_data['Epoch']),5))
                
                # --- Loop 3: Iterate through each epoch (time step) for each satellite ---
                for i_epoch, epoch in enumerate(sat_data['Epoch']):
                    # Round down to the nearest 5 minutes
                    time_dt = epoch.UTC[0] # Transform from spacepy TickTock to datetime
                    minutes_to_subtract = time_dt.minute % 5
                    rounded_dt = time_dt - dt.timedelta(
                        minutes=minutes_to_subtract,
                        seconds=time_dt.second,
                        microseconds=time_dt.microsecond
                    )
                    # Find the index of this rounded time in the QD_data['DateTime'] array.
                    # Assumes QD_data is a global variable
                    time_index = int(np.where(QD_data['DateTime']==rounded_dt)[0])
                    
                    # Identify DST conditions during this epoch from QD_data.
                    epoch_dst = QD_data['Dst'][time_index]
                    if epoch_dst > -20:
                        i_dst = 'Dst > -20 nT'
                    elif epoch_dst <= -20 and epoch_dst > -50:
                        i_dst = '-50 nT < Dst < -20 nT'
                    elif epoch_dst <= -50:
                        i_dst = 'Dst < -50 nT'

                    # Get the L-shell and energy value for the current (Mu, Epoch) point.
                    Lshell = sat_data['L_LGM_T89IGRF'][i_epoch]
                    energy_value = K_data_values[i_epoch,Mu_val]
                    
                    # --- Primary Filter Condition ---
                    # Do NOT extrapolate outside of energy channel range or beyond 6.2 MeV!
                    if (energy_value >= echannel_min and energy_value <= echannel_max and energy_value <= 6.2 and Lshell <= 6):
                        # Find the closest energy bin in Zhao_coeffs for the current energy_value.
                        i_energy = np.argmin(np.abs(energy_value-energy_bins))
                        ebin_value = energy_bins[i_energy]
                        
                        # Find the MLT bin index (0-11 for 0-22 MLT in 2-hour bins).
                        # (MLT + 1) % 24 ensures 23-24 MLT maps to 0-1, then //2 for bin.
                        i_MLT = int(((sat_data['MLT'][i_epoch] + 1) % 24) // 2)

                        # For E < 1 MeV, m=10 at L<2, m=8 at 2<=L<4, and m=6 at 4<=L<=6
                        coeff_key_list = list(Zhao_coeffs[ebin_value][i_dst].keys())
                        coeff_data = Zhao_coeffs[ebin_value][i_dst]
                        
                        # --- Loop 4: Iterate through each expected Legendre coefficient ---
                        for i_c, coeff in enumerate(coeff_key_list):
                            coeff_data_temp = coeff_data[coeff]['data_matrix'].values
                            if ebin_value < 1: # For E < 1 MeV
                                # Find Lshell bin
                                i_L = int((Lshell - 0.9) // 0.2)
                                if Lshell < 2: # m=10 (all 5 coeffs)
                                        Zhao_epoch_coeffs[satellite][K_val][Mu_value][i_epoch,i_c] = coeff_data_temp[i_MLT,i_L]
                                elif Lshell >= 2 and Lshell < 4: # m=8 (c2,c4,c6,c8)
                                    if coeff != 'c10':
                                        Zhao_epoch_coeffs[satellite][K_val][Mu_value][i_epoch,i_c] = coeff_data_temp[i_MLT,i_L]
                                elif Lshell >= 4: # m=6 (c2,c4,c6)
                                    if coeff != 'c10' and coeff != 'c8':
                                        Zhao_epoch_coeffs[satellite][K_val][Mu_value][i_epoch,i_c] = coeff_data_temp[i_MLT,i_L]  
                            # For E >= 1 MeV, m=6 for 3<=L<=6
                            elif ebin_value >= 1: # For E >= 1 MeV
                                if Lshell >= 3: # m=6 (c2,c4,c6)
                                    # Find Lshell bin
                                    i_L = int((Lshell - 2.9) // 0.2)
                                    if coeff != 'c10' and coeff != 'c8':
                                        Zhao_epoch_coeffs[satellite][K_val][Mu_value][i_epoch,i_c] = coeff_data_temp[i_MLT,i_L]
                                    else:
                                        # Lshell < 3 for E >= 1 MeV is outside the defined range for non-zero coeffs.
                                        # Coefficients for these points remain 0.0 (from initialization).
                                        pass 
                # After processing all coefficients for a specific (satellite, Mu_value) combination and all epochs, 
                # convert the accumulated NumPy array into a Pandas DataFrame. 
                Zhao_epoch_coeffs[satellite][K_val][Mu_value] = pd.DataFrame(Zhao_epoch_coeffs[satellite][K_val][Mu_value], index=epoch_str, columns=full_coeff_list)         
    print("Zhao Coefficients Extracted \n")
    return Zhao_epoch_coeffs

#%% Create PAD from Zhao 2018 model coefficients
def define_Legendre(alpha):
    alpha_rad = np.deg2rad(alpha)
    P = np.array([legendre(2)(np.cos(alpha_rad)), legendre(4)(np.cos(alpha_rad)), legendre(6)(np.cos(alpha_rad)),
                  legendre(8)(np.cos(alpha_rad)), legendre(10)(np.cos(alpha_rad))])
    return P.transpose()

def create_PAD(gps_data, Zhao_epoch_coeffs, AlphaofK):
    print("Creating PAD Models...")
    alpha_init = np.linspace(0,180,361)
    PAD_models = {}
    for satellite, sat_data in Zhao_epoch_coeffs.items():
        print(f"    Modeling PAD for satellite {satellite}", end='\r')
        PAD_models[satellite] = {}
        K_set = np.array(list(sat_data.keys()), dtype=float)
        local90PA = gps_data[satellite]['local90PA']
        loss_cone = gps_data[satellite]['loss_cone']
        for K_val, K_data in sat_data.items():
            i_K = np.where(K_set == K_val)[0][0]
            PAD_models[satellite][K_val] = {}
            for Mu_val, Mu_data in K_data.items():
                coeff_data = Mu_data.values
                epoch_list = Mu_data.index.tolist()
                PAD_models[satellite][K_val][Mu_val] = {} #{{np.zeros((len(epoch_list),len(alpha_init)+6))}}
                PAD_models[satellite][K_val][Mu_val]['Model'] = np.zeros((len(epoch_list),len(alpha_init)+6))
                PAD_models[satellite][K_val][Mu_val]['pitch_angles'] = np.zeros((len(epoch_list),len(alpha_init)+6))
                for i_epoch, epoch in enumerate(epoch_list):
                    alpha_local90 = local90PA[i_epoch]
                    alpha_local90_add = np.array((alpha_local90, 180-alpha_local90))
                    alpha_loss_cone = loss_cone[i_epoch]
                    alpha_loss_cone_add = np.array((alpha_loss_cone, 180-alpha_loss_cone))
                    alphaofK = AlphaofK[satellite].values[i_epoch,i_K]
                    alphaofK_add = np.array((alphaofK, 180-alphaofK))
                    alpha_epoch = np.append(alpha_init, [alpha_local90_add,alphaofK_add,alpha_loss_cone_add])
                    alpha_epoch.sort()
                    PAD_models[satellite][K_val][Mu_val]['pitch_angles'][i_epoch,:] = alpha_epoch
                    P = define_Legendre(alpha_epoch)
                    PAD_models[satellite][K_val][Mu_val]['Model'][i_epoch,:] = np.sum(coeff_data[i_epoch,:] * P, axis=1) + 1
                PAD_models[satellite][K_val][Mu_val]['pitch_angles'] = pd.DataFrame(PAD_models[satellite][K_val][Mu_val]['pitch_angles'], index=epoch_list)
                PAD_models[satellite][K_val][Mu_val]['Model'] = pd.DataFrame(PAD_models[satellite][K_val][Mu_val]['Model'], index=epoch_list)
    print("PAD Models Completed")
    return PAD_models

#%% Find Integral of Normalized PAD between Loss Cone and Local 90 PA
def P0_int_eq(x,a):
    return -(2*np.sqrt(1 - a*x))/a
def P2_int_eq(x,a):
    return P0_int_eq(x,a) * (1 - 1/a - x/2)
def P4_int_eq(x,a):
    return P0_int_eq(x,a) * (1 + 7/(3*a**2) - 10/(3*a) - (5*x)/3 + (7*x)/(6*a) + (7*x**2)/8)
def P6_int_eq(x,a):
    return P0_int_eq(x,a) * (1 - 33/(5*a**3) + 63/(5*a**2) - 7/a - (7*x)/2 - (33*x)/(10*a**2) + (63*x)/(10*a) + (189*x**2)/40 - (99*x**2)/(40*a) - (33*x**3)/16)
def P8_int_eq(x,a):
    return P0_int_eq(x,a) * (1 + 143/(7*a**4) - 1716/(35*a**3) + 198/(5*a**2) - 12/a - 6*x + (143*x)/(14*a**3) - (858*x)/(35*a**2) + (99*x)/(5*a) + (297*x**2)/20 + (429*x**2)/(56*a**2) - (1287*x**2)/(70*a) - (429*x**3)/28 + (715*x**3)/(112*a) + (715*x**4)/128)
def P10_int_eq(x,a):
    return P0_int_eq(x,a) * (1 - 4199/(63*a**5) + 12155/(63*a**4) - 1430/(7*a**3) + 286/(3*a**2) - 55/(3*a) - (55*x)/6 - (4199*x)/(126*a**4) + (12155*x)/(126*a**3) - (715*x)/(7*a**2) + (143*x)/(3*a) + (143*x**2)/4 - (4199*x**2)/(168*a**3) + (12155*x**2)/(168*a**2) - (2145*x**2)/(28*a) - (3575*x**3)/56 - (20995*x**3)/(1008*a**2) + (60775*x**3)/(1008*a) + (60775*x**4)/1152 - (20995*x**4)/(1152*a) - (4199*x**5)/256)
def define_Legendre_Int_eq(b_sat, b_eq, b_fpt, b_min):
    low = b_min / b_fpt
    high = b_eq / b_sat
    P_int = np.array([
        P2_int_eq(high,b_sat/b_eq) - P2_int_eq(low,b_sat/b_eq),
        P4_int_eq(high,b_sat/b_eq) - P4_int_eq(low,b_sat/b_eq),
        P6_int_eq(high,b_sat/b_eq) - P6_int_eq(low,b_sat/b_eq),
        P8_int_eq(high,b_sat/b_eq) - P8_int_eq(low,b_sat/b_eq),
       P10_int_eq(high,b_sat/b_eq) - P10_int_eq(low,b_sat/b_eq)])
    return P_int.transpose()

def PAD_Scale_Factor(gps_data, Zhao_epoch_coeffs, alphaofK):
    print('Calculating Scale Factor...')
    PAD_scale_factor = {}
    for satellite, coeff_data in Zhao_epoch_coeffs.items():
        PAD_scale_factor[satellite] = {}
        K_set = np.array(list(coeff_data.keys()), dtype=float)
        
        sat_data = gps_data[satellite]
        
        # Inputs for PAD Integration between LC and local90
        b_satellite = sat_data['b_satellite']
        b_equator = sat_data['b_equator']
        b_footpoint = sat_data['b_footpoint']
        b_min = sat_data['b_min']

        # Inputs for PAD model generation (and logic restriction)
        loss_cone = sat_data['loss_cone']
        alphaofK_data = alphaofK[satellite].values
        
        for K_val, K_data in coeff_data.items():
            i_K = np.where(K_set == K_val)[0][0]
            Mu_set = np.array(list(K_data.keys()), dtype=float)
            epoch_list = K_data[Mu_set[0]].index.tolist()
            
            # Initialize arrays
            PAD_models = np.zeros((len(epoch_list),len(Mu_set)))
            PAD_integral = np.zeros((len(epoch_list),len(Mu_set)))
            PAD_scale_factor[satellite][K_val]= np.zeros((len(epoch_list),len(Mu_set)))
            
            for Mu_val, Mu_data in K_data.items():
                coeffs = Mu_data.values
                
                # Only calculate values where any coefficients are nonzero
                coeff_mask = np.sum(coeffs,axis=1) != 0
                # Only calculate if the desired angle is larger than loss cone
                alpha_mask = (alphaofK_data[:,i_K] > loss_cone)
                # Combine masks
                valid_mask = coeff_mask & alpha_mask
                i_Mu = np.where(Mu_set == Mu_val)[0][0]
                
                # Find PAD Model value for Mu_set values
                P = define_Legendre(alphaofK_data[:,i_K])
                PAD_models[:,i_Mu] = np.sum(coeffs * P, axis=1) + 1
                
                # Find integral between loss cone and local 90 degree pitch angle
                P_int = define_Legendre_Int_eq(b_satellite, b_equator, b_footpoint, b_min)
                PAD_integral[:,i_Mu] = np.array(1/2 * b_satellite/b_equator * 
                                        (np.sum(coeffs * P_int, axis=1) + (P0_int_eq(b_equator/b_satellite,b_satellite/b_equator)-P0_int_eq(b_min/b_footpoint,b_satellite/b_equator))))
                PAD_scale_factor[satellite][K_val][valid_mask,i_Mu] = PAD_models[valid_mask,i_Mu]/PAD_integral[valid_mask,i_Mu]

            # PAD scale factor is the flux value of the model at desired angle divided by the integral of the model
            PAD_scale_factor[satellite][K_val] = pd.DataFrame(PAD_scale_factor[satellite][K_val], index=epoch_list, columns=Mu_set) 
    print('Scale Factor Calculated\n')
    return PAD_scale_factor

# %%
