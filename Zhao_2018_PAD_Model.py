#%% Import relevant libraries
import re
import numpy as np
import pandas as pd
import datetime as dt
from scipy.special import eval_legendre

import importlib
import GPS_PSD_func
importlib.reload(GPS_PSD_func)

#%% Initialize Global Variables
ZHAO_MEDIAN_FILEPATH = '/home/wzt0020/sat_data_analysis/satellite-data-analysis/Zhao_2018_model_files/PAD_model_coeff_median.txt'

#%% Extract coefficients from Zhao_2018
def import_zhao_coeffs(filepath=None):
    """
    Parses the Zhao et al. (2018) PAD model coefficient text file into a nested dictionary.

    Structure: zhao_coeffs[energy][dst_range][coeff_name] -> {'MLT_values', 'L_values', 'data_matrix'}
    
    Args:
        filepath (str, optional): Path to the coefficient file. Defaults to global ZHAO_MEDIAN_FILEPATH.
        NOTE: Energy, MLT, and L values are midpoints of the bins, not edges.

    Returns:
        dict: Nested dictionary structure: coefficients[energy][dst_range][coeff_name].
              Contains 'MLT_values', 'L_values', and 'data_matrix' (DataFrame).
    """
    
    if filepath is None:
        filepath = ZHAO_MEDIAN_FILEPATH
        
    print("Importing Zhao Coefficients... \r")
    global zhao_coeffs
    zhao_coeffs = {}

    # Regex patterns for different header lines
    energy_pattern = re.compile(r"E=(\d+\.?\d*|\.\d+)\s*(keV|MeV)")
    dst_gt_pattern = re.compile(r"Dst\s*([<>])\s*(-?\d+)\s*nT")
    dst_range_pattern = re.compile(r"(-?\d+)\s*nT\s*<\s*Dst\s*<\s*(-?\d+)\s*nT")
    coeff_pattern = re.compile(r"c(\d+)") # Captures the number after 'c'
    mlt_l_header_pattern = re.compile(r"MLT\\L,\s*(.*)") # Captures everything after "MLT\L,"

    # State variables for parsing
    current_energy = None
    current_dst_range = None
    current_coeff_block = None
    current_l_values = None
    current_mlt_values = []
    current_data_rows = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            stripped_line = line.strip()
            if not stripped_line: continue

            # 1. Detect Energy Header
            match_energy = energy_pattern.match(stripped_line)
            if match_energy:
                val = float(match_energy.group(1))
                unit = match_energy.group(2)
                current_energy = val / 1000.0 if unit == 'keV' else val
                
                zhao_coeffs[current_energy] = {}
                current_dst_range = None 
                continue

            # 2. Detect Dst Range Header
            match_dst_gt = dst_gt_pattern.match(stripped_line)
            match_dst_range = dst_range_pattern.match(stripped_line)

            if match_dst_gt:
                current_dst_range = f"Dst {match_dst_gt.group(1)} {int(match_dst_gt.group(2))} nT"
                zhao_coeffs[current_energy][current_dst_range] = {}
                current_coeff_block = None
                continue
            elif match_dst_range:
                current_dst_range = f"{int(match_dst_range.group(1))} nT < Dst < {int(match_dst_range.group(2))} nT"
                zhao_coeffs[current_energy][current_dst_range] = {}
                current_coeff_block = None
                continue
                        
            # Ensure we are in a valid energy and Dst context
            if current_energy is None or current_dst_range is None:
                print(f"Warning: Skipping line {line_num} ('{stripped_line}') - No active Energy or Dst context.")
                continue

            # 3. Detect Coefficient Header (c2, c4, etc.)
            match_coeff = coeff_pattern.match(stripped_line)
            if match_coeff:
                current_coeff_block = f"c{match_coeff.group(1)}"
                zhao_coeffs[current_energy][current_dst_range][current_coeff_block] = {}
                # Reset data buffers for new block
                current_l_values = None
                current_mlt_values = []
                current_data_rows = []
                continue

            # Ensure we are in a valid coefficient context
            if current_coeff_block is None:
                print(f"Warning: Skipping line {line_num} ('{stripped_line}') - No active Coefficient context.")
                continue

            # 4. Detect L-Values Header line (defines columns)
            # NOTE: for energies < 1MeV, L=1-6, for energies > 1MeV, L=3-6
            match_mlt_l_header = mlt_l_header_pattern.match(stripped_line)
            if match_mlt_l_header:
                l_values_str = match_mlt_l_header.group(1).replace(',', ' ').split()
                try:
                    current_l_values = np.array([float(val) for val in l_values_str])
                    zhao_coeffs[current_energy][current_dst_range][current_coeff_block]['L_values'] = current_l_values
                except ValueError:
                    print(f"Warning: Skipping line {line_num} ('{stripped_line}') - Malformed L_values header.")
                    current_l_values = None # Invalidate L_values for this block
                continue

            # Ensure L_values header has been parsed for data rows
            if current_l_values is None:
                print(f"Warning: Skipping line {line_num} ('{stripped_line}') - No L_values header found for data.")
                continue

            # 5. Parse Data Rows
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
            if current_mlt_values:
                # Create Pandas DataFrame with MLT as index and L as columns
                data_df = pd.DataFrame(current_data_rows, index=current_mlt_values, columns=current_l_values)
                zhao_coeffs[current_energy][current_dst_range][current_coeff_block]['MLT_values'] = np.array(current_mlt_values)
                zhao_coeffs[current_energy][current_dst_range][current_coeff_block]['data_matrix'] = data_df
        print("Zhao Coefficients Imported.\n")
    return zhao_coeffs

#%% Find PAD coefficients from Zhao 2018 model
def find_Zhao_PAD_coeffs(sat_data, QD_data, energyofmualpha, extMag = 'T89c'):
    """
    Look up Zhao coefficients for satellite epochs based on Energy, Dst, L, and MLT.

    Args:
        sat_data (dict): Processed satellite data (Epoch, MLT, L_shell, Position).
        QD_data (dict): Qin-Denton geomagnetic indices (DateTime, Dst).
        energyofmualpha (dict): Energy values for specific Mu/K coordinates.
        extMag (str): Magnetic field model label ('T89c', 'TS04').

    Returns:
        dict: Nested dict {K_val: {Mu_val: DataFrame}} containing 5 coeffs per epoch.
    """
    extMag_label = 'T89' if extMag == 'T89c' else extMag
    full_coeff_list = ['c2', 'c4', 'c6', 'c8', 'c10']
    
    # Pre-load keys and data
    # Note: 'zhao_coeffs' must be available in global scope or passed in. 
    # Assuming it was loaded via import_Zhao_coeffs() and assigned globally or available.
    if 'zhao_coeffs' not in globals():
        raise ValueError("Global variable 'zhao_coeffs' not found. Run import_Zhao_coeffs first.")

    # Extract energy bins from the global zhao_coeffs dictionary to serve as keys in the Zhao_epoch_coeff DataFrame
    energy_bins = np.array(list(zhao_coeffs.keys()), dtype=float)
    # Define the valid range for energy_value besed on instrument energy channels.
    echannel_min = sat_data['Energy_Channels'][0]
    
    # --- Vectorized Pre-calculations ---
    # 1. Time Alignment (Round to 5 mins)
    epochs_dt = sat_data['Epoch'].UTC
    # Rounding vectorization using Pandas is usually fastest
    times_pd = pd.to_datetime(epochs_dt)
    rounded_times = times_pd.round('5min').to_pydatetime()

    # 2. Find Dst indices
    # Assumes QD_data['DateTime'] is sorted. using searchsorted is O(log N) vs argmin O(N)
    # Ensure QD_data times are sorted for searchsorted
    qd_times = QD_data['DateTime']
    time_indices = np.searchsorted(qd_times, rounded_times)
    # Clip to bounds just in case
    time_indices = np.clip(time_indices, 0, len(qd_times)-1)

    # 3. Extract Bulk Parameters
    dst_values = QD_data['Dst'][time_indices]
    l_shells = np.atleast_1d(sat_data[f'L_LGM_{extMag_label}IGRF'])

    # 4. Calculate Magnetic Latitude (Vectorized)
    # R = norm(Position)
    pos_data = sat_data['Position'].data
    R = np.linalg.norm(pos_data, axis=1)
    # Lat = arccos(sqrt(R/L)) * 180/pi
    # Clip R/L to <= 1.0 to avoid NaNs from numerical noise
    ratio_rl = np.clip(R / l_shells, 0, 1.0)
    mag_latitudes = np.rad2deg(np.arccos(np.sqrt(ratio_rl)))

    # 5. Determine Dst Strings per epoch
    # We can map Dst values to indices 0, 1, 2 for the three categories
    dst_cats = np.zeros_like(dst_values, dtype=object)
    dst_cats[dst_values > -20] = 'Dst > -20 nT'
    dst_cats[(dst_values <= -20) & (dst_values > -50)] = '-50 nT < Dst < -20 nT'
    dst_cats[dst_values <= -50] = 'Dst < -50 nT'

    # 6. MLT Binning
    # ((MLT + 1) % 24) // 2
    mlt_data = np.atleast_1d(sat_data['MLT'])
    mlt_bins = (((mlt_data + 1) % 24) // 2).astype(int)

    # --- Processing Loop ---
    zhao_epoch_coeffs = {}
    # Prepare a list of formatted epoch strings for the column headers of the final Pandas DataFrame for each Mu_value.
    epoch_str = [dt_obj.strftime("%Y-%m-%dT%H:%M:%S") for dt_obj in sat_data['Epoch'].UTC]
    
    # --- Loop 1: Iterate through each K value (K_val is the index, K_data is the energy data for that Mu) ---
    for K_val, K_data in energyofmualpha.items():
        zhao_epoch_coeffs[K_val] = {}
        K_data_values = K_data.values   

        # --- Loop 2: Iterate through each Mu value in Mu_set ---
        # Get the Mu_set array for this satellite for use as a key in zhao_epoch_coeffs.
        Mu_set = np.array(list(energyofmualpha[K_val].columns.tolist()), dtype=float)
        for i_mu, Mu_value in enumerate(Mu_set):
            # Result container: (N_epochs, 5 coeffs)
            
            coeffs_matrix = np.zeros((len(epochs_dt), 5))
            energies = K_data_values[:, i_mu]
        
            # Find closest energy bin indices for all epochs
            # abs(energy - bins) -> argmin
            # Using broadcasting: (N_epochs, 1) - (1, N_bins)
            diffs = np.abs(energies[:, None] - energy_bins[None, :])
            ebin_indices = np.argmin(diffs, axis=1)
            target_ebins = energy_bins[ebin_indices]

            # Identify valid rows (Primary Filter)
            # E > min, E <= 6.2, L <= 6
            # Note: Removed mag_lat/local90 filter lines as they were commented out in source
            valid_mask = (energies >= (echannel_min - 0.15)) & \
                         (energies <= 6.2) & \
                         (l_shells <= 6)

            # To optimize the lookup, we iterate only over unique combinations of 
            # (EnergyBin, DstRange) present in the valid data.
            # This significantly reduces dictionary lookups compared to per-epoch iteration.

            valid_indices = np.where(valid_mask)[0]

            # If no valid data, skip
            if len(valid_indices) == 0:
                zhao_epoch_coeffs[K_val][Mu_value] = pd.DataFrame(coeffs_matrix, index=epoch_str, columns=full_coeff_list)
                continue

            # Iterate over valid indices to fill coefficients
            # (Further vectorization is hard due to ragged L-bins in the Zhao dictionary structure)
            for idx in valid_indices:
                ebin = target_ebins[idx]
                dst_cat = dst_cats[idx]
                l_curr = l_shells[idx]
                mlt_idx = mlt_bins[idx]
                
                # Retrieve the coeff block
                coeff_dict = zhao_coeffs[ebin][dst_cat]
                
                # Logic for L-binning based on Energy Regime
                # E < 1 MeV: L bins start at 1.0, step 0.2
                # E >= 1 MeV: L bins start at 3.0, step 0.2
                if ebin < 1:
                    l_bin_idx = int((l_curr - 0.9) // 0.2)
                    # Bounds checks and coefficient selection based on L
                    if l_curr < 2:
                        for ic, c in enumerate(full_coeff_list):
                             coeffs_matrix[idx, ic] = coeff_dict[c]['data_matrix'].values[mlt_idx, l_bin_idx]
                    elif 2 <= l_curr < 4:
                        for ic, c in enumerate(full_coeff_list):
                            if c != 'c10':
                                coeffs_matrix[idx, ic] = coeff_dict[c]['data_matrix'].values[mlt_idx, l_bin_idx]
                    elif l_curr >= 4:
                         for ic, c in enumerate(full_coeff_list):
                            if c not in ['c8', 'c10']:
                                coeffs_matrix[idx, ic] = coeff_dict[c]['data_matrix'].values[mlt_idx, l_bin_idx]
                else:
                    if l_curr >= 3:
                        l_bin_idx = int((l_curr - 2.9) // 0.2)
                        for ic, c in enumerate(full_coeff_list):
                            if c not in ['c8', 'c10']:
                                coeffs_matrix[idx, ic] = coeff_dict[c]['data_matrix'].values[mlt_idx, l_bin_idx]

            zhao_epoch_coeffs[K_val][Mu_value] = pd.DataFrame(coeffs_matrix, index=epoch_str, columns=full_coeff_list)
            
    return zhao_epoch_coeffs

#%% Create PAD from Zhao 2018 model coefficients
def define_Legendre(alpha):
    """
    Computes even Legendre polynomials (P2, P4, P6, P8, P10) for given pitch angles.

    Args:
        alpha (float or array): Pitch angles in degrees.

    Returns:
        numpy.ndarray: Transposed array of shape (5, len(alpha)) containing Pn(cos(alpha)).
    """
    
    alpha_rad = np.deg2rad(alpha)
    cos_alpha = np.cos(alpha_rad)
    # Use eval_legendre for efficiency (avoids object creation overhead)
    # Orders: 2, 4, 6, 8, 10
    P = np.array([eval_legendre(n, cos_alpha) for n in [2, 4, 6, 8, 10]])
    return P.transpose()

def create_PAD(sat_data, QD_data, energyofmualpha, extMag = 'T89c'):
    """
    Generates the full Pitch Angle Distribution (PAD) model for all epochs.

    Args:
        zhao_epoch_coeffs (dict): Fitted coefficients from find_Zhao_PAD_coeffs.

    Returns:
        dict: PAD_model containing 'Model' (flux) and 'pitch_angles' for each K/Mu.
    """

    global zhao_epoch_coeffs
    zhao_epoch_coeffs = find_Zhao_PAD_coeffs(sat_data, QD_data, energyofmualpha, extMag)
    
    alpha_init = np.linspace(0,180,361)

    # Pre-compute Legendre basis for the fixed alpha grid (Shape: 361 x 5)
    # define_Legendre returns (N, 5), we need (5, N) for the dot product or use broadcasting
    P_basis = define_Legendre(alpha_init) # Shape: (361, 5)

    PAD_model = {}
    for K_val, K_data in zhao_epoch_coeffs.items():
        PAD_model[K_val] = {}
        for Mu_val, Mu_data in K_data.items():
            coeff_data = Mu_data.values # Shape: (N_epochs, 5)
            epoch_list = Mu_data.index.tolist()
            
            PAD_model[K_val][Mu_val] = {}
            
            # --- Vectorized Calculation ---
            # Instead of looping epochs: Flux = 1 + coeffs @ P_basis.T
            # (N_epochs, 5) @ (5, 361) -> (N_epochs, 361)
            model_matrix = 1 + np.dot(coeff_data, P_basis.T)
            
            # Create repeated pitch angle matrix (N_epochs, 361)
            pitch_angle_matrix = np.tile(alpha_init, (len(epoch_list), 1))
            
            PAD_model[K_val][Mu_val]['pitch_angles'] = pd.DataFrame(pitch_angle_matrix, index=epoch_list)
            PAD_model[K_val][Mu_val]['Model'] = pd.DataFrame(model_matrix, index=epoch_list)
            
    return PAD_model

#%% Find Integral of Normalized PAD between Loss Cone and Local 90 PA
# Analytical integrals of Legendre Polynomials over cos(alpha)
def P0_int_eq(x,a):
    return -2/a*np.sqrt(1 - a*x)
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
    """
    Calculates the definite integrals of Legendre polynomials between loss cone and local 90.

    Args:
        b_sat, b_eq, b_fpt, b_min (array): Magnetic field values at satellite, equator, footpoint, min.

    Returns:
        numpy.ndarray: Transposed array of integrated values for orders 2, 4, 6, 8, 10.
    """
    
    low = b_min / b_fpt
    high = b_eq / b_sat
    ratio = b_sat / b_eq

    P_int = np.array([
        P2_int_eq(high, ratio) - P2_int_eq(low, ratio),
        P4_int_eq(high, ratio) - P4_int_eq(low, ratio),
        P6_int_eq(high, ratio) - P6_int_eq(low, ratio),
        P8_int_eq(high, ratio) - P8_int_eq(low, ratio),
        P10_int_eq(high, ratio) - P10_int_eq(low, ratio)
    ])
    return P_int.transpose()

def PAD_Scale_Factor(sat_data, QD_data, energyofmualpha, alphaofK, extMag = 'T89c'):
    """
    Calculates the scale factor (normalization ratio) to map the PAD model to measured flux.

    Args:
        sat_data (dict): Contains magnetic field data ('b_satellite', etc.)
        Zhao_epoch_coeffs (dict): Coefficients from find_Zhao_PAD_coeffs.
        alphaofK (DataFrame): Pitch angles for specific K values.

    Returns:
        dict: PAD_scale_factor[K_val] as a DataFrame of scaling ratios.
    """

    if not zhao_epoch_coeffs:
        global zhao_epoch_coeffs
        zhao_epoch_coeffs = find_Zhao_PAD_coeffs(sat_data, QD_data, energyofmualpha, extMag)
    
    print('Calculating Scale Factor...')
    PAD_int_out = {}
    PAD_scale_factor = {}
    K_set = np.array(list(zhao_epoch_coeffs.keys()), dtype=float)
    
    # Extract B-field arrays (assumed to be aligned with epoch_list length)
    b_satellite = sat_data['b_satellite']
    b_equator = sat_data['b_min']
    b_footpoint = sat_data['b_footpoint']
    b_min = sat_data['b_min']

    # Inputs for PAD model generation (and logic restriction)
    loss_cone = sat_data['loss_cone']
    alphaofK_data = alphaofK.values
    
    for K_val, K_data in zhao_epoch_coeffs.items():
        i_K = np.where(K_set == K_val)[0][0]
        Mu_set = np.array(list(K_data.keys()), dtype=float)
        epoch_list = K_data[Mu_set[0]].index.tolist()
        
        # Initialize containers
        # Using a dictionary of DataFrames immediately saves re-wrapping later
        PAD_int_out[K_val] = pd.DataFrame(np.nan, index=epoch_list, columns=Mu_set)
        PAD_scale_factor[K_val] = pd.DataFrame(np.nan, index=epoch_list, columns=Mu_set)
        
        for Mu_val, Mu_data in K_data.items():
            coeffs = Mu_data.values # Shape (N_epochs, 5)
            
            # --- 1. Create Mask ---
            # Valid if: Coeffs exist AND particle is outside loss cone
            valid_mask = (np.sum(coeffs, axis=1) != 0) & (alphaofK_data[:, i_K] > loss_cone)
            
            if not np.any(valid_mask): continue # Skip if no valid data

            # --- 2. Subset Data ---
            # Only work with the valid rows to save computation
            coeffs_sub = coeffs[valid_mask]
            alpha_sub = alphaofK_data[valid_mask, i_K]
            b_sat_sub = b_satellite[valid_mask]
            b_eq_sub = b_equator[valid_mask]
            b_fpt_sub = b_footpoint[valid_mask]
            b_min_sub = b_min[valid_mask]

            # --- 3. Compute Model Value ---
            # Note: Alpha varies by epoch here, so we can't use a fixed basis.
            P = define_Legendre(alpha_sub)
            
            # --- Vectorized Integral Calculation ---
            # Calculate integrals for basis functions
            P_int = define_Legendre_Int_eq(b_sat_sub, b_eq_sub, b_fpt_sub, b_min_sub)
                        
            # Integral of Isotropic term (P0)
            # P0 integral = [P0_int_eq(high) - P0_int_eq(low)]
            ratio = b_sat_sub / b_eq_sub
            low = b_min_sub / b_fpt_sub
            high = b_eq_sub / b_sat_sub
            P0_term = P0_int_eq(high, ratio) - P0_int_eq(low, ratio)

            # Full Integral: GeometricFactor * ( Sum(c_n * I_n) + I_0 )
            # Note: 2*pi for azimuthal symmetry
            integral_val = (2 * np.pi * ratio * (np.sum(coeffs_sub * P_int, axis=1) + P0_term))

            # Model Value at Alpha: Sum(c_n * P_n) + 1
            PAD_models_val = np.sum(coeffs_sub * P, axis=1) + 1
            
            # --- 5. Assign Results ---
            # Map the subset results back to the full array using the mask
            PAD_int_out[K_val][Mu_val].values[valid_mask] = integral_val
            
            # Calculate Ratio and assign
            # No need for extra divide checks here since integral_val should be valid if physics holds
            ratio_result = PAD_models_val / integral_val
            PAD_scale_factor[K_val][Mu_val].values[valid_mask] = ratio_result

    print('Scale Factor Calculated\n')
    return PAD_scale_factor