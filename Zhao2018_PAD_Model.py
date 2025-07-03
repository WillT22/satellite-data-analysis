#%% Initialize
import re
import numpy as np
import pandas as pd

#%%
global Zhao_median_filepath
Zhao_median_filepath = '/home/will/Zhao_2018_model_files/PAD_model_coeff_median.txt'

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

#%%