from spacepy import pycdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import glob

textsize = 16

# Folder containing CDF files
folder_path = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/"
ephemeris_path = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/ephemeris/"

# Get all CDF file paths in the folder
#cdf_file_paths = glob.glob(folder_path + "*.cdf") 
cdf_file_paths_A = glob.glob(folder_path + "rbspa*[!r]*.cdf") 
cdf_file_paths_B = glob.glob(folder_path + "rbspb*[!r]*.cdf") 
#ephem_file_paths = glob.glob(ephemeris_path + "*.cdf")
ephem_file_paths_A = glob.glob(ephemeris_path + "rbsp-a*[!r]*.cdf")
ephem_file_paths_B = glob.glob(ephemeris_path + "rbsp-b*[!r]*.cdf")


# Function for reading in RBSP flux data
def process_flux_data(file_paths):
    Epoch = []
    L = []
    FESA = None
    for file_path in file_paths:
        # Extract filename without path
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")
        # Load the CDF data
        cdf_data = pycdf.CDF(file_path)

        Epoch.extend(cdf_data["Epoch"][:])
        L.extend(cdf_data["L"][:])
        if FESA is None:
            FESA = cdf_data["FESA"][:]
            energy_channels = cdf_data["FESA_Energy"][:]
        else:
                FESA = np.vstack((FESA, cdf_data["FESA"][:]))
        cdf_data.close()
    return Epoch, L, FESA, energy_channels

print("Processing Flux Data:")
Epoch_A, L_A, FESA_A, energy_channels_A = process_flux_data(cdf_file_paths_A)
Epoch_B, L_B, FESA_B, energy_channels_B = process_flux_data(cdf_file_paths_B)

# Handle cases where only A or B data is present (check which lists are not empty)
if not Epoch_A and not L_A and not FESA_A:
    print("No RBSPA data found in the folder.")
if not Epoch_B and not L_B and not FESA_B:
    print("No RBSPB data found in the folder.")
    
# Find the earliest and latest Epoch values
if Epoch_A and Epoch_B: 
    min_epoch = min(min(Epoch_A), min(Epoch_B))
    max_epoch = max(max(Epoch_A), max(Epoch_B))
else:
    # Handle cases where either Epoch_A or Epoch_B is empty
    if Epoch_A:
        min_epoch = min(Epoch_A)
        max_epoch = max(Epoch_A)
    elif Epoch_B:
        min_epoch = min(Epoch_B)
        max_epoch = max(Epoch_B)
    else:
        min_epoch = None
        max_epoch = None
    
# Calculate minimum and maximum FESA values across all channels
# divide by 1000 for keV to compare to Zhao 2018
if FESA_A is not None and FESA_B is not None:
    fesa_min = np.min(np.vstack((FESA_A/1000, FESA_B/1000)))
    fesa_max = np.max(np.vstack((FESA_A/1000, FESA_B/1000)))
elif FESA_A is not None:
    fesa_min = np.min(FESA_A/1000)
    fesa_max = np.max(FESA_A/1000)
elif FESA_B is not None:
    fesa_min = np.min(FESA_B/1000)
    fesa_max = np.max(FESA_B/1000)
if fesa_min<1:
    fesa_min = 1
norm = colors.LogNorm(vmin=fesa_min, vmax= fesa_max)

'''
Interpolate Lm_eq from ephemeris data
'''
# Function for reading in RBSP ephemeris data
def process_ephem_data(ephem_file_paths):
    Epoch_ephem = []
    Lm_eq = []
    for f, file_path in enumerate(ephem_file_paths):
        # Extract filename without path
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")
        # Load the CDF data
        ephem_data = pycdf.CDF(file_path)
        
        if f < len(ephem_file_paths) - 1:  # Exclude the last file for Epoch_ephem_A and Lm_eq_A
            Epoch_ephem.extend(ephem_data['Epoch'][:-1]) 
            Lm_eq.extend(ephem_data['Lm_eq'][:-1])
        else:
            Epoch_ephem.extend(ephem_data['Epoch'][:])
            Lm_eq.extend(ephem_data['Lm_eq'][:])
        ephem_data.close()
    return Epoch_ephem, Lm_eq

print("Processing Ephemeris Data:")
Epoch_ephem_A, Lm_eq_A = process_ephem_data(ephem_file_paths_A)
Epoch_ephem_B, Lm_eq_B = process_ephem_data(ephem_file_paths_B)

# Interpolate for RBSP data
def interpolate_Lm_eq(Epoch, Epoch_ephem, Lm_eq):
    Epoch_float = [epoch.timestamp() for epoch in Epoch]
    Epoch_ephem_float = [epoch_ephem.timestamp() for epoch_ephem in Epoch_ephem]
    Lm_eq_interp = np.interp(Epoch_float, Epoch_ephem_float, Lm_eq)
    return Lm_eq_interp

print("Interpolating McIlwain L:")
Lm_eq_A_interp = interpolate_Lm_eq(Epoch_A, Epoch_ephem_A, Lm_eq_A)
Lm_eq_B_interp = interpolate_Lm_eq(Epoch_B, Epoch_ephem_B, Lm_eq_B)


'''
Plot RBSP Flux Data with ephemeris Lm_eq
'''
print("Plotting Data:")
# Create a custom colormap based on 'nipy_spectral' to match with IDL rainbow
cmap = plt.get_cmap('nipy_spectral') 
new_cmap = cmap(np.linspace(0, 0.875, 256))  # Use only the first 87.5% of the colormap

# Create a new colormap object
custom_cmap = colors.ListedColormap(new_cmap)

# Create the figure with subplots
fig, axes = plt.subplots(len(energy_channels_A), 1, figsize=(16, 40), sharex=True)

# Loop through each energy channel
for i, ax in enumerate(axes.flat):
  # Create the scatter plot on the current subplot
  # divide by 1000 for keV to compare to Zhao 2018
  if FESA_A is not None:
      subplot = ax.scatter(Epoch_A, Lm_eq_A_interp, c=FESA_A[:, i]/1000, cmap=custom_cmap, norm=norm)
  if FESA_B is not None:
      subplot = ax.scatter(Epoch_B, Lm_eq_B_interp, c=FESA_B[:, i]/1000, cmap=custom_cmap, norm=norm)

  # Add labels and title
  ax.set_ylabel('L', fontsize=textsize)
  ax.set_title(f'RBSP REPT {energy_channels_A[i]:.2f} MeV Electron Spin-Averaged Flux', fontsize=textsize)
  if min_epoch is not None and max_epoch is not None:
        ax.set_xlim(min_epoch, max_epoch) 
  ax.tick_params(axis='both', which='major', labelsize=textsize)
  ax.set_yticks(np.arange(2, 8, 1))  # Set ticks from 2 to 7 with interval 1
  ax.set_ylim(2, 7)
  ax.grid(True)

# Add x-axis label for last plot
ax.set_xlabel('UTC', fontsize=textsize)
fig.suptitle('April 21-26, 2017 RBSP-B REPT Data', fontsize=textsize+4, y=0.9)

# Remove extra subplots if there aren't enough energy channels
if len(energy_channels_A) < len(axes.flat):
  for ax in axes.flat[len(energy_channels_A):]:
    fig.delaxes(ax)

# Create a single colorbar outside the loop for efficiency
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot, cax=cbar_ax, label='Flux')
# divide by 1000 for keV to compare to Zhao 2018
cbar.set_label(label = 'Flux 'r"(cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)", fontsize=textsize)
cbar.ax.tick_params(labelsize=textsize) 

# Show the plot
plt.show()
