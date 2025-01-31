from spacepy import pycdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import glob

textsize = 16

# Folder containing CDF files
folder_path = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/"
ephemeris_path = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/ephemeris"

# Get all CDF file paths in the folder
#cdf_file_paths = glob.glob(folder_path + "rbspa*[!r]*.cdf") 
cdf_file_paths = glob.glob(folder_path + "*.cdf") 
ephem_file_paths = glob.glob(ephemeris_path + "*.cdf")

# Initialize empty lists to store data from all files
Epoch = []
L = []
FESA = None

for file_path in cdf_file_paths:
    print(f"Processing file: {file_path}")
    # Load the CDF data
    cdf_data = pycdf.CDF(file_path)

    # Append data to lists
    Epoch.extend(cdf_data["Epoch"][:])
    L.extend(cdf_data["L"][:])

    # Check if FESA is None (first iteration)
    if FESA is None:
        # If first iteration, directly assign FESA
        FESA = cdf_data["FESA"][:]
        energy_channels = cdf_data["FESA_Energy"][:]
    else:
        # For subsequent iterations, stack vertically
        FESA = np.vstack((FESA, cdf_data["FESA"][:]))
        
    cdf_data.close()

# Calculate minimum and maximum FESA values across all channels
# divide by 1000 for keV to compare to Zhao 2018
fesa_min = np.min(FESA/1000)
if fesa_min<1:
    fesa_min = 1
norm = colors.LogNorm(vmin=fesa_min, vmax= np.max(FESA/1000))

# Create a custom colormap based on 'nipy_spectral'
cmap = plt.get_cmap('nipy_spectral') 
new_cmap = cmap(np.linspace(0, 0.875, 256))  # Use only the first 87.5% of the colormap

# Create a new colormap object
custom_cmap = colors.ListedColormap(new_cmap)

# Create the figure with subplots
fig, axes = plt.subplots(len(energy_channels), 1, figsize=(16, 40), sharex=True)

# Loop through each energy channel
for i, ax in enumerate(axes.flat):
  # Create the scatter plot on the current subplot
  # divide by 1000 for keV to compare to Zhao 2018
  subplot = ax.scatter(Epoch, L, c=FESA[:, i]/1000, cmap=custom_cmap, norm=norm)

  # Add labels and title
  ax.set_ylabel('L', fontsize=textsize)
  ax.set_title(f'RBSP-B REPT {energy_channels[i]:.2f} MeV Electron Spin-Averaged Flux', fontsize=textsize)

  ax.tick_params(axis='both', which='major', labelsize=textsize)
  ax.set_ylim(2, 7)
  ax.grid(True)

# Add x-axis label for last plot
ax.set_xlabel('UTC', fontsize=textsize)
fig.suptitle('April 21-26, 2017 RBSP-B REPT Data', fontsize=textsize+4, y=0.9)

# Remove extra subplots if there aren't enough energy channels
if len(energy_channels) < len(axes.flat):
  for ax in axes.flat[len(energy_channels):]:
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