from spacepy import pycdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import glob

textsize = 16

# Folder containing CDF files
folder_path = "C:/Users/Will/Box/Multipoint_Box/REPT Data/April 2017 Storm/"
ephemeris_path = "C:/Users/Will/Box/Multipoint_Box/REPT Data/April 2017 Storm/ephemeris/"

# Get all CDF file paths in the folder
#cdf_file_paths = glob.glob(folder_path + "rbspa*[!r]*.cdf") 
cdf_file_paths = glob.glob(folder_path + "*.cdf") 
ephem_file_paths = glob.glob(ephemeris_path + "*.cdf")

# Initialize empty dictionaries to store data from all files (separate for A and B)
Epoch_A = []
L_A = []
FESA_A = None

Epoch_B = []
L_B = []
FESA_B = None

for file_path in cdf_file_paths:
    # Extract filename without path
    file_name = os.path.basename(file_path)
    print(f"Processing file: {file_name}")
    # Load the CDF data
    cdf_data = pycdf.CDF(file_path)

    # Separate data based on filename prefix
    if file_name.startswith("rbspa"):
        # Store data in A variables
        Epoch_A.extend(cdf_data["Epoch"][:])
        L_A.extend(cdf_data["L"][:])
        if FESA_A is None:
            FESA_A = cdf_data["FESA"][:]
            energy_channels = cdf_data["FESA_Energy"][:]
        else:
            FESA_A = np.vstack((FESA_A, cdf_data["FESA"][:]))
    else:
        # Store data in B variables
        Epoch_B.extend(cdf_data["Epoch"][:])
        L_B.extend(cdf_data["L"][:])
        if FESA_B is None:
            FESA_B = cdf_data["FESA"][:]
            energy_channels = cdf_data["FESA_Energy"][:]
        else:
            FESA_B = np.vstack((FESA_B, cdf_data["FESA"][:]))
    cdf_data.close()

# Handle cases where only A or B data is present (check which lists are not empty)
if not Epoch_A and not L_A and not FESA_A:
    print("No RBSPA data found in the folder.")
if not Epoch_B and not L_B and not FESA_B:
    print("No RBSPB data found in the folder.")
    
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

Epoch_ephem_A = []
Lm_eq_A = []
Epoch_ephem_B = []
Lm_eq_B = []

# Read in RBPS data
for file_path in enumerate(ephem_file_paths):
    # Extract filename without path
    file_name = os.path.basename(file_path)
    print(f"Processing file: {file_name}")
    # Load the CDF data
    ephem_data = pycdf.CDF(file_path)
    
    Epoch_ephem = ephem_data['Epoch'][:]
    Lm_eq = ephem_data['Lm_eq'][:]
    
    # Separate data based on filename prefix
    if file_name.startswith("rbsp-a"):
        # Store data in A variables
        if f < len(ephem_file_paths) - 1:  # Exclude the last file for Epoch_ephem_A and Lm_eq_A
            Epoch_ephem_A.extend(ephem_data['Epoch'][:-1]) 
            Lm_eq_A.extend(ephem_data['Lm_eq'][:-1])
        else:
            Epoch_ephem_A.extend(ephem_data['Epoch'][:])
            Lm_eq_A.extend(ephem_data['Lm_eq'][:])
    elif file_name.startswith("rbsp-b"):
        # Store data in A variables
        if f < len(ephem_file_paths) - 1:  # Exclude the last file for Epoch_ephem_A and Lm_eq_A
            Epoch_ephem_B.extend(ephem_data['Epoch'][:-1]) 
            Lm_eq_B.extend(ephem_data['Lm_eq'][:-1])
        else:
            Epoch_ephem_B.extend(ephem_data['Epoch'][:])
            Lm_eq_B.extend(ephem_data['Lm_eq'][:])
    ephem_data.close()

# Interpolate for RBSP-A data
if Epoch_ephem_A != []:
    for t in range(len(Epoch_ephem_A)-1):
        #t1 = Epoch_ephem[t]                 #t2 = Epoch_ephem[t+1]
        #Lm_p1 = Lm_eq[t]                    #Lm_p2 = Lm_eq[t+1]
        # Calculate change in time between two points
        delta_t = Epoch_ephem_A[t+1]-Epoch_ephem_A[t]
        # Calculate slope of linear equations (convert time delta to float)
        m = (Lm_eq_A[t+1]-Lm_eq_A[t])/delta_t.total_seconds()
        
        # Caulcuate time since first point (set first point to t=0)
        time_since_start = Epoch_ephem_A[t]-Epoch_ephem_A[0]
        # Calculate y-int of linear equations (convert time delta to float)
        b = Lm_eq_A[t] - m * time_since_start.total_seconds()
'''
# Interpolate for RBSP-B data
if Epoch_ephem_B != []:
    for t in range(len(Epoch_ephem_B)-1):
        #t1 = Epoch_ephem[t]                 #t2 = Epoch_ephem[t+1]
        #Lm_p1 = Lm_eq[t]                    #Lm_p2 = Lm_eq[t+1]
        # Calculate change in time between two points
        delta_t = Epoch_ephem_B[t+1]-Epoch_ephem_B[t]
        # Calculate slope of linear equations (convert time delta to float)
        m = (Lm_eq_B[t+1]-Lm_eq_B[t])/delta_t.total_seconds()
        
        # Caulcuate time since first point (set first point to t=0)
        time_since_start = Epoch_ephem_B[t]-Epoch_ephem_B[0]
        # Calculate y-int of linear equations (convert time delta to float)
        b = Lm_eq_B[t] - m * time_since_start.total_seconds()
'''
'''
Plot RBSP Flux Data with ephemeris Lm_eq
'''
# Create a custom colormap based on 'nipy_spectral' to match with IDL rainbow
cmap = plt.get_cmap('nipy_spectral') 
new_cmap = cmap(np.linspace(0, 0.875, 256))  # Use only the first 87.5% of the colormap

# Create a new colormap object
custom_cmap = colors.ListedColormap(new_cmap)
'''
# Create the figure with subplots
fig, axes = plt.subplots(len(energy_channels), 1, figsize=(16, 40), sharex=True)

# Loop through each energy channel
for i, ax in enumerate(axes.flat):
  # Create the scatter plot on the current subplot
  # divide by 1000 for keV to compare to Zhao 2018
  if FESA_A is not None:
      subplot = ax.scatter(Epoch_A, L_A, c=FESA_A[:, i]/1000, cmap=custom_cmap, norm=norm)
  if FESA_B is not None:
      subplot = ax.scatter(Epoch_B, L_B, c=FESA_B[:, i]/1000, cmap=custom_cmap, norm=norm)

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
'''