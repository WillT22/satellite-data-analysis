from spacepy import pycdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

textsize = 16

#file_path = "C:/Users/Will/Box/HERT_Box/REPT Data/April 2017 Storm/rbspa_rel03_ect-rept-sci-l2_20170421_v5.4.0.cdf"
file_path = "C:/Users/wzt0020/Box/HERT_Box/REPT Data/April 2017 Storm/rbspa_rel03_ect-rept-sci-l2_20170421_v5.4.0.cdf"
cdf_data = pycdf.CDF(file_path)
print(cdf_data)

Epoch = cdf_data["Epoch"][:]
L = cdf_data["L"][:]
FESA = cdf_data["FESA"][:]
# Calculate minimum and maximum FESA values across all channels
fesa_min = np.min(FESA)
if fesa_min<1:
    fesa_min = 1
norm = colors.LogNorm(vmin=fesa_min, vmax= np.max(FESA))
energy_channels = cdf_data["FESA_Energy"][:]

# Create the figure with subplots
fig, axes = plt.subplots(len(energy_channels), 1, figsize=(16, 40), sharex=True)

# Loop through each energy channel
for i, ax in enumerate(axes.flat):
  # Create the scatter plot on the current subplot
  subplot = ax.scatter(Epoch, L, c=FESA[:, i], cmap='viridis', norm=norm)

  # Add labels and title
  ax.set_ylabel('L', fontsize=textsize)
  ax.set_title(f'RBSP-A REPT {energy_channels[i]:.2f} MeV Electron Spin-Averaged Flux', fontsize=textsize)

  ax.tick_params(axis='both', which='major', labelsize=textsize)
  ax.set_ylim(0, 7)
  ax.grid(True)

# Add x-axis label for last plot
ax.set_xlabel('UTC', fontsize=textsize)

# Remove extra subplots if there aren't enough energy channels
if len(energy_channels) < len(axes.flat):
  for ax in axes.flat[len(energy_channels):]:
    fig.delaxes(ax)

# Create a single colorbar outside the loop for efficiency
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
cbar = fig.colorbar(subplot, cax=cbar_ax, label='Flux')
cbar.set_label(label = 'Flux', fontsize=textsize)
cbar.ax.tick_params(labelsize=textsize) 

# Show the plot
plt.show()
    
# Close the CDF file
cdf_data.close()