from spacepy import pycdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

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


for i in range(energy_channels.shape[0]):
    # Set the figure size (width, height) in inches
    plt.figure(figsize=(16, 4)) 
    
    # Create the scatter plot
    plt.scatter(Epoch, L, c=FESA[:,i], cmap='viridis', norm=norm) 
    
    # Add labels and title
    plt.xlabel('UTC', fontsize=12)
    plt.ylabel('L', fontsize=12)
    plt.title(f'RBSP-A REPT {energy_channels[i]:.2f} MeV Electron Spin-Averaged Flux', fontsize=16)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 7) 
    
    # Add grid lines
    plt.grid(True) 
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label(label = 'Flux', fontsize=12)
    cbar.ax.tick_params(labelsize=12) 
    
    # Show the plot
    plt.show()