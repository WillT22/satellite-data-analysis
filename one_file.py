from spacepy import pycdf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import spacepy.coordinates as coords
from spacepy.time import Ticktock

textsize = 16

file_path = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/rbspa_rel03_ect-rept-sci-l2_20170421_v5.4.0.cdf"
cdf_data = pycdf.CDF(file_path)
#print(cdf_data)
Epoch = cdf_data["Epoch"][:]
B_Eq = cdf_data['B_Eq'][:]
L_star = cdf_data['L_star'][:]


#print(cdf_data["FESA"].attrs)

ephemeris_path = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/ephemeris/rbsp-a_mag-ephem_def-1min-t89d_20170421_v01.cdf"
ephem_data = pycdf.CDF(ephemeris_path)
#print(ephem_data)
Epoch_ephem = ephem_data['Epoch'][:]
alpha = ephem_data['Alpha'][:]
Bm = ephem_data['Bm'][:]
Lm_eq = ephem_data['Lm_eq'][:]

Bm_calc = np.outer(B_Eq, 1/np.sin(alpha*np.pi/180)**2)

# Create the scatter plot
#plt.scatter(Epoch_ephem, Lm_eq) 
#plt.show()
    
    
# Create the figure with subplots
fig, axes = plt.subplots(len(alpha), 1, figsize=(20, 40), sharex=True)

# Loop through each pitch angle
for i, ax in enumerate(axes.flat):
  # Create the scatter plot on the current subplot
  subplot = ax.scatter(Epoch, L_star, c=Bm_calc[:, i], norm=colors.LogNorm())
  #subplot = ax.scatter(Epoch_ephem, Lm_eq, c=Bm[:, i], norm=colors.LogNorm())

  # Add labels and title
  ax.set_ylabel('L', fontsize=textsize)
  ax.set_title(f'{alpha[i]:.0f} Degree Pitch Angle', fontsize=textsize)
  # Force labels for first and last x-axis tick marks 
  min_epoch = datetime(1970, 1, 1) + timedelta(hours=math.floor((Epoch_ephem.min() - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
  max_epoch = datetime(1970, 1, 1) + timedelta(hours=math.ceil((Epoch_ephem.max() - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
  ax.set_xlim(min_epoch, max_epoch) 
  # Set time labels every 12 hours
  ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=12) )
  ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H')) 
  ax.tick_params(axis='both', which='major', labelsize=textsize)
  ax.set_yticks(np.arange(2, 8, 1))  # Set ticks from 2 to 7 with interval 1
  ax.set_ylim(2, 7)
  ax.grid(True)
  
  cbar = plt.colorbar(subplot, ax=ax, shrink=0.9, pad=0.01)  # Adjust shrink as needed
  # Flux is in (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)
  cbar.set_label(label = 'B', fontsize=textsize)
  cbar.ax.tick_params(labelsize=textsize)

# Add x-axis label for last plot
ax.set_xlabel('UTC', fontsize=textsize)
fig.suptitle('April 21-26, 2017 RBSP REPT Data', fontsize=textsize+4, y=0.9)

plt.show()