from spacepy import pycdf, toolbox as tb
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
#from matplotlib import colors
from spacepy.time import Ticktock
from spacepy.coordinates import Coords
import spacepy.irbempy as irbem
import spacepy.omni as omni

#tb.update(QDomni=True)
#tb.update(omni2=True)

textsize = 16

#l2
#file_path = "C:/Users/Will/Box/Multipoint_Box/REPT Data/April 2017 Storm/l2/rbspa_rel03_ect-rept-sci-l2_20170421_v5.4.0.cdf"
#l3
file_path = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/l3/rbspa_rel03_ect-rept-sci-l3_20170421_v5.6.0.cdf"
cdf_data = pycdf.CDF(file_path)
#print(cdf_data)
Epoch = cdf_data['Epoch'][:]
Position = cdf_data['Position'][:]
#print(cdf_data['Position'].attrs)
B_Eq = cdf_data['B_Eq'][:]
#L_star = cdf_data['L_star'][:]

#l3 data
alpha = cdf_data['FEDU_Alpha'][:]
FEDU = cdf_data['FEDU'][:]

#print(cdf_data["FESA"].attrs)
cdf_data.close()

Re = 6378.137
Position = Position / Re

ephemeris_path = "C:/Users/Will/Box/Multipoint_Box/REPT Data/April 2017 Storm/ephemeris/rbsp-a_mag-ephem_def-1min-t89d_20170421_v01.cdf"
ephem_data = pycdf.CDF(ephemeris_path)
#print(ephem_data)
Epoch_ephem = ephem_data['Epoch'][:]
alpha_ephem = ephem_data['Alpha'][:]
Bm = ephem_data['Bm'][:]
Lm_ephem = ephem_data['Lm_eq'][:]
Lstar_ephem = ephem_data['Lstar'][:]
K_ephem = ephem_data['K'][:]
ephem_data.close()

# Calculate magnetic field strength at mirror points
Bm_calc = np.outer(B_Eq, 1/np.sin(alpha*np.pi/180)**2)

# Find second adiabatic invariant and L* from IRBEM
Bmin = np.zeros((len(Epoch)))
Bmirr = np.zeros((len(Epoch)))
Lm = np.zeros((len(Epoch)))
Lstar = np.zeros((len(Epoch)))
MLT = np.zeros((len(Epoch)))
Xj = np.zeros((len(Epoch)))

#Bmin = np.zeros((len(Epoch), len(alpha)))
#Bmirr = np.zeros((len(Epoch), len(alpha)))
#Lm = np.zeros((len(Epoch), len(alpha)))
#Lstar = np.zeros((len(Epoch), len(alpha)))
#MLT = np.zeros((len(Epoch), len(alpha)))
#Xj = np.zeros((len(Epoch), len(alpha)))

print("Converting Time")
time = Ticktock(Epoch, 'UTC')
print("Converting Position")
position = Coords(Position, 'GEO', 'car')

print("Obtaining Omnivals")
mag_key_mapping = {
    'Kp_index': 'Kp',
    'Dst_index': 'Dst',
    'PC_N_index': 'dens',  # 'N' maps to 'dens'
    'Plasma_bulk_speed': 'velo',  # 'V' maps to 'velo'
    'Flow_pressure': 'Pdyn',  # 'Pressure' maps to 'Pdyn'  (Using Flow_pressure)
    'By_GSM': 'ByIMF',  # 'BY_GSM' maps to 'ByIMF'
    'Bz_GSM': 'BzIMF',  # 'BZ_GSM' maps to 'BzIMF'
    'AL_index': 'AL',
}
omnivals_refined = {}
mag_key_unused = ['G1', 'G2', 'G3', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6']
for key in mag_key_unused:
    omnivals_refined[key] = np.full(len(time), np.nan)
omnivals=omni.get_omni(time, dbase='OMNI2hourly')
omnivals['Kp_index'] = omnivals['Kp_index']/10
for cdf_key, mag_key in mag_key_mapping.items():
    if cdf_key in omnivals:  # Check if the key exists in the CDF
        omnivals_refined[mag_key] = omnivals[cdf_key][:].copy()
    else:
        print(f"Warning: Key '{cdf_key}' not found in CDF data. Skipping.")

#print("Calculating Magnetic Mirror Strength")
#results = irbem.find_Bmirror(time, position, alpha=alpha, extMag='0', omnivals=omnivals_refined)
#keys = ["Blocal", "Bmirr"]

#print("Calculating Lm")
#results = irbem.get_Lm(time, position, alpha=alpha, extMag='T89', omnivals=omnivals_refined)
#keys = ["Bmin", "Bmirr", "Lm", "MLT", "Xj"]

#print("Calculating L*")
#results = irbem.get_Lstar(time, position, alpha=alpha, extMag='T89', omnivals=omnivals_refined)
#keys = ["Bmin", "Bmirr", "Lm", "Lstar", "MLT", "Xj"]
#for j, key in enumerate(keys):
#    locals()[key] = results[key]

# Create the scatter plot
#plt.scatter(Epoch[0:100], Lstar[0:100,1]) 
#plt.show()

# Interpolate for RBSP data
def interpolate_LK(Epoch, Epoch_ephem, Lm_ephem, Lstar_ephem, K_ephem):
    Epoch_float = [epoch.timestamp() for epoch in Epoch]
    Epoch_ephem_float = [epoch_ephem.timestamp() for epoch_ephem in Epoch_ephem]
    Lm_interp = np.interp(Epoch_float, Epoch_ephem_float, Lm_ephem)
    Lstar_interp = np.interp(Epoch_float, Epoch_ephem_float, Lstar_ephem)
    K_interp = np.interp(Epoch_float, Epoch_ephem_float, K_ephem)
    return Lm_interp, Lstar_interp, K_interp

print("Interpolating Ephemeris Data:")
Lm_interp, Lstar_interp, K_interp = interpolate_LK(Epoch, Epoch_ephem, Lm_ephem, Lstar_ephem[:,10], K_ephem[:,10])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))  
ax1.scatter(Epoch, Lstar_interp)
ax2.scatter(Epoch, Lstar, color='orange')
ax1.set_title("Ephemeris L*")  # Top plot label
ax2.set_title("L*")           # Bottom plot label
# Force labels for first and last x-axis tick marks 
min_epoch = datetime(1970, 1, 1) + timedelta(hours=math.floor((Epoch.min() - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
max_epoch = datetime(1970, 1, 1) + timedelta(hours=math.ceil((Epoch.max() - datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
ax2.set_xlim(min_epoch, max_epoch) 
ax2.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3) )
ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H'))
ax1.set_yticks(np.arange(2, 8, 1))
ax1.set_ylim(2, 7)
ax2.set_yticks(np.arange(2, 8, 1))
ax2.set_ylim(2, 7)
fig.autofmt_xdate()
plt.show()

fig, ax = plt.subplots()  
ax.scatter(Lstar_interp, Lstar, s=2, color = 'black')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Normalized Ephemeris L*')
ax.set_ylabel('Normalized L*')
ax.set_title('Comparison of Normalized Ephemeris L* and L*')
ax.grid(True)
ax.set_aspect('equal')
plt.show()
    
'''
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
'''