from spacepy import pycdf
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


#file_path = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/rbspa_rel03_ect-rept-sci-l2_20170421_v5.4.0.cdf"
#cdf_data = pycdf.CDF(file_path)
#print(cdf_data)

#print(cdf_data["FESA"].attrs)

ephemeris_path = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/ephemeris/rbsp-a_mag-ephem_def-1min-t89d_20170421_v01.cdf"
ephem_data = pycdf.CDF(ephemeris_path)
#print(ephem_data)
Epoch_ephem = ephem_data['Epoch'][:]

Lm_eq = ephem_data['Lm_eq'][:]

# find local mins to isolate individual parabolas
local_max = False
local_max_time = []
local_min_time = []
for l in range(1, len(Lm_eq)):
    if not local_max and Lm_eq[l-1] < Lm_eq[l]:
        local_max = True
        local_max_time.append(Epoch_ephem[l])
    elif local_max and Lm_eq[l-1] > Lm_eq[l]:
        local_max = False
        local_min_time.append(Epoch_ephem[l])
        
min_timestamp = [epoch.timestamp() for epoch in local_min_time]
min_timestamp = np.array(min_timestamp)
max_timestamp = [epoch.timestamp() for epoch in local_max_time]
max_timestamp = np.array(max_timestamp)
        
# Initialize lists to store popt and pcov for each parabola
all_popt = []
all_pcov = []

# Define the quadratic function
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the curve
timestamps = [epoch.timestamp() for epoch in Epoch_ephem]
timestamps = np.array(timestamps)

# Loop through local minima to fit individual parabolas
for i in range(len(min_timestamp)):  # Adjust range to avoid potential index errors  
    if i == 0:
        start_time = timestamps[0]
        end_time = min_timestamp[i]
    elif i == len(min_timestamp)-1:
        start_time = min_timestamp[i]
        end_time = timestamps[-1]
    else:
        start_time = min_timestamp[i]
        end_time = min_timestamp[i + 1]

    # Find indices of data points within the current parabola
    indices = np.where((timestamps >= start_time) & (timestamps <= end_time))[0]

    # Extract data for the current parabola
    parabola_timestamps = timestamps[indices]
    parabola_lm_eq = Lm_eq[indices]

    # Fit the curve
    try:
        popt_parabola, pcov_parabola = curve_fit(quadratic, parabola_timestamps, parabola_lm_eq)
        all_popt.append(popt_parabola)
        all_pcov.append(pcov_parabola)
    except RuntimeError:
        print(f"Curve fitting failed for parabola {i+1}")
        all_popt.append(None)  # Or handle the error differently
        all_pcov.append(None)


    # Create the scatter plot
    plt.scatter(Epoch_ephem, Lm_eq) 
    #plt.scatter(timestamps, Lm_eq) 
    # Show the plot
    plt.show()