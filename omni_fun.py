import os
from spacepy import pycdf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

folder = '/home/will/.spacepy/data/omni2cdfs'
file = 'omni2_h0_mrg1hr_20000101_v01.cdf'

omni_data = pycdf.CDF(os.path.join(folder,file))

processed_variables_data={}
for key in omni_data.keys():
    processed_variables_data[key] = omni_data[key][:]

start_date  = "01/01/2000"
stop_date   = "03/01/2000" # exclusive, end of the last day you want to see

start_date_dt = dt.datetime.strptime(start_date, "%m/%d/%Y")
stop_date_dt = dt.datetime.strptime(stop_date, "%m/%d/%Y")

Epoch = omni_data['Epoch'][:]
plot_mask = (Epoch >= start_date_dt) & (Epoch < stop_date_dt)

birthday = dt.datetime.strptime("01/26/2000", "%m/%d/%Y")

plot_variables = [
        'BZ_GSM',
        'BY_GSM',
        'V',
        'N',
        'Pressure',
        'DST',
        'KP',
        'AE',
    ]

fig, axes = plt.subplots(len(plot_variables), 1, figsize=(12, 2.5 * len(plot_variables)), sharex=True)
fig.subplots_adjust(hspace=0.1) # Reduce vertical space between subplots
base_font_size = 14 
for i, var_name in enumerate(plot_variables):
    ax = axes[i] # Get the current subplot axis
    var_data = omni_data[var_name][:]
    var_attrs = omni_data[var_name].attrs # Get attributes for labels

    # Plot the data
    ax.plot(Epoch[plot_mask], var_data[plot_mask], label=var_name)

    # Set Y-axis label and add a grid
    y_label_text = var_name
    if 'UNITS' in var_attrs:
        y_label_text += f" ({var_attrs['UNITS']})"
    elif 'UNIT' in var_attrs: # Some CDFs use UNIT
        y_label_text += f" ({var_attrs['UNIT']})"

    if var_name == 'KP':
        ax.set_ylabel('Kp*10',fontsize=base_font_size)
    else:
        ax.set_ylabel(y_label_text,fontsize=base_font_size)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.axvline(birthday, color='red', linestyle='--', linewidth=1.5)

    if var_name == 'N':
        ax.set_ylim(0, 50)
    elif var_name == 'Pressure':
        ax.set_ylim(0, 30)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=7)) # Major ticks every 1 day
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=12)) # Minor ticks every 6 hours
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d')) # Format date on x-axis
ax.tick_params(axis='both', which='major', labelsize=base_font_size)

# Rotate tick labels for better readability if needed
fig.autofmt_xdate()

# Set common X-axis label only for the bottom-most subplot
axes[-1].set_xlabel('Time (UTC)', fontsize=base_font_size)

# Add a main title for the entire figure
fig.suptitle('OMNI2 Data in 2000', fontsize=16, y=0.9) # Adjust y position if needed

plt.show()