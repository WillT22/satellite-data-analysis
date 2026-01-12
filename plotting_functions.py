import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from matplotlib import colors
import matplotlib.animation as animation
import datetime as dt
import pandas as pd

# Add current directory to path for local imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_script_dir)

from GPS_PSD_func import energy_spectra  # Required for GPS spectral fit

#%% RBSP (ONLY) Plotting Functions
'''


PLOTS ONLY DATA FROM RBSP SATELLITES


'''

#%% GPS (ONLY) Plotting Functions
'''


PLOTS ONLY DATA FROM GPS SATELLITES


'''

#%% Plot Flux for a Single Energy Channel for GPS Data
def plot_gps_flux(gps_data, start_date, stop_date, extMag='T89c', 
                          target_energy=2.1, textsize=16):
    """
    Args:
        gps_data (dict): Dictionary containing processed GPS satellite data.
        start_date (datetime): Start time for the plot.
        stop_date (datetime): Stop time for the plot.
        extMag (str): External magnetic field model identifier (e.g., 'T89c', 'TS04').
        target_energy (float): The energy channel to plot (MeV). Closest match will be found.
        textsize (int): Font size for plot labels.
    """
    
    # 1. Setup Parameters
    # Find index of closest energy channel from the first satellite
    first_sat = list(gps_data.keys())[0]
    energy_channels = gps_data[first_sat]['Energy_Channels']
    i_energy = np.argmin(np.abs(energy_channels - target_energy))
    actual_energy = energy_channels[i_energy]

    # 2. Configure Limits
    min_val = np.nanmin(np.log10(1e2))
    max_val = np.nanmax(np.log10(1e6))
    extMag_label = 'T89' if extMag == 'T89c' else extMag

    # 3. Create Plot
    fig, ax = plt.subplots(figsize=(16, 4))
    
    scatter_A = None
    for satellite, sat_data in gps_data.items():     
        # Filter valid flux data
        flux_plot = sat_data['electron_diff_flux'][:, i_energy]
        flux_mask = (flux_plot > 0) & (~np.isnan(flux_plot))
        
        # Determine L-shell variable key
        l_key = f'L_LGM_{extMag_label}IGRF'
        if l_key not in sat_data:
             print(f"Warning: L-shell key {l_key} not found for {satellite}. Skipping.")
             continue

        # Scatter Plot: Time vs L, colored by log10(Flux)
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[flux_mask], 
                               sat_data[l_key][flux_mask],
                               c=np.log10(flux_plot[flux_mask]), 
                               vmin=min_val, vmax=max_val)

    if scatter_A is None:
        print("No valid flux data found to plot.")
        plt.close(fig)
        return

    # 4. Format Axes
    ax.set_title(f"GPS CXD, {actual_energy:.2f} MeV Electron Differential Flux", fontsize=textsize + 2)
    ax.set_ylabel(r"McIlwain L", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # Calculate Axis Limits (Midnight to Midnight)
    min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    ax.set_ylim(3, 6)
    
    # Time formatting
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    ax.grid(True)

    # 5. Add Colorbar
    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, 
                        format=ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    cbar.set_label(label=r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)
    plt.show()


#%% Plot Phase Space Density (PSD) for GPS data
def plot_gps_psd(gps_data, start_date, stop_date, K=0.1, Mu=2000, textsize=16):
    """
    Args:
        gps_data (dict): Dictionary containing processed GPS satellite data, including PSD.
        start_date (datetime): Start time for the plot.
        stop_date (datetime): Stop time for the plot.
        K (float, optional): Specific K value to plot. Defaults to 0.1.
        Mu (float, optional): Specific Mu value to plot. Defaults to 2000.
        textsize (int, optional): Font size for plot labels. Defaults to 16.
    """

    # 1. Setup Parameters
    K_set = np.array(list(gps_data[next(iter(gps_data))]['PSD'].keys()))
    Mu_set = np.array(list(gps_data[next(iter(gps_data))]['PSD'][K_set[0]].keys()))
    i_K = np.where(K_set == K)[0]
    i_mu = np.where(Mu_set == Mu)[0]

    # 2. Configure Plot
    fig, ax = plt.subplots(figsize=(16, 4))

    # Custom Colormap
    colorscheme = plt.cm.get_cmap('turbo')(np.linspace(0, 0.85, 256))
    cmap = colors.ListedColormap(colorscheme)

    min_val = np.nanmin(np.log10(1e-12))
    max_val = np.nanmax(np.log10(1e-5))

    scatter_A = None

    # 3. Plot Data
    for satellite, sat_data in gps_data.items():

        psd_plot = sat_data['PSD'][K].values[:,i_mu].copy().flatten()
        psd_mask = (psd_plot > 0) & (~np.isnan(psd_plot))
        
        l_key = 'Lstar'
        if l_key not in sat_data: continue

        scatter_A = ax.scatter(sat_data['Epoch'].UTC[psd_mask], 
                               sat_data[l_key][psd_mask, i_K],
                               c=np.log10(psd_plot[psd_mask]), 
                               cmap=cmap, vmin=min_val, vmax=max_val, 
                               marker='*', s=80, alpha=0.7)

    if scatter_A is None:
        print("No valid PSD data found to plot.")
        plt.close(fig)
        return

    # 4. Format Axes
    ax.set_title(f"GPS CXD, K={K:.1f} $G^{{1/2}}R_E$, $\\mu$={Mu:.0f} $MeV/G$", fontsize=textsize + 2)
    ax.set_ylabel(r"L*", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    ax.set_ylim(3, 5.5)
    
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    ax.grid(True)

    # 5. Add Colorbar
    cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, 
                        format=ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
    
    tick_locations = np.arange(min_val, max_val + 1)
    cbar.set_ticks(tick_locations)
    cbar.set_label(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize)
    cbar.ax.tick_params(labelsize=textsize)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=0.82, right=0.95)
    plt.show()

#%% Plot Energies corresponding to Mu and Alpha across L*
def plot_energy_mu_alpha(gps_data, energyofmualpha, start_date, stop_date, K=0.1, Mu=2000, textsize=16):
    """
    Args:
        gps_data (dict): Processed GPS data containing Lstar.
        energyofmualpha (dict): Dictionary of calculated energies for specific Mu/K.
        start_date, stop_date (datetime): Time range for the plot.
        K (float): Specific K value to plot. Defaults to 0.1.
        Mu (float): Specific Mu value to plot. Defaults to 2000.
        textsize (int): Font size for labels.
    """
    from matplotlib import colors
    
    # 1. Setup Parameters
    # Get K_set and Mu_set from the first satellite to find indices
    first_sat = list(energyofmualpha.keys())[0]
    K_set = np.array(list(energyofmualpha[first_sat].keys()), dtype=float)
    # Assuming structure is consistent
    first_K_val = list(energyofmualpha[first_sat].keys())[0]
    Mu_set = np.array(list(energyofmualpha[first_sat][first_K_val].columns), dtype=float)

    try:
        i_K = np.where(np.isclose(K_set, K))[0][0]
        i_mu = np.where(np.isclose(Mu_set, Mu))[0][0]
    except IndexError:
        print(f"Requested K={K} or Mu={Mu} not found in energy data.")
        return

    # 2. Configure Colormap (Time)
    # Use truncated plasma to avoid too-bright yellow at the end if desired, matching previous style
    colormap_name = 'plasma'
    plasma = plt.cm.get_cmap(colormap_name)
    cmap = colors.LinearSegmentedColormap.from_list('truncated_plasma', plasma(np.linspace(0, 0.9, 256)))
    
    vmin = mdates.date2num(start_date)
    vmax = mdates.date2num(stop_date)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # 3. Plot Data
    fig, ax = plt.subplots(figsize=(20, 8))
    
    scatter_plot = None
    for satellite, sat_data in gps_data.items():
        if satellite not in energyofmualpha: continue
        
        # Masking
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= start_date) & (sat_data['Epoch'].UTC <= stop_date)
        
        # Get energy values for the specific Mu/K
        # energyofmualpha[sat][K] is a DataFrame with Mu columns
        energy_plot = energyofmualpha[satellite][K].values[:, i_mu].copy().flatten()
        
        # Valid Data Mask
        energy_mask = (energy_plot > 0) & (~np.isnan(energy_plot))
        
        if 'Lstar' not in sat_data: continue
        lstar_mask = (sat_data['Lstar'][:, i_K] > 0).flatten()
        
        combined_mask = energy_mask & sat_iepoch_mask & lstar_mask
        
        # Scatter Plot: L* vs Energy, colored by Time
        scatter_plot = ax.scatter(sat_data['Lstar'][combined_mask, i_K], 
                                  energy_plot[combined_mask], 
                                  c=mdates.date2num(sat_data['Epoch'].UTC[combined_mask]), 
                                  cmap=cmap, vmin=vmin, vmax=vmax)
    
    if scatter_plot is None:
        print("No valid energy data found to plot.")
        plt.close(fig)
        return

    # 4. Format Plot
    # Colorbar
    cbar = fig.colorbar(scatter_plot, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Time (UTC)', fontsize=textsize)
    cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    cbar.ax.tick_params(labelsize=textsize-2)

    # Annotations
    ax.text(0.5, 0.92, r"K = " + f"{K:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{Mu:.0f}" + r" $MeV/G$", 
            transform=ax.transAxes, fontsize=textsize) 

    # Axes
    ax.set_xlim(3.8, 5.2)
    ax.set_ylim(1, 5)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlabel(r"L*", fontsize=textsize)
    ax.set_ylabel(r"Energy (MeV)", fontsize=textsize)
    ax.grid(True)
    
    plt.show()

#%% Combined Plotting Functions
'''


PLOTS DATA FROM BOTH GPS AND RBSP SATELLITES TOGETHER


'''

#%% Plot Combined Flux from REPT and CXD for All Energy Channels + DST
def plot_combined_flux_all_channels(gps_data, REPT_data, QD_storm_data, start_date, stop_date, extMag='T89c', max_energy=4, target_K_set=0.1, textsize=16):
    """
    Args:
        gps_data (dict): Processed GPS data.
        REPT_data (dict): Processed REPT data.
        QD_storm_data (dict): Qin-Denton OMNI data for DST plotting.
        start_date, stop_date (datetime): Time range.
        extMag (str): Magnetic model label.
        textsize (int): Base font size.
    """
    
    # 1. Select Energy Channels (< 4 MeV)
    # Get channels from the first available REPT satellite
    first_rept = list(REPT_data.keys())[0]
    energy_channels = REPT_data[first_rept]['Energy_Channels']
    energy_channels = energy_channels[energy_channels < max_energy]

    extMag_label = 'T89' if extMag == 'T89c' else extMag

    # 2. Setup Multi-Panel Plot
    fig, axes = plt.subplots(len(energy_channels) + 1, 1, figsize=(24, 10), sharex=True, sharey=False)
    
    colormap_name = 'viridis'
    cmap = plt.cm.get_cmap(colormap_name)
    
    scatter_A = None # Placeholder for colorbar mapping

    # 3. Loop through Energy Channels
    for i_energy, energy in enumerate(energy_channels):
        ax = axes[i_energy]
        
        # --- A. Plot REPT Data (Background) ---
        for satellite, sat_data in REPT_data.items():
            sat_iepoch_mask = (sat_data['Epoch'].UTC >= start_date) & (sat_data['Epoch'].UTC <= stop_date)
            
            # Extract and Average Flux
            flux_slice = sat_data['FEDU_averaged'][sat_iepoch_mask, :, i_energy]
            flux_temp_mask = np.where(flux_slice >= 0, flux_slice, np.nan)
            flux_plot = np.nanmean(flux_temp_mask, axis=1) / 2
            
            flux_mask = (flux_plot > 0) & (flux_plot != np.nan)
            combined_mask = np.zeros_like(sat_iepoch_mask, dtype=bool)
            combined_mask[sat_iepoch_mask] = flux_mask
            
            vmax = 7 
            
            scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], 
                                   sat_data[f'L_LGM_{extMag_label}IGRF'][combined_mask],
                                   c=np.log10(flux_plot[flux_mask]), 
                                   cmap=cmap, vmin=0, vmax=vmax, zorder=2)

        # --- B. Plot GPS CXD Data (Overlay) ---
        for satellite, sat_data in gps_data.items():    
            energy_input = {}
            sat_iepoch_mask = (sat_data['Epoch'].UTC >= start_date) & (sat_data['Epoch'].UTC <= stop_date)
            epoch_index = sat_data['Epoch'].UTC
            
            energy_input[target_K_set] = {}
            energy_input[target_K_set][energy] = pd.Series(
                data=np.full(len(epoch_index), energy), 
                index=epoch_index
            )
        
            # Calculate Flux using Spectral Fit
            flux_result = energy_spectra(sat_data, energy_input)
            flux_plot = flux_result[target_K_set][energy]
            
            flux_mask = (flux_plot > 0) & (flux_plot != np.nan)

            ax.scatter(sat_data['Epoch'].UTC[flux_mask], 
                       sat_data[f'L_LGM_{extMag_label}IGRF'][flux_mask],
                       marker='*', s=80, alpha=0.7,
                       c=np.log10(flux_plot[flux_mask]), 
                       vmin=0, vmax=vmax, zorder=1)

        # Formatting
        ax.set_title(f"{energy:.2f} MeV", fontsize=textsize+4)
        ax.tick_params(axis='both', labelsize=textsize, pad=5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_ylim(3, 6.5)
        ax.grid(True)

    # 4. Colorbar (Global)
    if scatter_A:
        cbar_ax = fig.add_axes([0.96, 0.27, 0.02, 0.61]) 
        cbar = fig.colorbar(scatter_A, cax=cbar_ax, 
                            format=ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
        cbar.set_label(label=r'Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize, labelpad=5)
        cbar.ax.tick_params(labelsize=textsize - 2)

    # 5. Plot DST Index (Bottom Subplot)
    ax = axes[-1]
    QD_dates_array = np.array(QD_storm_data['DateTime'])
    iepoch_mask = (QD_dates_array >= start_date) & (QD_dates_array <= stop_date)
    
    ax.plot(QD_dates_array[iepoch_mask], QD_storm_data['Dst'][iepoch_mask], color='black')
    
    ax.tick_params(axis='both', labelsize=textsize, pad=5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    
    min_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.floor((start_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12) 
    max_epoch = dt.datetime(1970, 1, 1) + dt.timedelta(hours=np.ceil((stop_date - dt.datetime(1970, 1, 1)).total_seconds() / 3600 / 12) * 12)
    ax.set_xlim(min_epoch, max_epoch)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0)
    ax.set_ylabel(r'DST (nT)', fontsize=textsize)
    ax.grid(True)

    # X-Axis Formatting (Bottom Only)
    ax.set_xlabel('Time (UTC)', fontsize=textsize+2, labelpad=2)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    ax.tick_params(axis='x', labelsize=textsize+2, pad=12)

    # Global Labels
    fig.text(0.08, 0.575, r'McIlwain L', fontsize=textsize+2, rotation='vertical', va='center')

    # Legend
    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='RBSP')
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None', markersize=12, label='GPS')
    
    fig.legend(handles=[handle_rbsp, handle_gps],
               title='Satellite',
               title_fontsize=textsize,
               loc='upper right',
               bbox_to_anchor=(0.98, 1.005),
               handlelength=1,
               fontsize=textsize-2)

    plt.subplots_adjust(right=0.95, hspace=0.35)
    plt.show()

#%% Plot Combined Phase Space Density (PSD) from REPT and GPS CXD
def plot_combined_psd(gps_data, REPT_data, start_date, stop_date, K=0.1, Mu=2000, textsize=16):
    """
    Args:
        gps_data (dict): Dictionary containing processed GPS satellite data.
        REPT_data (dict): Dictionary containing processed REPT satellite data.
        start_date (datetime): Start time for the plot.
        stop_date (datetime): Stop time for the plot.
        extMag (str, optional): External magnetic field model identifier. Defaults to 'T89c'.
        K (float, optional): Specific K value to plot. Defaults to 0.1.
        Mu (float, optional): Specific Mu value to plot. Defaults to 2000.
        textsize (int, optional): Font size for plot labels. Defaults to 16.
    """
    # 1. Setup Parameters (Derive indices from REPT data structure)
    # Assumes REPT_data[sat]['PSD'][K] is a DataFrame with Mu columns
    # We need i_K and i_mu for Lstar indexing which is (N_epoch, N_K)
    
    # Get K_set/Mu_set from first REPT satellite to find indices
    first_rept = list(REPT_data.keys())[0]
    K_set = np.array(list(REPT_data[first_rept]['PSD'].keys()), dtype=float)
    Mu_set = np.array(list(REPT_data[first_rept]['PSD'][K_set[0]].columns), dtype=float)

    try:
        i_K = np.where(np.isclose(K_set, K))[0][0]
        i_mu = np.where(np.isclose(Mu_set, Mu))[0][0]
    except IndexError:
         print(f"Requested K={K} or Mu={Mu} not found in REPT PSD data.")
         return

    # 2. Configure Plot
    fig, ax = plt.subplots(figsize=(24, 2.5))
    
    # Colormap (Turbo)
    colorscheme = plt.cm.get_cmap('turbo')(np.linspace(0, 0.85, 256))
    cmap = colors.ListedColormap(colorscheme)
    
    # Limits
    min_val = np.nanmin(np.log10(1e-12))
    max_val = np.nanmax(np.log10(1e-5))

    # 3. Plot REPT Data (Background Circles)
    scatter_A = None
    for satellite, sat_data in REPT_data.items():
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= start_date) & (sat_data['Epoch'].UTC <= stop_date)
        
        # Check if PSD key exists
        if 'PSD' not in sat_data or K not in sat_data['PSD']: continue

        psd_plot = sat_data['PSD'][K].values[:, i_mu].copy().flatten()
        
        psd_mask = (psd_plot > 0) & (~np.isnan(psd_plot))
        lstar_mask = sat_data['Lstar'][:, 0] > 0
        combined_mask = psd_mask & lstar_mask & sat_iepoch_mask
        
        scatter_A = ax.scatter(sat_data['Epoch'].UTC[combined_mask], 
                               sat_data['Lstar'][combined_mask, i_K],
                               c=np.log10(psd_plot[combined_mask]), 
                               cmap=cmap, vmin=min_val, vmax=max_val, zorder=2)

    # 4. Plot GPS Data (Overlay Stars)
    for satellite, sat_data in gps_data.items():
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= start_date) & (sat_data['Epoch'].UTC <= stop_date)
        # Accessing using scalar K (key) and scalar index i_mu (col)
        psd_plot = sat_data['PSD'][K].values[:, i_mu].copy().flatten()
        
        psd_mask = (psd_plot > 0) & (~np.isnan(psd_plot))
        combined_mask = psd_mask & sat_iepoch_mask
        
        if 'Lstar' in sat_data:
            # Assuming GPS Lstar has same shape/structure as REPT regarding K index
            l_data = sat_data['Lstar'][:, i_K]
            
            ax.scatter(sat_data['Epoch'].UTC[combined_mask], 
                        l_data[combined_mask], 
                        marker='*', s=80, alpha=0.7,
                        c=np.log10(psd_plot[combined_mask]), 
                        cmap=cmap, vmin=min_val, vmax=max_val, zorder=1)

    # 5. Formatting
    ax.set_title(f"RBSP REPT & GPS CXD Phase Space Density, K={K:.1f} $G^{{1/2}}R_E$, $\\mu$={Mu:.0f} $MeV/G$", 
                 fontsize=textsize+10, y=1.1)
    ax.set_ylabel(r"L*", fontsize=textsize)
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    
    ax.set_xlim(start_date, stop_date)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_ylim(3.6, 5.4)
    ax.grid(True)

    # Colorbar
    if scatter_A:
        cbar = fig.colorbar(scatter_A, ax=ax, fraction=0.03, pad=0.01, 
                            format=ticker.FuncFormatter(lambda val, pos: r"$10^{{{:.0f}}}$".format(val)))
        tick_locations = np.arange(min_val, max_val + 1)
        cbar.set_ticks(tick_locations)
        cbar.set_label(r"PSD $(c/MeV/cm)^3$", fontsize=textsize)
        cbar.ax.tick_params(labelsize=textsize)

    # Legend
    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=10, label='RBSP') 
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None',
                               markersize=12, label='GPS') 
    
    ax.legend(handles=[handle_rbsp, handle_gps],
              title='Satellite',
              title_fontsize=textsize,
              loc='upper right',
              bbox_to_anchor=(1.09, 1.85),
              handlelength=1,
              fontsize=textsize-2)

    plt.xticks(fontsize=textsize)
    plt.subplots_adjust(top=1, right=0.95)
    plt.show()

#%% Plot Pitch Angle Distribution (PAD) Comparison. 
# Compares instantaneous REPT PAD vs Model vs GPS Data.
def plot_pad_comparison(gps_data, gps_flux, gps_energy, gps_alpha, REPT_data, 
                             REPT_energyofmualpha, QD_storm_data, time_select,
                             gps_pad_models=None, REPT_sat_select='rbspa', 
                             extMag='T89c', K=0.1, Mu=2000, textsize=16):
    """ 
    Args:
        gps_data (dict): GPS satellite data.
        gps_flux, gps_energy, gps_alpha (dict): Pre-calculated GPS dictionaries.
        REPT_data (dict): REPT satellite data.
        REPT_energyofmualpha (dict): REPT energy calculations.
        QD_storm_data (dict): Qin-Denton OMNI data.
        time_select (datetime): Snapshot time.
        gps_pad_models (dict, optional): Pre-calculated GPS PAD models. Defaults to None.
        REPT_sat_select (str, optional): REPT satellite to use as reference ('rbspa' or 'rbspb'). Defaults to 'rbspa'.
        extMag (str, optional): Magnetic model. Defaults to 'T89c'.
        K, Mu (float, optional): Invariants. Defaults to 0.1 and 2000.
        textsize (int, optional): Font size. Defaults to 16.
    """

    # 1. Setup Labels
    rbsp_label = 'RBSP-A' if REPT_sat_select == 'rbspa' else 'RBSP-B'
    extMag_label = 'T89' if extMag == 'T89c' else extMag

    # 2. Nearest REPT Epoch & DST
    nearest_it_REPT = np.argmin(np.abs(REPT_data[REPT_sat_select]['Epoch'].UTC - time_select))
    nearest_time_REPT = REPT_data[REPT_sat_select]['Epoch'].UTC[nearest_it_REPT]
    
    QD_dates = np.array(QD_storm_data['DateTime'])
    dst_idx = np.argmin(np.abs(QD_dates - nearest_time_REPT))
    dst_val = QD_storm_data['Dst'][dst_idx]
    
    if dst_val > -20: i_dst = 'Dst > -20 nT'
    elif dst_val > -50: i_dst = '-50 nT < Dst < -20 nT'
    else: i_dst = 'Dst < -50 nT'

    # 3. REPT Binning Info
    # Access Energy/Alpha using K/Mu keys
    energy_at_mu = REPT_energyofmualpha[REPT_sat_select][K][Mu].iloc[nearest_it_REPT]
    rept_energies = REPT_data[REPT_sat_select]['Energy_Channels']
    i_energy = np.argmin(np.abs(rept_energies[0:6] - energy_at_mu))
    rept_E = rept_energies[i_energy]
    
    from Zhao_2018_PAD_Model import import_zhao_coeffs, create_PAD
    zhao_coeffs = import_zhao_coeffs()
    zhao_E_keys = np.array(list(zhao_coeffs.keys()))
    closest_E = zhao_E_keys[np.argmin(np.abs(zhao_E_keys - rept_E))]
    
    L_bins = zhao_coeffs[closest_E][i_dst]['c2']['L_values']
    MLT_bins = zhao_coeffs[closest_E][i_dst]['c2']['MLT_values']
    
    L_ref = REPT_data[REPT_sat_select][f'L_LGM_{extMag_label}IGRF'][nearest_it_REPT]
    MLT_ref = REPT_data[REPT_sat_select]['MLT'][nearest_it_REPT]
    
    L_ibin = np.argmin(np.abs(L_bins - L_ref))
    MLT_ibin = np.argmin(np.abs(MLT_bins - MLT_ref))
    L_bin = L_bins[L_ibin]
    MLT_bin = MLT_bins[MLT_ibin]
    
    print(f'{REPT_sat_select} found at MLT = {MLT_ref:.2f}, L = {L_ref:.2f}, E = {rept_E:.2f}')

    # 4. REPT Pitch Angles
    PA_local = REPT_data[REPT_sat_select]['Pitch_Angles']
    B_min = REPT_data[REPT_sat_select]['b_min'][nearest_it_REPT]
    B_sat = REPT_data[REPT_sat_select]['b_satellite'][nearest_it_REPT]
    
    arg = np.sin(np.deg2rad(PA_local))**2 * (B_min / B_sat)
    PA_eq = np.rad2deg(np.arcsin(np.sqrt(np.clip(arg, 0, 1))))
    PA_eq = np.unique(np.concatenate((PA_eq, 180 - PA_eq)))
    PA_local90 = PA_eq[len(PA_local)-1]
    
    PAD_data = REPT_data[REPT_sat_select]['FEDU'][nearest_it_REPT, :, i_energy]
    PAD_data = np.insert(PAD_data, len(PA_local), PAD_data[len(PA_local)-1])

    # 5. REPT Model Generation
    rept_epoch_data = {}
    K_set = np.array(list(REPT_data[REPT_sat_select]['PSD'].keys()), dtype=float)
    for k_key, v in REPT_data[REPT_sat_select].items():
        if k_key in ['Energy_Channels', 'Pitch_Angles']: 
            rept_epoch_data[k_key] = v
        elif k_key in ['FEDU', 'FEDU_averaged']: 
            rept_epoch_data[k_key] = v[nearest_it_REPT,:,:]
        elif k_key == 'Mu_calc': 
            continue
        elif k_key == 'PSD':
            rept_epoch_data[k_key] = {}
            if K in v and Mu in v[K]:
                 val = v[K][Mu].values[nearest_it_REPT]
                 rept_epoch_data[k_key][K] = pd.DataFrame(val, index=[nearest_time_REPT], columns=[Mu])
        elif k_key == 'Lstar': 
            if v.ndim > 1:
                 k_idx = np.argmin(np.abs(K_set - K))
                 rept_epoch_data[k_key] = v[nearest_it_REPT, k_idx]
            else:
                 rept_epoch_data[k_key] = v[nearest_it_REPT]
        else: 
            rept_epoch_data[k_key] = v[nearest_it_REPT]
    
    e_val = REPT_energyofmualpha[REPT_sat_select][K][Mu].iloc[nearest_it_REPT]
    e_dict = {K: pd.DataFrame(e_val, index=[str(nearest_time_REPT)], columns=[Mu])}

    pad_model_rept = create_PAD(rept_epoch_data, QD_storm_data, e_dict, extMag = extMag)
    
    Model_PA = pad_model_rept[K][Mu]['pitch_angles'].values[0]
    Model_closest_PA_idx = np.argmin(np.abs(Model_PA - PA_local90))
    Model_PAD_vals = pad_model_rept[K][Mu]['Model'].values[0]
    Model_scale = PAD_data[len(PA_local)-1] / Model_PAD_vals[Model_closest_PA_idx]

    # 6. Collect GPS Data
    time_win = dt.timedelta(minutes=30)
    t_low = time_select - time_win
    t_high = time_select + time_win
    
    if gps_pad_models is None:
        gps_pad_models = {}
        for satellite, sat_data in gps_data.items():
            print(f"Modeling PAD for satellite {satellite}", end='\r')
            gps_pad_models[satellite] = create_PAD(sat_data, QD_storm_data, gps_energy[satellite], extMag)

    Model_GPS_PA = {}
    Model_GPS_PAD = {}
    near_time_idx = {}
    
    for sat, dat in gps_data.items():
        mask = (dat['Epoch'].UTC >= t_low) & (dat['Epoch'].UTC <= t_high)
        idxs = np.where(mask)[0]
        
        Model_GPS_PA[sat] = []
        Model_GPS_PAD[sat] = []
        near_time_idx[sat] = []
        
        for idx in idxs:
            curr_E = gps_energy[sat][K][Mu].iloc[idx]
            curr_L = dat[f'L_LGM_{extMag_label}IGRF'][idx]
            curr_MLT = np.atleast_1d(dat['MLT'])[idx]
            
            L_idx = np.argmin(np.abs(L_bins - curr_L))
            MLT_idx = int(((curr_MLT + 1) % 24) // 2)
            E_match = np.argmin(np.abs(REPT_data[REPT_sat_select]['Energy_Channels'][0:6] - curr_E)) == i_energy
            
            if E_match and (L_idx == L_ibin) and (MLT_idx == MLT_ibin):
                near_time_idx[sat].append(idx)
                
                pa = gps_pad_models[sat][K][Mu]['pitch_angles'].values[idx, :]
                mod = gps_pad_models[sat][K][Mu]['Model'].values[idx, :]
                
                Model_GPS_PA[sat].append(pa)
                Model_GPS_PAD[sat].append(mod)
                
        if len(Model_GPS_PA[sat]) > 0:
            Model_GPS_PA[sat] = np.array(Model_GPS_PA[sat])
            Model_GPS_PAD[sat] = np.array(Model_GPS_PAD[sat])
        else:
            del Model_GPS_PA[sat]
            del Model_GPS_PAD[sat]

    if not Model_GPS_PAD:
        print(f"No matching GPS satellites found in MLT={MLT_bin}, L={L_bin}")

    # 7. Create Plot
    fig, ax = plt.subplots(figsize=(9, 9))
    
    ax.scatter(PA_eq[PAD_data > 0], PAD_data[PAD_data > 0], label=rbsp_label, 
               zorder=3, color='black', marker='+', s=200)
    
    ax.plot(Model_PA, Model_PAD_vals * Model_scale, label='RBSP Model', 
            zorder=2, linewidth=4, alpha=0.7, linestyle='solid')
    
    for sat, pads in Model_GPS_PAD.items():
        for i, idx in enumerate(near_time_idx[sat]):
            scale = gps_flux[sat][K][Mu].values[idx]
            l_plot = ax.plot(Model_GPS_PA[sat][i], pads[i] * scale, label=sat, 
                    zorder=1, alpha=0.7, linewidth=3, linestyle='dotted')
            
            col = l_plot[0].get_color()
            loss = gps_data[sat]['loss_cone'][idx]
            loc90 = gps_data[sat]['local90PA'][idx]
            alpha_val = gps_alpha[sat][K].iloc[idx]
            
            for val, sty in [(loss, '-.'), (loc90, '-'), (alpha_val, '--')]:
                ax.vlines(val, 0, 1e8, color=col, linestyle=sty)
                ax.vlines(180-val, 0, 1e8, color=col, linestyle=sty)

    ax.text(0.54, 0.96, r"K = " + f"{K:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{Mu:.0f}" + r" $MeV/G$", 
            transform=ax.transAxes, fontsize=textsize)

    gray = [0.6, 0.6, 0.6]
    h_loss = mlines.Line2D([], [], color=gray, linestyle='-.', linewidth=2, label='GPS Loss Cone')
    h_90 = mlines.Line2D([], [], color=gray, linestyle='-', linewidth=2, label='GPS Local 90')
    h_alpha = mlines.Line2D([], [], color=gray, linestyle='--', linewidth=2, label=r'PA at K=' + f'{K:.1f}')
    
    exist_h, exist_l = ax.get_legend_handles_labels()
    final_h = exist_h + [h_loss, h_90, h_alpha]
    final_l = exist_l + [h_loss.get_label(), h_90.get_label(), h_alpha.get_label()]
    
    ax.legend(handles=final_h, labels=final_l, fontsize=textsize-4, loc='lower center')
    
    ax.set_xlim(0, 180)
    y_min = np.floor(np.log10(np.nanmin(Model_PAD_vals * Model_scale)))
    y_max = np.ceil(np.log10(np.nanmax(Model_PAD_vals * Model_scale)))
    ax.set_ylim(10**y_min, 10**y_max)
    
    plt.yscale('log')
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlabel(r"Equatorial Pitch Angle (degrees)", fontsize=textsize)
    ax.set_ylabel(r'Directional Flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)', fontsize=textsize)
    ax.grid(True)
    ax.set_title(f"Time: {time_select.strftime('%Y-%m-%d %H:%M')}", fontsize=textsize)
    
    plt.show()

#%% Plot PSD Radial Profiles with Averaging (L* Binned)
def plot_radial_profile_static(gps_data, REPT_data, 
                                time_start, time_stop, 
                                gps_time_start=None, gps_time_stop=None, SHOW_GPS_DATA=True,
                                REPT_sat_select='rbspb',K=0.1, Mu=2000, 
                                MLT_range=12, lstar_delta=0.1, time_delta=30, 
                                min_val = 1e-9, max_val = 1e-5, textsize=16):
    """
    Generates a static PSD Radial Profile plot with L* binning and REPT background.
    
    Args:
        gps_data (dict): Dictionary containing processed GPS satellite data.
        REPT_data (dict): Dictionary containing processed REPT satellite data.
        time_start (datetime): Start time for the plot axis.
        time_stop (datetime): End time for the plot axis.
        gps_time_start (datetime, optional): Start time for collecting GPS profiles. Defaults to time_start.
        gps_time_stop (datetime, optional): End time for collecting GPS profiles. Defaults to time_stop.
        REPT_sat_select (str, optional): REPT satellite to use ('rbspa' or 'rbspb'). Defaults to 'rbspa'.
        K (float, optional): Second adiabatic invariant. Defaults to 0.1.
        Mu (float, optional): First adiabatic invariant. Defaults to 2000.
        MLT_range (float, optional): MLT acceptance window width. Defaults to 12.
        lstar_delta (float, optional): L* bin width. Defaults to 0.1.
        time_delta (int, optional): GPS time integration window in minutes. Defaults to 30.
        min_val (float, optional): Minimum PSD value for colormap. Defaults to 1e-9.
        max_val (float, optional): Maximum PSD value for colormap. Defaults to 1e-5.
        textsize (int, optional): Base font size. Defaults to 16.
    """
    
    # Handle default arguments
    if gps_time_start is None: gps_time_start = time_start
    if gps_time_stop is None: gps_time_stop = time_stop
    
    # 1. Setup Parameters & Indices
    try:
        # Get K index from REPT data structure
        K_set = np.array(list(REPT_data[REPT_sat_select]['PSD'].keys()), dtype=float)
        i_K = np.where(np.isclose(K_set, K))[0][0]
        actual_K = list(REPT_data[REPT_sat_select]['PSD'].keys())[i_K] # Handle potential float/string keys

        # Get Mu index from REPT data structure
        # Assuming REPT_data[sat]['PSD'][K] is DataFrame with Mu columns
        Mu_set = np.array(list(REPT_data[REPT_sat_select]['PSD'][actual_K].columns), dtype=float)
        i_mu = np.where(np.isclose(Mu_set, Mu))[0][0]

    except IndexError:
        print(f"Requested K={K} or Mu={Mu} not found in REPT data.")
        return
    except KeyError:
        print(f"Satellite {REPT_sat_select} not found in REPT data.")
        return

    # Generate Time Steps
    time_intervals_GPS = np.arange(gps_time_start, gps_time_stop + dt.timedelta(minutes=time_delta), 
                                   dt.timedelta(minutes=time_delta)).astype(dt.datetime)

    # 4. Collect GPS Data
    temp_data = []
    for satellite, sat_data in gps_data.items():
        # Apply Time Mask
        window_start = time_start - dt.timedelta(minutes=time_delta-1)
        window_stop = time_stop + dt.timedelta(minutes=time_delta-1)
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= window_start) & (sat_data['Epoch'].UTC <= window_stop)

        # Extract Data
        sat_epoch = sat_data['Epoch'].UTC[sat_iepoch_mask]
        sat_MLT = sat_data['MLT'][sat_iepoch_mask]
        sat_Lstar = sat_data['Lstar'][sat_iepoch_mask, i_K].flatten()
        
        # Check if PSD exists in gps_data structure
        if 'PSD' in sat_data and actual_K in sat_data['PSD']:
             sat_PSD = sat_data['PSD'][actual_K].values[sat_iepoch_mask, i_mu].flatten()
        else:
             # Skip satellite if PSD not found
             continue

        sat_name_array = np.full(len(sat_epoch), satellite, dtype='<U10')

        # Combine arrays
        valid_mask = ~np.isnan(sat_Lstar) & ~np.isnan(sat_PSD)
        
        if np.sum(valid_mask) > 0:
            combined_satellite_data = np.vstack((
                sat_epoch[valid_mask],
                sat_name_array[valid_mask],
                sat_Lstar[valid_mask],
                sat_MLT[valid_mask],
                sat_PSD[valid_mask]
            )).T
            temp_data.append(combined_satellite_data)

    if not temp_data:
        print("No valid GPS data found in the specified window.")
        return

    GPS_plot_data = np.concatenate(temp_data, axis=0)
    GPS_plot_data = GPS_plot_data[GPS_plot_data[:, 0].argsort()]

    # 5. MLT Filtering against REPT
    nearest_time = np.zeros(len(GPS_plot_data), dtype=int)
    MLT_mask = np.zeros(len(GPS_plot_data), dtype=bool)
    
    rept_epochs = REPT_data[REPT_sat_select]['Epoch'].UTC
    
    for i_epoch, epoch in enumerate(GPS_plot_data[:,0]):
        nearest_time[i_epoch] = np.argmin(np.abs(rept_epochs - epoch))
        MLT_ref = REPT_data[REPT_sat_select]['MLT'][nearest_time[i_epoch]]
        MLT_gps = GPS_plot_data[i_epoch, 3]
        
        mlt_diff = np.minimum(np.abs(MLT_ref - MLT_gps), 24 - np.abs(MLT_ref - MLT_gps))
        MLT_mask[i_epoch] = (mlt_diff <= MLT_range/2)

    # 6. Plotting Setup (RBSP Background)
    Epoch_np = np.array(REPT_data[REPT_sat_select]['Epoch'].UTC)
    time_mask_REPT = (Epoch_np >= time_start) & (Epoch_np <= time_stop)
    time_range_REPT = Epoch_np[time_mask_REPT]
    
    time_range_num = mdates.date2num(time_range_REPT)
    sort_indices = np.argsort(time_range_num)
    time_range_num_sorted = time_range_num[sort_indices]

    lstar_range = REPT_data[REPT_sat_select]['Lstar'][time_mask_REPT, i_K].flatten()
    psd_range = REPT_data[REPT_sat_select]['PSD'][actual_K].values[:, i_mu].flatten()[time_mask_REPT]
    
    lstar_range_sorted = lstar_range[sort_indices]
    psd_range_sorted = psd_range[sort_indices]

    # Calculate binned averages for GPS lines
    valid_l = lstar_range[lstar_range > 0]
    lstar_min = np.nanmin(valid_l) if len(valid_l) > 0 else 3
    lstar_max = np.nanmax(valid_l) if len(valid_l) > 0 else 6
    
    lstar_intervals = np.arange(np.floor(lstar_min/lstar_delta)*lstar_delta, 
                                np.ceil(lstar_max/lstar_delta)*lstar_delta + lstar_delta, 
                                lstar_delta)

    avg_psd = np.zeros((len(time_intervals_GPS), len(lstar_intervals))) * np.nan
    
    if SHOW_GPS_DATA:
        for i_time, time_int in enumerate(time_intervals_GPS):
            t_start = time_int - dt.timedelta(minutes=time_delta/2)
            t_end = time_int + dt.timedelta(minutes=time_delta/2)
            
            time_mask_GPS = (GPS_plot_data[:,0] >= t_start) & (GPS_plot_data[:,0] < t_end)
            
            for i_lstar, lstar_val in enumerate(lstar_intervals):
                lstar_mask = (GPS_plot_data[:,2] >= (lstar_val - lstar_delta/2)) & \
                            (GPS_plot_data[:,2] < (lstar_val + lstar_delta/2))
                
                combined_mask = time_mask_GPS & lstar_mask & MLT_mask
                
                if np.sum(combined_mask) > 1:
                    psd_data = GPS_plot_data[combined_mask, 4].astype(float)
                    valid_psd = psd_data[(~np.isnan(psd_data)) & (psd_data > min_val)]
                    if len(valid_psd) > 0:
                        avg_psd[i_time, i_lstar] = np.nanmean(valid_psd)

    # 7. Create Figure
    fig, ax = plt.subplots(figsize=(24, 10))
    
    colormap_name = 'plasma'
    plasma = plt.cm.get_cmap(colormap_name)
    cmap = colors.LinearSegmentedColormap.from_list('truncated_plasma', plasma(np.linspace(0, 0.9, 256)))
    
    vmin = mdates.date2num(time_start)
    vmax = mdates.date2num(time_stop)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Background REPT
    scatter_plot = ax.scatter(lstar_range_sorted, psd_range_sorted, 
                              c=time_range_num_sorted, cmap=cmap, norm=norm, 
                              marker='o', s=20, alpha=0.3)

    # Colorbar
    cbar = fig.colorbar(scatter_plot, ax=ax, orientation='vertical', pad=0.04)
    cbar.solids.set_alpha(1)
    cbar.set_label('Time (UTC)', fontsize=textsize+2, labelpad=20)
    cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    cbar.ax.tick_params(labelsize=textsize)

    # GPS Lines
    for i_time, time_int in enumerate(time_intervals_GPS):
        if np.sum(~np.isnan(avg_psd[i_time,:]) & (avg_psd[i_time,:] > min_val)) > 0:
            range_mask = ~np.isnan(avg_psd[i_time,:]) & (avg_psd[i_time,:] > min_val)
            
            ax.plot(lstar_intervals[range_mask], avg_psd[i_time, range_mask],
                    marker='*', markersize=16,
                    color=cmap(norm(mdates.date2num(time_int))),
                    label=time_int.strftime("%d-%m-%Y %H:%M"))
            
            yoff = -5 if i_time == len(time_intervals_GPS)-1 else -10
            
            ax.annotate(time_int.strftime("%H:%M"), 
                        (lstar_intervals[~np.isnan(avg_psd[i_time,:])][-1], 
                         avg_psd[i_time,:][~np.isnan(avg_psd[i_time,:])][-1]), 
                        xytext=(10, yoff),
                        textcoords='offset points',
                        fontsize=textsize+2, 
                        color=cmap(norm(mdates.date2num(time_int))),
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlim(4, 5.3)
    ax.set_xlabel(r"L*", fontsize=textsize+2, labelpad=10)
    ax.set_ylim(min_val, max_val)
    ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize+2)
    plt.yscale('log')
    ax.grid(True)

    ax.text(0.02, 0.98, r"K = " + f"{K:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{Mu:.0f}" + r" $MeV/G$", 
            transform=ax.transAxes, fontsize=textsize+4, verticalalignment='top')

    if REPT_sat_select == 'rbspa': rbsp_label = 'RBSP-A'
    else: rbsp_label = 'RBSP-B'
    
    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label=rbsp_label) 
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None', markersize=12, label='GPS') 
    
    ax.legend(handles=[handle_rbsp, handle_gps], title='Satellite', title_fontsize=textsize,
              loc='lower right', bbox_to_anchor=(1.0, 0), handlelength=1, fontsize=textsize-2)

    title_str = f"Time Interval: {time_start.strftime('%Y-%m-%d %H:%M')} to {time_stop.strftime('%Y-%m-%d %H:%M')}"
    ax.set_title(title_str, fontsize=textsize+10)
    plt.show()

#%% Plot PSD Radial Profiles with Averaging (L* Binned)
def plot_radial_profile_dynamic(gps_data, REPT_data, 
                                time_start, time_stop, 
                                gps_time_start=None, gps_time_stop=None, SHOW_GPS_DATA=True,
                                REPT_sat_select='rbspb', K=0.1, Mu=2000, 
                                anim_name='test', 
                                MLT_range=12, lstar_delta=0.1, time_delta=30, 
                                min_val=1e-9, max_val=1e-5, textsize=16):
    """
    Generates an animated MP4 of the PSD Radial Profile.
    Plots REPT data cumulatively and overlays GPS radial profiles.
    
    Args:
        gps_data (dict): Dictionary containing processed GPS satellite data (must include 'PSD').
        REPT_data (dict): Dictionary containing processed REPT satellite data.
        time_start (datetime): Start time for the plot axis and animation.
        time_stop (datetime): End time for the plot axis and animation.
        gps_time_start (datetime, optional): Start time for collecting GPS profiles. Defaults to time_start.
        gps_time_stop (datetime, optional): End time for collecting GPS profiles. Defaults to time_stop.
        SHOW_GPS_DATA (bool, optional): Toggle to show/hide GPS overlays. Defaults to True.
        REPT_sat_select (str, optional): REPT satellite ('rbspa' or 'rbspb'). Defaults to 'rbspb'.
        K (float, optional): Second adiabatic invariant. Defaults to 0.1.
        Mu (float, optional): First adiabatic invariant. Defaults to 2000.
        anim_name (str, optional): Filename for the output MP4. Defaults to 'test'.
        MLT_range (float, optional): MLT acceptance window. Defaults to 12.
        lstar_delta (float, optional): L* bin width. Defaults to 0.1.
        time_delta (int, optional): GPS time integration window in minutes. Defaults to 30.
        min_val (float, optional): Minimum PSD value for colormap. Defaults to 1e-9.
        max_val (float, optional): Maximum PSD value for colormap. Defaults to 1e-5.
        textsize (int, optional): Base font size. Defaults to 16.
    """
    
    # Handle default arguments for GPS collection window
    if gps_time_start is None: gps_time_start = time_start
    if gps_time_stop is None: gps_time_stop = time_stop

    # 1. Setup Indices (K/Mu) from REPT Data Structure
    try:
        # Get K index
        K_keys = np.array(list(REPT_data[REPT_sat_select]['PSD'].keys()), dtype=float)
        i_K = np.where(np.isclose(K_keys, K))[0][0]
        actual_K = list(REPT_data[REPT_sat_select]['PSD'].keys())[i_K]

        # Get Mu index
        Mu_keys = np.array(list(REPT_data[REPT_sat_select]['PSD'][actual_K].columns), dtype=float)
        i_mu = np.where(np.isclose(Mu_keys, Mu))[0][0]
        actual_Mu = list(REPT_data[REPT_sat_select]['PSD'][actual_K].columns)[i_mu]
    except (IndexError, KeyError):
        print(f"Error: Requested K={K} or Mu={Mu} not found in REPT data for {REPT_sat_select}.")
        return

    gps_scale = 1 

    # Generate Time Steps for GPS Lines
    time_intervals_GPS = np.arange(gps_time_start, gps_time_stop + dt.timedelta(minutes=time_delta), 
                                   dt.timedelta(minutes=time_delta)).astype(dt.datetime)

    # 2. Collect and Filter GPS Data
    temp_data = []
    
    # Pre-calculate Window for GPS data selection (buffer included)
    window_start = time_start - dt.timedelta(minutes=time_delta)
    window_stop = time_stop + dt.timedelta(minutes=time_delta)

    for satellite, sat_data in gps_data.items():
        # Check if PSD exists for this satellite
        if 'PSD' not in sat_data or actual_K not in sat_data['PSD']:
            continue
            
        sat_iepoch_mask = (sat_data['Epoch'].UTC >= window_start) & (sat_data['Epoch'].UTC <= window_stop)
        
        sat_epoch = sat_data['Epoch'].UTC[sat_iepoch_mask]
        sat_MLT = sat_data['MLT'][sat_iepoch_mask]
        sat_Lstar = sat_data['Lstar'][sat_iepoch_mask, i_K].flatten()
        sat_PSD = sat_data['PSD'][actual_K].values[sat_iepoch_mask, i_mu].flatten()
        
        sat_name_array = np.full(len(sat_epoch), satellite, dtype='<U10')

        valid_mask = ~np.isnan(sat_Lstar) & ~np.isnan(sat_PSD)
        
        if np.sum(valid_mask) > 0:
            combined_satellite_data = np.vstack((
                sat_epoch[valid_mask],
                sat_name_array[valid_mask],
                sat_Lstar[valid_mask],
                sat_MLT[valid_mask],
                sat_PSD[valid_mask]
            )).T
            temp_data.append(combined_satellite_data)

    if not temp_data:
        print("No valid GPS data found in time range.")
        return

    GPS_plot_data = np.concatenate(temp_data, axis=0)
    GPS_plot_data = GPS_plot_data[GPS_plot_data[:, 0].argsort()]

    # 3. MLT Filtering against REPT
    # Find nearest REPT time for each GPS point to compare MLT
    MLT_mask = np.zeros(len(GPS_plot_data), dtype=bool)
    
    rept_epochs = REPT_data[REPT_sat_select]['Epoch'].UTC
    
    for i_epoch, epoch in enumerate(GPS_plot_data[:,0]):
        # Find closest REPT epoch
        nearest_idx = np.argmin(np.abs(rept_epochs - epoch))
        MLT_ref = REPT_data[REPT_sat_select]['MLT'][nearest_idx]
        MLT_gps = GPS_plot_data[i_epoch, 3] # Index 3 is MLT
        
        # Calculate circular MLT difference
        mlt_diff = np.minimum(np.abs(MLT_ref - MLT_gps), 24 - np.abs(MLT_ref - MLT_gps))
        MLT_mask[i_epoch] = (mlt_diff <= MLT_range/2)

    # 4. Prepare REPT Background Data for Animation
    # We sort by time so the scatter plot can build up cumulatively
    Epoch_np = np.array(REPT_data[REPT_sat_select]['Epoch'].UTC)
    time_mask_REPT = (Epoch_np >= time_start) & (Epoch_np <= time_stop)
    time_range_REPT = Epoch_np[time_mask_REPT]
    
    time_range_num = mdates.date2num(time_range_REPT)
    sort_indices = np.argsort(time_range_num)
    
    time_range_REPT_sorted = time_range_REPT[sort_indices]
    time_range_num_sorted = time_range_num[sort_indices]

    lstar_range = REPT_data[REPT_sat_select]['Lstar'][time_mask_REPT, i_K].flatten()
    psd_range = REPT_data[REPT_sat_select]['PSD'][actual_K].values[:, i_mu].flatten()[time_mask_REPT]
    
    lstar_range_sorted = lstar_range[sort_indices]
    psd_range_sorted = psd_range[sort_indices]

    # 5. Pre-calculate Averaged GPS Radial Profiles
    # Define L* bins based on REPT data range
    valid_l = lstar_range[lstar_range > 0]
    lstar_min = np.nanmin(valid_l) if len(valid_l) > 0 else 3.0
    lstar_max = np.nanmax(valid_l) if len(valid_l) > 0 else 6.0
    
    lstar_intervals = np.arange(np.floor(lstar_min/lstar_delta)*lstar_delta, 
                                np.ceil(lstar_max/lstar_delta)*lstar_delta + lstar_delta, 
                                lstar_delta)

    avg_psd = np.zeros((len(time_intervals_GPS), len(lstar_intervals))) * np.nan
    
    for i_time, time_int in enumerate(time_intervals_GPS):
        t_start = time_int - dt.timedelta(minutes=time_delta/2)
        t_end = time_int + dt.timedelta(minutes=time_delta/2)
        
        # Filter GPS data for this snapshot window
        time_mask_GPS = (GPS_plot_data[:,0] >= t_start) & (GPS_plot_data[:,0] < t_end)
        
        for i_lstar, lstar_val in enumerate(lstar_intervals):
            lstar_mask = (GPS_plot_data[:,2] >= (lstar_val - lstar_delta/2)) & \
                         (GPS_plot_data[:,2] < (lstar_val + lstar_delta/2))
            
            combined_mask = time_mask_GPS & lstar_mask & MLT_mask
            
            if np.sum(combined_mask) > 1:
                # Extract PSD (Index 4), force float
                psd_vals = GPS_plot_data[combined_mask, 4].astype(float)
                valid_psd = psd_vals[(~np.isnan(psd_vals)) & (psd_vals > min_val)]
                
                if len(valid_psd) > 0:
                    avg_psd[i_time, i_lstar] = np.nanmean(valid_psd) * gps_scale

    # 6. Initialize Figure and Plotting Elements
    fig, ax = plt.subplots(figsize=(24, 8), dpi=100)
    
    # Setup Colormap
    colormap_name = 'plasma'
    plasma = plt.cm.get_cmap(colormap_name)
    cmap = colors.LinearSegmentedColormap.from_list('truncated_plasma', plasma(np.linspace(0, 0.9, 256)))
    
    vmin = mdates.date2num(time_start)
    vmax = mdates.date2num(time_stop)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Initialize Empty Scatter Plot (to be updated in animation)
    scatter_plot = ax.scatter([], [], c=[], cmap=cmap, norm=norm, marker='o', s=30, alpha=0.3)
    scatter_plot.set_clim(vmin, vmax)

    # Pre-create GPS Artists (Lines and Annotations) - Initially Invisible
    gps_artists = []
    
    if SHOW_GPS_DATA:
        for i_time, time_int in enumerate(time_intervals_GPS):
            # Check if we have valid data to plot for this time
            has_data = np.sum(~np.isnan(avg_psd[i_time,:]) & (avg_psd[i_time,:] > min_val)) > 0
            
            if has_data:
                range_mask = ~np.isnan(avg_psd[i_time,:]) & (avg_psd[i_time,:] > min_val)
                x_vals = lstar_intervals[range_mask]
                y_vals = avg_psd[i_time, range_mask]
                
                # Create Line
                line, = ax.plot(x_vals, y_vals, marker='*', markersize=16,
                                color=cmap(norm(mdates.date2num(time_int))),
                                label=time_int.strftime("%d-%m-%Y %H:%M"),
                                visible=False)
                
                # Create Annotation
                yoff = -5 if i_time == len(time_intervals_GPS)-1 else -10
                
                ann = ax.annotate(time_int.strftime("%H:%M"), 
                            (x_vals[-1], y_vals[-1]), 
                            xytext=(10, yoff), textcoords='offset points',
                            fontsize=textsize+2, color=cmap(norm(mdates.date2num(time_int))),
                            fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
                            visible=False)
                
                # Store artists and their trigger time
                gps_artists.append({
                    'trigger_time': time_int,
                    'line': line,
                    'annotation': ann,
                    'shown': False
                })

    # Colorbar Setup
    cbar = fig.colorbar(scatter_plot, ax=ax, orientation='vertical', pad=0.04)
    cbar.solids.set_alpha(1)
    cbar.set_label('Time (UTC)', fontsize=textsize+2, labelpad=20)
    cbar.ax.yaxis.set_major_locator(mdates.HourLocator(interval=2))
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    cbar.ax.tick_params(labelsize=textsize)
    
    # Progress Line on Colorbar
    cbar_line = cbar.ax.axhline(vmin, color='white', lw=3)

    # Axis Formatting
    ax.tick_params(axis='both', labelsize=textsize, pad=10)
    ax.set_xlim(4, 5.3) 
    ax.set_xlabel(r"L*", fontsize=textsize+2, labelpad=10)
    ax.set_ylim(min_val, max_val)
    ax.set_ylabel(r"PSD $[(c/MeV/cm)^3]$", fontsize=textsize+2)
    plt.yscale('log')
    ax.grid(True)
    
    ax.text(0.02, 0.98, r"K = " + f"{K:.1f} " + r"$G^{{1/2}}R_E$, $\mu = $" + f"{Mu:.0f}" + r" $MeV/G$", 
            transform=ax.transAxes, fontsize=textsize+4, verticalalignment='top')
    
    if REPT_sat_select == 'rbspa': rbsp_label = 'RBSP-A'
    else: rbsp_label = 'RBSP-B'
    
    handle_rbsp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label=rbsp_label) 
    handle_gps = mlines.Line2D([], [], color='gray', marker='*', linestyle='None', markersize=12, label='GPS') 
    
    ax.legend(handles=[handle_rbsp, handle_gps], title='Satellite', title_fontsize=textsize, 
              loc='lower right', bbox_to_anchor=(1.0, 0), handlelength=1, fontsize=textsize-2)

    plt.tight_layout()

    # 7. Animation Update Function
    def update(frame):
        # Update Background Scatter (Cumulative)
        current_time_val = time_range_num_sorted[frame]
        current_x = lstar_range_sorted[:frame+1]
        current_y = psd_range_sorted[:frame+1]
        current_c = time_range_num_sorted[:frame+1]
        
        if len(current_x) > 0:
            pos_data = np.column_stack((current_x, current_y))
            scatter_plot.set_offsets(pos_data)
            scatter_plot.set_array(current_c)

        # Update Colorbar Indicator
        cbar_line.set_ydata([current_time_val, current_time_val])
        
        # Update Title
        current_time_dt = time_range_REPT_sorted[frame]
        # title_str = f"Time: {current_time_dt.strftime('%Y-%m-%d %H:%M')}"
        # ax.set_title(title_str, fontsize=textsize+10)

        # Toggle GPS Lines based on Time
        for item in gps_artists:
            if not item['shown'] and current_time_dt >= item['trigger_time']:
                item['line'].set_visible(True)
                item['annotation'].set_visible(True)
                item['shown'] = True 
        
        # Return all dynamic artists for blitting
        all_dynamic_artists = [scatter_plot, cbar_line]
        for item in gps_artists:
            all_dynamic_artists.append(item['line'])
            all_dynamic_artists.append(item['annotation'])
            
        return all_dynamic_artists

    # 8. Render and Save
    ani = animation.FuncAnimation(fig, update, frames=len(time_range_num_sorted), interval=20, blit=True)
    
    print(f"Saving Animation as {anim_name}.mp4...")
    try:
        ani.save(f'{anim_name}.mp4', writer='ffmpeg', fps=30, dpi=100)
        print(f"Success! Saved as {anim_name}.mp4")
    except Exception as e:
        print(f"Error saving animation: {e}")

    plt.close(fig)