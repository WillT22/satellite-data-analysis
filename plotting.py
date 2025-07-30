import numpy as np
import pandas as pd
import os
import sys
import matplotlib
import matplotlib.pyplot as plt

textsize = 16
Mu_set = np.array((4000, 6000, 8000, 10000, 12000, 14000, 16000)) # MeV/G
K_set = np.array((0.1,1)) # R_E*G^(1/2)

def load_data(npzfile):
    print(f"Loading {npzfile}")
    loaded_data = {}
    for satellite, sat_data in npzfile.items():
        loaded_data[satellite] = {}
        if isinstance(sat_data, np.ndarray):
            if sat_data.ndim == 0 and sat_data.dtype == object:
                temp_inner_dict = sat_data.item()
                for item, item_data in temp_inner_dict.items():
                    loaded_data[satellite][item] = item_data
            elif sat_data.ndim > 0:
                loaded_data[satellite] = sat_data
        elif isinstance(sat_data, pd.DataFrame):
            loaded_data[satellite] = sat_data
    print("Data Loaded \n")
    return loaded_data

base_save_folder = "/home/will/GPS_data/april2017storm/"
PAD_models_filename = f"PAD_models.npz"
PAD_models_save_path = os.path.join(base_save_folder, PAD_models_filename)
PAD_models_load = np.load(PAD_models_save_path, allow_pickle=True)
PAD_models = load_data(PAD_models_load)
PAD_models_load.close()

satellite = 'ns60'
k = 0.1
i_K = np.where(K_set == k)[0]
mu = 4000
i_mu = np.where(Mu_set == mu)[0]
i_epoch = 10

alpha_list = PAD_models['ns60'][k][mu]['pitch_angles'].values[i_epoch,:]

fig, ax = plt.subplots(figsize=(6, 1))
ax.plot(alpha_list, PAD_models[satellite][k][mu]['Model'].values[i_epoch,:], lw = 3)

ax.fill_between(alpha_list, PAD_models[satellite][k][mu]['Model'].values[i_epoch,:],
    y2=0, where=alpha_list <= 85, color='C0', alpha=0.4, zorder=1)
ax.fill_between(alpha_list, PAD_models[satellite][k][mu]['Model'].values[i_epoch,:],
    y2=0, where=alpha_list >= 95, color='C0', alpha=0.4, zorder=1)

ax.set_xlim(0, 180)
desired_xticks = np.arange(0, 181, 45) # From 0 to 180, step 45
ax.set_xticks(desired_xticks)
ax.tick_params(axis='x', labelrotation=90)
ax.set_xlabel('Pitch Angle')
ax.yaxis.set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

#%% Compare L* for April 2017 storm
base_save_folder = "/home/will/GPS_data/april2017storm/"
complete_filename = f"storm_data_T89c.npz"
complete_save_path = os.path.join(base_save_folder, complete_filename)
complete_load = np.load(complete_save_path, allow_pickle=True)
storm_data_complete = load_data(complete_load)
complete_load.close()

complete_filename = f"storm_data_TS04.npz"
complete_save_path = os.path.join(base_save_folder, complete_filename)
complete_load = np.load(complete_save_path, allow_pickle=True)
storm_data_TS04 = load_data(complete_load)
complete_load.close()

fig, ax = plt.subplots(figsize=(14, 3))
for satellite, sat_data in storm_data_complete.items():
    lstar_mask = sat_data['Lstar'][:,0] > -1e31
    ax.scatter(sat_data['Epoch'].UTC[lstar_mask], sat_data['Lstar'][lstar_mask,0])
ax.set_xlabel('UTC', fontsize=textsize)
ax.set_ylabel(r"L* T89", fontsize=textsize)
ax.tick_params(axis='both', which='major', labelsize=textsize)
ax.grid(True)

fig, ax = plt.subplots(figsize=(14, 3))
for satellite, sat_data in storm_data_TS04.items():
    lstar_mask = sat_data['Lstar'][:,0] > -1e31
    ax.scatter(sat_data['Epoch'].UTC[lstar_mask], sat_data['Lstar'][lstar_mask,0])
ax.set_xlabel('UTC', fontsize=textsize)
ax.set_ylabel(r"L* TS04", fontsize=textsize)
ax.tick_params(axis='both', which='major', labelsize=textsize)
ax.grid(True)

fig, ax = plt.subplots(figsize=(14, 6))
for satellite, sat_data in storm_data_TS04.items():
    lstar_mask = (sat_data['Lstar'][:,0] > -1e31) & (storm_data_complete[satellite]['Lstar'][:,0] > -1e31)
    ax.scatter(storm_data_complete[satellite]['Lstar'][lstar_mask,0], sat_data['Lstar'][lstar_mask,0])
ax.plot(np.linspace(0,8,1000),np.linspace(0,8,1000), color='black', linestyle = '--')
ax.set(xlim=(4.5,5.8),ylim=(4.8,5.7))
ax.set_xlabel(r"L* T89", fontsize=textsize)
ax.set_ylabel(r"L* TS04", fontsize=textsize)
ax.tick_params(axis='both', which='major', labelsize=textsize)
ax.set_aspect('equal')
ax.grid(True)

#%% Compare L* for August 2018 storm
base_save_folder = "/home/will/GPS_data/august2018storm/"
complete_filename = f"storm_data_T89c_aug2018.npz"
complete_save_path = os.path.join(base_save_folder, complete_filename)
complete_load = np.load(complete_save_path, allow_pickle=True)
storm_data_complete = load_data(complete_load)
complete_load.close()

complete_filename = f"storm_data_TS04_aug2018.npz"
complete_save_path = os.path.join(base_save_folder, complete_filename)
complete_load = np.load(complete_save_path, allow_pickle=True)
storm_data_TS04 = load_data(complete_load)
complete_load.close()

fig, ax = plt.subplots(figsize=(14, 3))
for satellite, sat_data in storm_data_complete.items():
    lstar_mask = sat_data['Lstar'][:,0] > -1e31
    ax.scatter(sat_data['Epoch'].UTC[lstar_mask], sat_data['Lstar'][lstar_mask,0])
ax.set_xlabel('UTC', fontsize=textsize)
ax.set_ylabel(r"L* T89", fontsize=textsize)
ax.tick_params(axis='both', which='major', labelsize=textsize)
ax.grid(True)

fig, ax = plt.subplots(figsize=(14, 3))
for satellite, sat_data in storm_data_TS04.items():
    lstar_mask = sat_data['Lstar'][:,0] > -1e31
    ax.scatter(sat_data['Epoch'].UTC[lstar_mask], sat_data['Lstar'][lstar_mask,0])
ax.set_xlabel('UTC', fontsize=textsize)
ax.set_ylabel(r"L* TS04", fontsize=textsize)
ax.tick_params(axis='both', which='major', labelsize=textsize)
ax.grid(True)

fig, ax = plt.subplots(figsize=(14, 6))
for satellite, sat_data in storm_data_TS04.items():
    lstar_mask = (sat_data['Lstar'][:,0] > -1e31) & (storm_data_complete[satellite]['Lstar'][:,0] > -1e31)
    ax.scatter(storm_data_complete[satellite]['Lstar'][lstar_mask,0], sat_data['Lstar'][lstar_mask,0])
ax.plot(np.linspace(0,8,1000),np.linspace(0,8,1000), color='black', linestyle = '--')
ax.set(xlim=(4.7,5.6),ylim=(4.7,5.2))
ax.set_xlabel(r"L* T89", fontsize=textsize)
ax.set_ylabel(r"L* TS04", fontsize=textsize)
ax.tick_params(axis='both', which='major', labelsize=textsize)
ax.set_aspect('equal')
ax.grid(True)
# %%
