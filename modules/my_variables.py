#%%
import numpy as np

#%%
sim_path_MAC = '/Users/fedor/Documents/DEBER-Simulation-2.0/'
#sim_path_FTIAN = '/home/fedor/Yandex.Disk/Study/Simulation/'

EE = np.logspace(0, 4.4, 1000)

THETA_deg = np.linspace(0.1, 180, 1000)
THETA = np.deg2rad(THETA_deg)

WW = np.logspace(0, 4.4, 3000)
WW_ext = np.logspace(-4, 4.4, 5000)

