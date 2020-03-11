#%% Import
import numpy as np
import os
import matplotlib.pyplot as plt
import importlib

import my_constants as mc
mc = importlib.reload(mc)

os.chdir(os.path.join(mc.sim_folder, 'SE_files'))


#%%
arr_y = np.linspace(0, 50, 51)
arr_z = np.cos(2*np.pi/50 * arr_y)

plt.plot(arr_y, arr_z)


#%% vertices
V = np.zeros(((len(arr_y)+2)*2, 1+3))


