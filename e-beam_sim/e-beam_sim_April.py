#%% Import
import numpy as np
import os
import importlib
# import matplotlib.pyplot as plt

import my_constants as mc
import my_utilities as mu
import MC_functions_April as mcf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
mcf = importlib.reload(mcf)

os.chdir(os.path.join(mc.sim_folder, 'e-beam_sim'))

import plot_data as pd
pd = importlib.reload(pd)


#%%
E0 = 10e+3
d_PMMA = 500e-7
# d_PMMA = 1


# n_files = 1000
# n_tracks = 100

n_files = 1
n_tracks = 10

num = 0


while num < n_files:
    
    DATA = mcf.get_DATA(d_PMMA, E0, n_tracks)
    
    DATA_PMMA = DATA[np.where(DATA[:, 2] == 0)]
    DATA_PMMA_inel = DATA_PMMA[np.where(DATA_PMMA[:, 3] != 0)]
    
    dest_dir = os.path.join(mc.sim_folder, 'e_DATA', 'Harris_April')
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    fname_PMMA_inel = os.path.join(dest_dir, 'DATA_PMMA_inel_' + str(num) + '.npy')
    
    # np.save(fname_PMMA_inel, DATA_PMMA_inel)
    
    print('file ' + fname_PMMA_inel + ' is ready')    
    
    num += 1


#%%
pd.plot_DATA(DATA, 4, 6, d_PMMA*1e+7)

