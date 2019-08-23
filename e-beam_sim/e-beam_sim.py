#%% Import
import numpy as np
import os
import importlib

import MC_functions as mcf
import my_constants as mc
import plot_data as pd

mc = importlib.reload(mc)
mcf = importlib.reload(mcf)
pd = importlib.reload(pd)

os.chdir(mc.sim_path_MAC + 'e-beam_sim')


#%%
## Usual
n_files = 1
n_tracks = 1

d_PMMA = 100e-7
E0 = 20e+3

num = 0

while num < n_files:
    
    DATA = mcf.get_DATA(d_PMMA, E0, n_tracks)
    
#    fname = '../e_DATA/DATA_Pn_20keV_300nm/DATA_Pn_' + str(num) + '.npy'
#    np.save(fname, DATA)
    
#    print('file ' + fname + ' is ready')

    num += 1

#%%
#DATA = np.load('../e_DATA/DATA_Pn_20keV_300nm/DATA_Pn_0.npy')

#%%
pd.plot_DATA(DATA, d_PMMA)

#%%
#x, y, z = DATA[:, 5], DATA[:, 6], DATA[:, 7]

#%%


