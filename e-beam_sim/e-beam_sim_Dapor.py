#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt

import my_constants as mc
import my_utilities as mu
import MC_functions_Dapor_new as mcf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
mcf = importlib.reload(mcf)

os.chdir(os.path.join(mc.sim_folder, 'e-beam_sim'))

import plot_data as pd
pd = importlib.reload(pd)


#%%
#E0_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800,\
#           850, 900, 950, 1000]

E0 = 250

n_files = 1000
n_tracks = 100

num = 0


while num < n_files:
    
    DATA = mcf.get_DATA(E0, n_tracks)
    
    inds = np.where(np.logical_and(DATA[:, 5] < 0, DATA[:, 7] < 50))
    n_2ndaries = len(inds[0])
    
    fname = '../e_DATA/Secondaries/250/DATA_' + str(num) + '.npy'
    
    np.save(fname, np.array((n_tracks, n_2ndaries)))
    
    print('file ' + fname + ' is ready')

    num += 1


#%%
pd.plot_DATA(DATA, 3, 5)

