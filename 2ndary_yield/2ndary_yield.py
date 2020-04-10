#%% Import
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder, '2ndary_yield'))


#%%
source = os.path.join(mc.sim_folder, 'e_DATA', 'Secondaries', '250')

n_files = 64

n_total = 0
n_2nd = 0


for i in range(n_files):
    
    mu.pbar(i, n_files)
    
    DATA = np.load(os.path.join(source, 'n_2nd_for_100_prim_tracks_' + str(i) + '.npy'))
    
    n_total += 100
    n_2nd += DATA


print(n_2nd/n_total)
    

#%%
E0 = 250
Em = 307
dm = 2.16

d = dm*1.2*(E0/Em)**(-0.67) * (1-np.exp(-1.614 * (E0/Em)**1.67))

d

