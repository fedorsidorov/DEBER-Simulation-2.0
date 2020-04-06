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

n = 1000

e_total = n * 100
e_2ndaries = 0


for i in range(1000):
    
    mu.pbar(i, n)
    
    DATA = np.load(os.path.join(source, 'DATA_' + str(i) + '.npy'))
    
    inds = np.where(np.logical_and(DATA[:, 3] < 50, DATA[:, 6] < 0))
    
    e_2ndaries += len(inds[0])


print(e_2ndaries/e_total)
    

#%%
E0 = 250
Em = 307
dm = 2.16

d = dm*1.2*(E0/Em)**(-0.67) * (1-np.exp(-1.614 * (E0/Em)**1.67))

d

