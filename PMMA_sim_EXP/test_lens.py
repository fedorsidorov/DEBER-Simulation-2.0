#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib

import my_constants as mc
import my_utilities as mu
import chain_functions as cf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
cf = importlib.reload(cf)

import time

os.chdir(mc.sim_folder + 'PMMA_sim')


#%%
m = np.load('harris_x_before.npy')
mw = np.load('harris_y_before_SZ.npy')


sample = np.zeros(1000000)

for i in range(len(sample)):
    
    sample[i] = cf.get_chain_len(m, mw)
    

#%%
xx = np.load('harris_x_before.npy')
yy = np.load('harris_y_before_SZ.npy')

plt.semilogx(xx, yy / np.max(yy), label='Schulz-Zimm')


#lens = np.load('Harris_lens_arr.npy')
lens = sample
mass = lens*100

bins = np.logspace(2, 7.1, 51)
#bins = np.linspace(1e+2, 12e+6, 101)

hist, edges = np.histogram(mass, bins)

plt.semilogx(edges[:-1], hist/np.max(hist), label='sample')

plt.title('Harris initial molecular weight distribution')
plt.xlabel('molecular weight')
plt.ylabel('density')

plt.legend()
plt.grid()
plt.show()




#%%
source_dir = '/Volumes/ELEMENTS/Chains_Harris'

f_names = os.listdir(source_dir)


#%%
diams = np.zeros(len(f_names))

i = 0

for chain in f_names:
    
    if 'chain' not in chain:
        continue
    
    mu.upd_progress_bar(i, len(f_names))
    
    now_chain = np.load(source_dir + '/' + chain)
    
    c_max = np.max(now_chain, axis=0)
    c_min = np.min(now_chain, axis=0)
    
    diams[i] = np.max(c_max - c_min)
    
    i += 1

