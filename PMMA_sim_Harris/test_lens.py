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

os.chdir(mc.sim_folder + 'PMMA_sim_Harris')


#%%
m = np.load('harris_x_before.npy')
mw = np.load('harris_y_before_SZ.npy')


sample = np.zeros(1000000)

for i in range(len(sample)):
    
    sample[i] = cf.get_chain_len(m, mw)
    

#%%
fig, ax = plt.subplots()

xx = m
yy = mw

mass = sample*100

bins = np.logspace(2, 7.1, 21)

plt.hist(mass, bins, label='simulation')
plt.gca().set_xscale('log')

plt.plot(xx, yy*3.4e+11, label='experiment')

plt.title('Harris initial molecular weight distribution')
plt.xlabel('molecular weight')
plt.ylabel('N$_{entries}$')

ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

plt.xlim(1e+2, 1e+8)

plt.legend()
plt.grid()
plt.show()

plt.savefig('Harris_initial_100.png', dpi=300)


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

