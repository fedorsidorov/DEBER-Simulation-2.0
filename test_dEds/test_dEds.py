#%% Import
import numpy as np
from numpy import linalg as LA
import os
import importlib
import matplotlib.pyplot as plt

import my_constants as mc
import my_utilities as mu
import MC_functions_osc as mcf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
mcf = importlib.reload(mcf)

os.chdir(os.path.join(mc.sim_folder, 'test_dEds'))


#%%
#data_folder = os.path.join(mc.sim_folder, 'e_DATA', 'primary', 'PMMA_osc')
data_folder = os.path.join(mc.sim_folder, 'e_DATA', 'primary', 'Si_osc')

n_files = 438
#n_files = 500
n_tracks = 10

dEds = np.zeros((len(mc.EE), 2))


for i in range(n_files):
    
    mu.pbar(i, n_files)
    
#    now_DATA = np.load(os.path.join(data_folder, 'DATA_PMMA_prim_' + str(i) + '.npy'))
    now_DATA = np.load(os.path.join(data_folder, 'DATA_Si_prim_' + str(i) + '.npy'))
    now_DATA_inel = now_DATA[np.where(now_DATA[:, 3] != 0)]
    
    
    for n_tr in range(n_tracks):
        
        now_arr = now_DATA_inel[np.where(now_DATA_inel[:, 0] == n_tr)]
        
        
        for i in range(len(now_arr) - 1):
            
            now_E = now_arr[i, 4]
            now_dE = now_arr[i, 8] + now_arr[i, 9]
            now_dr = now_arr[i+1, 5:8] - now_arr[i, 5:8]
            now_ds = LA.norm(now_dr)
            
            E_ind = mcf.get_closest_el_ind(mc.EE, now_E)
            
            dEds[E_ind, 0] += now_dE
            dEds[E_ind, 1] += now_ds


#%%
plt.semilogx(mc.EE, dEds[:, 0]/dEds[:, 1])

#S_Dapor = np.loadtxt('../E_loss/diel_responce/curves/S_dapor2015.txt')
#plt.semilogx(S_Dapor[:, 0], S_Dapor[:, 1]*1e+8, 'o', label='dapor2015.pdf')

S_MuElec = np.loadtxt('../E_loss/diel_responce/curves/Si_MuElec_S.txt')
plt.semilogx(S_MuElec[:, 0], S_MuElec[:, 1] * 1e+7, label='MuElec')

plt.ylim(0, 0.5e+9)


