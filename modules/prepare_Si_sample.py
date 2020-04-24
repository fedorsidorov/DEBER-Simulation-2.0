#%% Import
import numpy as np
#import scission_functions as sf
import os
import importlib

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)
#gryz = importlib.reload(gryz)

import matplotlib.pyplot as plt


#%% elastic
u_el = np.load(os.path.join(mc.sim_folder, 
        'elastic', 'final_arrays', 'Si', 'Si_muffin_u.npy'
        ))

u_el_diff_sample = np.load(os.path.join(mc.sim_folder,
        'elastic', 'final_arrays', 'Si', 'Si_muffin_diff_cs_plane_norm.npy'
        ))


#%% electron-electron
u_ee_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'MuElec', 'Si', 'u_ee.npy'
        ))
u_ee_6osc = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'MuElec', 'Si', 'u_6osc.npy'
        ))


#%%
# plt.loglog(mc.EE, u_ee_tot, label='total')
# plt.loglog(mc.EE, u_el, label='elastic')

# plt.legend()


#%%
u_ee_diff = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'MuElec', 'Si', 'sigmadiff_6.npy'
        ))

u_ee_diff_sample = np.zeros(np.shape(u_ee_diff))


for i in range(len(u_ee_diff)):
    
    for j in range(len(mc.EE)):
        
        
        if np.all(u_ee_diff[i, j, :] == 0):
            
            # u_ee_diff_sample[i, j, 0] = 1
            continue
        
        
        u_ee_diff_sample[i, j, :] = u_ee_diff[i, j, :] / np.sum(u_ee_diff[i, j, :])



#%% combine it all
u_processes = np.zeros((len(mc.EE), 7))

u_processes[:, 0 ] = u_el
u_processes[:, 1:] = u_ee_6osc


sigma_diff_sample_processes = np.zeros((7, len(mc.EE), len(mc.EE)))

sigma_diff_sample_processes[0, :, :] = u_el_diff_sample
sigma_diff_sample_processes[1:, :, :] = u_ee_diff_sample

E_bind = mc.Si_MuElec_Eb
E_cut = mc.E_cut_Si

