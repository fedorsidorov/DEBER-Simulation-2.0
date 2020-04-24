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
        'elastic', 'Si', 'u.npy'
        ))

diff_cs_el_cumulated = np.load(os.path.join(mc.sim_folder,
        'elastic', 'Si', 'diff_cs_cumulated.npy'
        ))


#%% electron-electron
u_ee_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'MuElec', 'Si', 'u_ee.npy'
        ))
u_ee_6osc = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'MuElec', 'Si', 'u_6osc.npy'
        ))


#%%
#plt.loglog(mc.EE, u_ee_tot, label='total')
#plt.loglog(mc.EE, u_el, label='elastic')
#
#plt.legend()


#%%
tau_ee_6osc_cumulated = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'MuElec', 'Si', 'tau_6osc_cumulated.npy'
        ))


#%% combine it all
u_table = np.zeros((len(mc.EE), 7))

u_table[:, 0] = u_el
u_table[:, 1:] = u_ee_6osc


u_table_norm = np.zeros(np.shape(u_table))

for i in range(len(mc.EE)):
    u_table_norm[i, :] = u_table[i, :] / np.sum(u_table[i, :])


#%%
tau_cumulated_table = np.zeros((7, len(mc.EE), len(mc.EE)))


tau_cumulated_table[0, :, :] = diff_cs_el_cumulated
tau_cumulated_table[1:, :, :] = tau_ee_6osc_cumulated


#%%
E_bind = np.zeros(7)

E_bind[1:] = mc.Si_MuElec_Eb

E_cut = mc.E_cut_Si

