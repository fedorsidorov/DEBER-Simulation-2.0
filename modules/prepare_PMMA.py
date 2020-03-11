#%% Import
import numpy as np
import os
import importlib

import my_constants as mc
import my_utilities as mu
import scission_functions_2020 as sf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
sf = importlib.reload(sf)

import matplotlib.pyplot as plt


#%% elastic
u_el = np.load(os.path.join(mc.sim_folder, 
        'elastic', 'PMMA', 'u.npy'
        ))

diff_cs_el_cumulated = np.load(os.path.join(mc.sim_folder,
        'elastic', 'PMMA', 'diff_cs_cumulated.npy'
        ))


#%% electron-electron
u_ee_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'u_ee.npy'
        ))
u_ee_4osc = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'u_4osc.npy'
        ))


#%%
#plt.loglog(mc.EE, u_ee_tot, label='total')
#plt.loglog(mc.EE, u_el, label='elastic')
#
#plt.legend()


#%%
tau_ee_4osc_cumulated = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'tau_4osc_cumulated.npy'
        ))




#%% phonons and polarons
u_ph = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'phonons_polarons', 'PMMA_phonon_U.npy'
        ))
u_pol = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'phonons_polarons', 'PMMA_polaron_U.npy'
        ))


#%% combine it all
u_table = np.zeros((len(mc.EE), 7))

u_table[:, 0] = u_el
u_table[:, 1:5] = u_ee_4osc
u_table[:, 5] = u_ph
u_table[:, 6] = u_pol


u_table_norm = np.zeros(np.shape(u_table))

for i in range(len(mc.EE)):
    u_table_norm[i, :] = u_table[i, :] / np.sum(u_table[i, :])


#%%
tau_cumulated_table = np.zeros((5, len(mc.EE), len(mc.EE)))

tau_cumulated_table[0, :, :] = diff_cs_el_cumulated
tau_cumulated_table[1:, :, :] = tau_ee_4osc_cumulated


#%%
E_bind = np.zeros(5) 

E_bind[1:] = mc.PMMA_Ebind

scission_probs = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'scission_probs.npy'
        ))

scission_probs[np.where(scission_probs) == -0] = 0


val_E_bind = sf.bonds_BDE

E_cut = mc.E_cut_PMMA

