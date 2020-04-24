#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)


#%% elastic
u_el = np.load(os.path.join(mc.sim_folder, 
        'elastic', 'final_arrays', 'PMMA', 'PMMA_muffin_u.npy'
        ))

u_el_diff_sample = np.load(os.path.join(mc.sim_folder,
        'elastic', 'final_arrays', 'PMMA', 'MMA_muffin_diff_cs_plane_norm.npy'
        ))


#%% electron-electron
u_ee = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'Mermin', 'IIMFP.npy'
        ))
# u_ee_pre = np.load(os.path.join(mc.sim_folder,
        # 'E_loss', 'diel_responce', 'PMMA', 'Ritsko_Henke', 'RH', 'u_RH.npy'
        # ))

u_ee_diff_sample = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'Mermin', 'DIIMFP_norm.npy'
        ))
# u_ee_diff_sample_pre = np.load(os.path.join(mc.sim_folder,
        # 'E_loss', 'diel_responce', 'PMMA', 'Ritsko_Henke', 'RH', 'tau_u_norm_RH.npy'
        # ))


#%% phonons and polarons
u_ph = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'phonons_polarons', 'PMMA_u_ph.npy'
        ))

u_pol = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'phonons_polarons', 'PMMA_u_pol_0p1_0p15.npy'
        ))


#%% combine it all
u_processes = np.zeros((len(mc.EE), 4))

u_processes[:, 0] = u_el
u_processes[:, 1] = u_ee
u_processes[:, 2] = u_ph
u_processes[:, 3] = u_pol


sigma_diff_sample_processes = np.zeros((2, len(mc.EE), len(mc.EE)))

sigma_diff_sample_processes[0, :, :] = u_el_diff_sample
sigma_diff_sample_processes[1:, :, :] = u_ee_diff_sample


#%%
# for i in range(4):
    # plt.loglog(mc.EE, 1/u_processes[:, i] * 1e+8)

# plt.ylim(1e+5, 1e+9)


# plt.xlim(0, 200)
# plt.ylim(1e+0, 1e+3)





