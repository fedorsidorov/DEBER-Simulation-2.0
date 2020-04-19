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
        # 'elastic', 'final_arrays', 'PMMA', 'u_muffin_extrap.npy'
        'elastic', 'final_arrays', 'PMMA', 'u_muffin.npy'
        ))

u_el_diff_sample = np.load(os.path.join(mc.sim_folder,
        'elastic', 'final_arrays', 'PMMA', 'diff_cs_plane_norm_muffin.npy'
        ))


#%% electron-electron
u_ee = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'u.npy'
        # 'E_loss', 'diel_responce', 'PMMA', 'easy', 'u.npy'
        ))

u_ee_diff_sample = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'tau_norm.npy'
        # 'E_loss', 'diel_responce', 'PMMA', 'easy', 'tau_u_norm.npy'
        ))


#%% phonons and polarons
u_ph = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'phonons_polarons', 'PMMA_u_phonon.npy'
        ))

u_pol = np.load(os.path.join(mc.sim_folder,
        # 'E_loss', 'phonons_polarons', 'PMMA_u_polaron_0p1_0p05.npy'
        # 'E_loss', 'phonons_polarons', 'PMMA_u_polaron_0p2_0p4.npy'
        # 'E_loss', 'phonons_polarons', 'PMMA_u_polaron_0p25_0p05.npy'
        # 'E_loss', 'phonons_polarons', 'PMMA_u_polaron_0p2_0p1.npy'
        # 'E_loss', 'phonons_polarons', 'PMMA_u_polaron_0p07_0p1.npy'
        # 'E_loss', 'phonons_polarons', 'PMMA_u_polaron_0p3_0p2.npy'
        'E_loss', 'phonons_polarons', 'PMMA_u_polaron_1p5_0p14.npy'
        # 'E_loss', 'phonons_polarons', 'PMMA_u_polaron_0p5.npy'
        ))


#%% combine it all
u_processes = np.zeros((len(mc.EE), 4))

u_processes[:, 0] = u_el
u_processes[:, 1] = u_ee
u_processes[:, 2] = u_ph
u_processes[:, 3] = u_pol

