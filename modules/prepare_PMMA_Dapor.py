#%% Import
import numpy as np
import os
import importlib

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)

#import matplotlib.pyplot as plt


#%% elastic
u_el = np.load(os.path.join(mc.sim_folder, 
        'elastic', 'PMMA', 'u.npy'
        ))
u_el_diff_cumulated = np.load(os.path.join(mc.sim_folder,
        'elastic', 'PMMA', 'diff_cs_cumulated.npy'
        ))


#%% electron-electron
u_ee = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'u_ee.npy'
        ))
u_ee_diff_cumulated = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'tau_cumulated.npy'
        ))


#%% phonons and polarons
u_ph = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'phonons_polarons', 'PMMA_u_phonon.npy'
        ))
u_pol = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'phonons_polarons', 'PMMA_u_polaron.npy'
        ))


#%% combine it all
u_processes = np.zeros((len(mc.EE), 4))

u_processes[:, 0] = u_el
u_processes[:, 1] = u_ee
u_processes[:, 2] = u_ph
u_processes[:, 3] = u_pol
