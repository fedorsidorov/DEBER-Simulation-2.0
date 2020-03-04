#%% Import
import os
import numpy as np
import matplotlib.pyplot as plt
#from scipy import integrate

import my_utilities as mu
import my_constants as mc

import importlib

mu = importlib.reload(mu)
mc = importlib.reload(mc)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski'
        ))


#%%
u_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'u.npy'
        ))
tau_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'tau_prec.npy'
        ))

u_C = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'u_C.npy'
        ))
tau_C = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'tau_C.npy'
        ))

u_O = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'u_O.npy'
        ))
tau_O = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'tau_O.npy'
        ))

u_val = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'PMMA', 'u_val.npy'
        ))
tau_val = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'PMMA', 'tau_val.npy'
        ))


#%%
tau_tot_int = mu.diff2int(tau_tot, mc.EE, mc.EE)
tau_val_int = mu.diff2int(tau_val, mc.EE, mc.EE)
tau_C_int = mu.diff2int(tau_C, mc.EE, mc.EE)
tau_O_int = mu.diff2int(tau_O, mc.EE, mc.EE)


#%%
for i in range(1, len(mc.EE), 100):
    
    plt.loglog(mc.EE, tau_tot_int[i, :])







