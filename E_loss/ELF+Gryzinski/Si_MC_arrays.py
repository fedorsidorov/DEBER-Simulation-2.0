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
        'E_loss', 'diel_responce', 'Si', 'u.npy'
        ))
tau_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Si', 'tau.npy'
        ))

u_1s = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'u_1s.npy'
        ))
tau_1s = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'tau_1s.npy'
        ))

u_2s = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'u_2s.npy'
        ))
tau_2s = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'tau_2s.npy'
        ))

u_2p = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'u_2p.npy'
        ))
tau_2p = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'tau_2p.npy'
        ))

u_val = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'u_val.npy'
        ))
tau_val = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'tau_val.npy'
        ))


#%%
#tau_tot_int = mu.diff2int(tau_tot, mc.EE, mc.EE)
#tau_val_int = mu.diff2int(tau_val, mc.EE, mc.EE)
#tau_1s_int = mu.diff2int(tau_1s, mc.EE, mc.EE)
#tau_2s_int = mu.diff2int(tau_2s, mc.EE, mc.EE)
#tau_2p_int = mu.diff2int(tau_2p, mc.EE, mc.EE)


#%%
tau_tot_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'tau_tot_int.npy'
        ))
tau_val_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'tau_val_int.npy'
        ))
tau_1s_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'tau_1s_int.npy'
        ))
tau_2s_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'tau_2s_int.npy'
        ))
tau_2p_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'tau_2p_int.npy'
        ))


#%%
#for i in range(1, len(mc.EE), 100):
#    
#    plt.loglog(mc.EE, tau_tot_int[i, :])
#


