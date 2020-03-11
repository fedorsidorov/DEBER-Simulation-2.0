#%% Import
import numpy as np
import os
import importlib

import my_constants as mc
import my_utilities as mu
import Gryzinski as gryz

mc = importlib.reload(mc)
mu = importlib.reload(mu)
gryz = importlib.reload(gryz)

import matplotlib.pyplot as plt


#%% elastic
u_el = np.load(os.path.join(mc.sim_folder, 
        'elastic', 'Si_elastic_U.npy'
        ))

tau_el_int = np.load(os.path.join(mc.sim_folder,
        'elastic', 'Si_elastic_int_CS.npy'
        ))


#%% electron-electron
u_ee_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Si', 'u.npy'
        ))
u_ee_val = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'u_val.npy'
        ))
u_ee_1s = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'u_1s.npy'
        ))
u_ee_2s = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'u_2s.npy'
        ))
u_ee_2p = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'u_2p.npy'
        ))


tau_ee_val_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'tau_val_int.npy'
        ))
tau_ee_1s_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'tau_1s_int.npy'
        ))
tau_ee_2s_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'tau_2s_int.npy'
        ))
tau_ee_2p_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'Si', 'tau_2p_int.npy'
        ))


#%%
plt.loglog(mc.EE, u_el, label='elastic')
plt.loglog(mc.EE, u_ee_tot, label='ee')
plt.loglog(mc.EE, u_ee_1s, label='1s')
plt.loglog(mc.EE, u_ee_2s, label='2s')
plt.loglog(mc.EE, u_ee_2p, label='2p')
plt.loglog(mc.EE, u_ee_val, '--', label='val')

plt.ylim(1e+1, 1e+9)

plt.grid()
plt.legend()


#%% combine it all
u_list = [u_el, u_ee_val, u_ee_1s, u_ee_2s, u_ee_2p]
u_proc = np.zeros((len(mc.EE), len(u_list)))


for i in range(len(u_list)):
    u_proc[:, i] = u_list[i]


#%%
tau_int_list = [tau_el_int, tau_ee_val_int, tau_ee_1s_int, tau_ee_2s_int, tau_ee_2p_int]


#%%
el_Eb = np.zeros(len(mc.EE)) ## dummy!!!
val_Eb = np.ones(len(mc.EE)) ## dummy!!!
C_core_Eb = np.ones(len(mc.EE)) * gryz.MMA_core_Eb[0]
O_core_Eb = np.ones(len(mc.EE)) * gryz.MMA_core_Eb[1]

E_bind = [el_Eb, val_Eb, C_core_Eb, O_core_Eb]

scission_probs = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'scission_probs.npy'
        ))


