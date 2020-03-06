#%% Import
import numpy as np
#import scission_functions as sf
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
        'elastic', 'PMMA_elastic_U.npy'
        ))

tau_el_int = np.load(os.path.join(mc.sim_folder,
        'elastic', 'PMMA_elastic_int_CS.npy'
        ))


#%% electron-electron
u_ee_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA', 'u.npy'
        ))
u_ee_val = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'PMMA', 'u_val.npy'
        ))
u_ee_C_core = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'u_C.npy'
        ))
u_ee_O_core = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'u_O.npy'
        ))

#plt.loglog(EE, u_ee, label='total')
#plt.loglog(EE, u_ee_val + u_ee_C + u_ee_O, '--', label='val+C+O')


#%%
tau_ee_val_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'PMMA', 'tau_val_int.npy'
        ))
tau_ee_C_core_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'PMMA', 'tau_C_int.npy'
        ))
tau_ee_O_core_int = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski', 'PMMA', 'tau_O_int.npy'
        ))


#%% phonons and polarons
u_ph = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'phonons_polarons', 'PMMA_phonon_U.npy'
        ))
u_pol = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'phonons_polarons', 'PMMA_polaron_U.npy'
        ))


#%%
plt.loglog(mc.EE, u_el, label='elastic')
plt.loglog(mc.EE, u_ee_tot, label='ee')
plt.loglog(mc.EE, u_ee_C_core, label='C 1S')
plt.loglog(mc.EE, u_ee_O_core, label='O 1S')
plt.loglog(mc.EE, u_ee_val, '--', label='val')
plt.loglog(mc.EE, u_ph, label='phonon')
plt.loglog(mc.EE, u_pol, label='polaron')

plt.ylim(1e+1, 1e+9)

plt.grid()
plt.legend()


#%% combine it all
u_list = [u_el, u_ee_val, u_ee_C_core, u_ee_O_core, u_ph, u_pol]
u_proc = np.zeros((len(mc.EE), len(u_list)))


for i in range(len(u_list)):
    u_proc[:, i] = u_list[i]


#%%
tau_int_list = [tau_el_int, tau_ee_val_int, tau_ee_C_core_int, tau_ee_O_core_int]


#%%
el_Eb = np.zeros(len(mc.EE)) ## dummy!!!
val_Eb = np.ones(len(mc.EE)) ## dummy!!!
C_core_Eb = np.ones(len(mc.EE)) * gryz.MMA_core_Eb[0]
O_core_Eb = np.ones(len(mc.EE)) * gryz.MMA_core_Eb[1]

E_bind = [el_Eb, val_Eb, C_core_Eb, O_core_Eb]

scission_probs = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'scission_probs.npy'
        ))

