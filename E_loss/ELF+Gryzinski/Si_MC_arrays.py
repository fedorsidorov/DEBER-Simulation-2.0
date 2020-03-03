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
        'E_loss', 'diel_responce', 'Si_valentin2012', 'u.npy'
        ))

S_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Si_valentin2012', 'S.npy'
        ))

tau_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Si_valentin2012', 'tau_prec.npy'
        ))

u_core = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'u_core.npy'
        ))

S_core = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'S_core.npy'
        ))

tau_core = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'tau_core.npy'
        ))


#%%
#EE = mc.EE
#EE = np.logspace(-2, 4.4, 2000)
#EE = np.logspace(-1, 4.4, 2000)
EE = np.logspace(-1, 4.4, 1000)

u_trapz = np.zeros(len(EE))
S_trapz = np.zeros(len(EE))


for i, E in enumerate(EE):
    
    inds = np.where(EE < E/2)[0]
    
    u_trapz[i] = np.trapz(tau_tot[i, inds], x=EE[inds])
    S_trapz[i] = np.trapz(tau_tot[i, inds]*EE[inds], x=EE[inds])
    

#%%
plt.loglog(mc.EE, u_tot)
plt.loglog(EE, u_trapz, '--')


#%%
plt.loglog(mc.EE, S_tot)
plt.loglog(EE, S_trapz, '--')

