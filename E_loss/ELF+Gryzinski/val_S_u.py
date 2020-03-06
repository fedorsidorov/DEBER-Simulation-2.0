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
layer = 'Si'

u_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', layer, 'u.npy'
        ))
S_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', layer, 'S.npy'
        ))
tau_tot = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', layer, 'tau.npy'
        ))
tau_tot_prec = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', layer, 'tau_prec.npy'
        ))

u_core = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', layer, 'u_core.npy'
        ))
S_core = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', layer, 'S_core.npy'
        ))

tau_core = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', layer, 'tau_core.npy'
        ))
tau_core_prec = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', layer, 'tau_core_prec.npy'
        ))


#%%
tau_val = tau_tot - tau_core


#%%
EE = mc.EE_prec

tau_val_prec = tau_tot_prec - tau_core_prec

u_val = np.zeros(len(EE))
S_val = np.zeros(len(EE))


for i, E in enumerate(EE):
    
    inds = np.where(EE < E/2)[0]
    
    u_val[i] = np.trapz(tau_val_prec[i, inds], x=EE[inds])
    S_val[i] = np.trapz(tau_val_prec[i, inds]*EE[inds], x=EE[inds])
   
    
#%%
u_val_f = mu.log_interp1d(EE, u_val)(mc.EE)
S_val_f = mu.log_interp1d(EE, S_val)(mc.EE)


#%%
plt.loglog(mc.EE, u_val_f)
plt.loglog(mc.EE, u_tot, '--')
plt.loglog(mc.EE, u_core)


#%%
plt.loglog(mc.EE, S_val_f)
plt.loglog(mc.EE, S_tot, '--')
plt.loglog(mc.EE, S_core)


#%%
#EE = mc.EE
#
#u_trapz = np.zeros(len(EE))
#S_trapz = np.zeros(len(EE))
#
#
#for i, E in enumerate(EE):
#    
#    inds = np.where(EE < E/2)[0]
#    
#    u_trapz[i] = np.trapz(tau_tot[i, inds], x=EE[inds])
#    S_trapz[i] = np.trapz(tau_tot[i, inds]*EE[inds], x=EE[inds])
#    
#
##%%
#plt.loglog(mc.EE, u_tot)
#plt.loglog(EE, u_trapz, '--')
#
#
##%%
#plt.loglog(mc.EE, S_tot)
#plt.loglog(EE, S_trapz, '--')

