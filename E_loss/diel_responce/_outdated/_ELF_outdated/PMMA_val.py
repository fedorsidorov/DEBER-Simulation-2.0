#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu
import E_loss_functions_2020 as elf

#from itertools import product

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)
elf = importlib.reload(elf)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce'
        ))


#%%
## Total
#S = np.load('PMMA_dapor2015/final/PMMA_S_f_dapor2015.npy') / mc.eV / 1e+2
#u = np.load('PMMA_dapor2015/final/PMMA_u_f_dapor2015.npy') / 1e+2
#tau = np.load('PMMA_dapor2015/final/PMMA_tau_f_dapor2015.npy') * mc.eV / 1e+2

EE_eV = np.load('PMMA_dapor2015/PMMA_EE_dapor2015.npy') / mc.eV
#EE_eV = mc.EE_eV

S = np.load('PMMA_dapor2015/PMMA_S_dapor2015.npy') / mc.eV / 1e+2
u = np.load('PMMA_dapor2015/PMMA_u_dapor2015.npy') / 1e+2
tau = np.load('PMMA_dapor2015/PMMA_tau_dapor2015.npy') * mc.eV / 1e+2

tau[np.where(np.isnan(tau))] = 0


## Core
S_core = elf.get_PMMA_C_1S_SP(EE_eV) + elf.get_PMMA_O_1S_SP(EE_eV)

u_core = elf.get_PMMA_C_Gryzinski_1S_U(EE_eV) + elf.get_PMMA_O_Gryzinski_1S_U(EE_eV)

tau_core = elf.get_PMMA_C_Gryzinski_1S_diff_U(EE_eV, EE_eV) +\
        elf.get_PMMA_O_Gryzinski_1S_diff_U(EE_eV, EE_eV)


## Valence
S_val = S - S_core
u_val = u - u_core
tau_val = tau - tau_core


#%%
plt.loglog(EE_eV, S, 'o')
plt.loglog(EE_eV, S_core)


#%%
plt.loglog(EE_eV, u)
plt.loglog(EE_eV, u_core)


#%%
S_val_test = np.zeros(len(EE_eV))
u_val_test = np.zeros(len(EE_eV))


for i in range(len(EE_eV)):
    
    inds = np.where(EE_eV <= EE_eV[i]/2)
    
    S_val_test[i] = np.trapz(tau_val[i, inds] * EE_eV[inds], x=EE_eV[inds])
    u_val_test[i] = np.trapz(tau_val[i, inds], x=EE_eV[inds])


#%%
plt.semilogx(EE_eV, S_val, 'o', label='val')
plt.semilogx(EE_eV, S_val_test, label='val_test')

plt.legend()


#%%
plt.semilogx(EE_eV, u_val, 'o', label='val')
plt.semilogx(EE_eV, u_val_test, label='val_test')

plt.legend()

