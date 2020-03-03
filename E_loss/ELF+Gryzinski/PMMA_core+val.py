#%% Import
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


import my_utilities as mu
import my_constants as mc
import E_loss_functions_2020 as elf
import PMMA_dapor2015 as pmma
import Gryzinski as gryz

import importlib

mu = importlib.reload(mu)
mc = importlib.reload(mc)
elf = importlib.reload(elf)
pmma = importlib.reload(pmma)
gryz = importlib.reload(gryz)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski'
        ))


#%%
EE = np.logspace(0, 4.4, 100)


#%% PMMA
def get_PMMA_val_tau(E, hw):
    return pmma.get_tau(E, hw) - gryz.get_Gr_PMMA_core_tau(E, hw)


def get_PMMA_val_S(E):
    
    def get_Y(hw):
        return get_PMMA_val_tau(E, hw) * hw
    
    return integrate.quad(get_Y, 0, E/2)[0]


def get_PMMA_val_u(E):
    
    def get_Y(hw):
        return get_PMMA_val_tau(E, hw)
    
    return integrate.quad(get_Y, 0, E/2)[0]


tau_PMMA = np.zeros((len(EE), len(EE)))
tau_PMMA_core = np.zeros((len(EE), len(EE)))

S_PMMA = np.zeros(len(EE))
S_PMMA_core = np.zeros(len(EE))
S_PMMA_val = np.zeros(len(EE))
S_PMMA_val_test = np.zeros(len(EE))

u_PMMA = np.zeros(len(EE))
u_PMMA_core = np.zeros(len(EE))
u_PMMA_val = np.zeros(len(EE))
u_PMMA_val_test = np.zeros(len(EE))


for i, E in enumerate(EE):
    
    mu.pbar(i, len(EE))
    
    S_PMMA[i] = pmma.get_S(E)
    S_PMMA_val[i] = pmma.get_S(E) - gryz.get_Gr_PMMA_core_S(E)
    S_PMMA_val_test[i] = get_PMMA_val_S(E)
    S_PMMA_core[i] = gryz.get_Gr_PMMA_core_S(E)
    
    u_PMMA[i] = pmma.get_u(E)
    u_PMMA_val[i] = pmma.get_u(E) - gryz.get_Gr_PMMA_core_u(E)
    u_PMMA_val_test[i] = get_PMMA_val_u(E)
    u_PMMA_core[i] = gryz.get_Gr_PMMA_core_u(E)
    
    for j, hw in enumerate(EE):
        tau_PMMA[i, j] = pmma.get_tau(E, hw)
        tau_PMMA_core[i, j] = gryz.get_Gr_PMMA_core_tau(E, hw)


#%%
plt.loglog(EE, S_PMMA, label='total')
plt.loglog(EE, S_PMMA_core, label='core')
plt.loglog(EE, S_PMMA_val, '.', label='val')
plt.loglog(EE, S_PMMA_val_test, '--', label='val_test')
plt.loglog(EE, elf.get_PMMA_Gryzinski_core_SP(EE), '.', label='old')

plt.legend()
plt.grid()


#%%
plt.loglog(EE, u_PMMA, label='total')
plt.loglog(EE, u_PMMA_core, label='core')
plt.loglog(EE, u_PMMA_val, '-', label='val')
plt.loglog(EE, u_PMMA_val_test, '--', label='val_test')
plt.loglog(EE, elf.get_PMMA_Gryzinski_core_U(EE), '.', label='old')

plt.legend()
plt.grid()


