#%% Import
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import my_utilities as mu
import my_constants as mc
import E_loss_functions_2020 as elf
import Si_valentin2012 as si
import Gryzinski as gryz

import importlib

mu = importlib.reload(mu)
mc = importlib.reload(mc)
elf = importlib.reload(elf)
si = importlib.reload(si)
gryz = importlib.reload(gryz)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'ELF+Gryzinski'
        ))


#%%
def get_Si_val_tau(E, hw):
    return si.get_tau(E, hw) - gryz.get_Gr_Si_core_tau(E, hw)


def get_Si_val_S(E):
    
    def get_Y(hw):
        return get_Si_val_tau(E, hw) * hw
    
    return integrate.quad(get_Y, 0, E/2)[0]


def get_Si_val_u(E):
    
    def get_Y(hw):
        return get_Si_val_tau(E, hw)
    
    return integrate.quad(get_Y, 0, E/2)[0]


#%%
EE = mc.EE

tau = np.zeros((len(EE), len(EE)))
tau_core = np.zeros((len(EE), len(EE)))

S = np.zeros(len(EE))
S_core = np.zeros(len(EE))
S_val = np.zeros(len(EE))
S_val_test = np.zeros(len(EE))
S_Gr_tot = np.zeros(len(EE))

u = np.zeros(len(EE))
u_core = np.zeros(len(EE))
u_val = np.zeros(len(EE))
u_val_test = np.zeros(len(EE))
u_Gr_tot = np.zeros(len(EE))


for i, E in enumerate(EE):
    
    mu.pbar(i, len(EE))
    
#    S[i] = si.get_S(E)
#    S_val[i] = si.get_S(E) - gryz.get_Gr_Si_core_S(E)
#    S_val_test[i] = get_Si_val_S(E)
#    S_core[i] = gryz.get_Gr_Si_core_S(E)
#    S_Gr_tot[i] = gryz.get_Gr_Si_total_S(E)
    
#    u[i] = si.get_u(E)
    u_val[i] = si.get_u(E) - gryz.get_Gr_Si_core_u(E)
    u_val_test[i] = get_Si_val_u(E)
    u_core[i] = gryz.get_Gr_Si_core_u(E)
#    u_Gr_tot[i] = gryz.get_Gr_Si_total_u(E)
    
    for j, hw in enumerate(EE):
#        tau[i, j] = si.get_tau(E, hw)
        tau_core[i, j] = gryz.get_Gr_Si_core_tau(E, hw)


#%%
plt.loglog(EE, S, label='total')
plt.loglog(EE, S_core, label='core')
plt.loglog(EE, S_val, '.', label='val')
plt.loglog(EE, S_val_test, '--', label='val_test')
plt.loglog(EE, elf.get_Si_Gryzinski_core_SP(EE), '.', label='old')
plt.loglog(EE, S_Gr_tot, '.-', label='total Gryzinski')

plt.xlim(1e+1, 1e+4)
plt.ylim(1e+6, 1e+9)

plt.legend()
plt.grid()


#%%
plt.loglog(EE, u, label='total')
plt.loglog(EE, u_core, label='core')
plt.loglog(EE, u_val, '.', label='val')
plt.loglog(EE, u_val_test, '--', label='val_test')
plt.loglog(EE, elf.get_Si_Gryzinski_core_U(EE), '.', label='old')
plt.loglog(EE, u_Gr_tot, '.-', label='total Gryzinski')

plt.xlim(1e+1, 1e+4)
plt.ylim(1e+4, 1e+8)

plt.legend()
plt.grid()


