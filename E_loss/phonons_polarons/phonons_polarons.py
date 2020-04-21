#%% Import
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder, 'E_loss', 'phonons_polarons'))


#%% PMMA phonons and polarons
def get_PMMA_U_phonon(EE, T=300):
    
    # a0 = 5.292e-11 * 1e+2 ## cm
    # kT = 8.617e-5 * T ## eV
    kT = mc.k_B * T
    hw = 0.1
    e_0 = 3.9
    e_inf = 2.2
    
    nT = (np.exp(hw/(kT)) - 1)**(-1)
    KT = 1/mc.a0 * ((nT + 1)/2) * ((e_0 - e_inf)/(e_0 * e_inf))
    
    U_PH = KT * hw/EE * np.log((1 + np.sqrt(1 - hw/EE))/(1 - np.sqrt(1 - hw/EE))) ## cm^-1
    
    return U_PH


def get_PMMA_U_polaron(EE, C_inv_nm, gamma):
    
    C = C_inv_nm * 1e+7 ## nm^-1 -> cm^-1
    
    U_POL = C * np.exp(-gamma * EE) ## cm^-1
    
    return U_POL


#%%
u_ph = get_PMMA_U_phonon(mc.EE)
u_pol = get_PMMA_U_polaron(mc.EE, 0.1, 0.15)

# plt.loglog(EE, u_ph)
# plt.loglog(EE, u_pol)

# plt.xlim(1, 1e+4)
# plt.ylim(1e+1, 1e+9)


#%%
D_ph = np.loadtxt('Dapor_thesis_l_ph.txt')
D_pol = np.loadtxt('Dapor_thesis_l_pol.txt')

plt.semilogy(D_ph[:, 0], D_ph[:, 1], 'o', label='Dapor phonon')
plt.semilogy(D_pol[:, 0], D_pol[:, 1], 'o', label='Dapor polaron')

ind = 523
EE = mc.EE[:ind]

u_ph = get_PMMA_U_phonon(EE)
u_pol = get_PMMA_U_polaron(EE, 1.5, 0.14)

plt.semilogy(EE, 1/u_ph * 1e+8, label='my phonon')
plt.semilogy(EE, 1/u_pol * 1e+8, label='my_polaron')

plt.xlim(0, 200)
plt.ylim(1e+0, 1e+3)

plt.legend()
plt.grid()
