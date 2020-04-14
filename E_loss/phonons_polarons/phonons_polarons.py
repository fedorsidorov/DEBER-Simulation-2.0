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
    
    a0 = 5.292e-11 * 1e+2 ## cm
    kT = 8.617e-5 * T ## eV
    hw = 0.1
    e_0 = 3.9
    e_inf = 2.2
    
    nT = (np.exp(hw/(kT)) - 1)**(-1)
    KT = 1/a0 * ((nT + 1)/2) * ((e_0 - e_inf)/(e_0*e_inf))
    
    U_PH = KT * hw/EE * np.log( (1 + np.sqrt(1 - hw/EE)) / (1 - np.sqrt(1 - hw/EE)) ) ## m^-1
    
    return U_PH


def get_PMMA_U_polaron(EE, C_inv_nm=0.1):
    
    C = C_inv_nm * 1e+7 ## nm^-1 -> cm^-1
    gamma = 0.15 ## ev^-1
    
    U_POL = C * np.exp(-gamma * EE) ## cm^-1
    
    return U_POL


#%%
u_ph = get_PMMA_U_phonon(mc.EE)
u_pol = get_PMMA_U_polaron(mc.EE, 0.25)

plt.loglog(mc.EE, u_ph)
plt.loglog(mc.EE, u_pol)

plt.xlim(1, 1e+4)
plt.ylim(1e+1, 1e+9)
