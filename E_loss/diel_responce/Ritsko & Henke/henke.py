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

os.chdir(os.path.join(mc.sim_folder, 'E_loss', 'diel_responce', 'Henke'))


#%%
# H = np.loadtxt('Henke_O.txt')[122:, :]
# Hm = np.loadtxt('Henke_H_my.txt')

# plt.semilogx(H[:, 0], H[:, 1])
# plt.loglog(Hm[:, 0], Hm[:, 2])

v = np.ones(len(mc.EE_10))
k = np.zeros(len(mc.EE_10))


mult = 1 / (2*np.pi) * mc.n_MMA * mc.r0


PMMA_dict = {
    'H': 8,
    'C': 5,
    'O': 2
    }


for el in ['H', 'C', 'O']:
    
    now_el = np.loadtxt('Henke_' + el + '.txt')
    
    now_el[np.where(now_el < 0)] = 0
    
    
    f1 = mu.semilogx_interp1d(now_el[:, 0], now_el[:, 1])(mc.EE_10)
    f2 = mu.log_interp1d(now_el[:, 0], now_el[:, 2])(mc.EE_10)
    
    lambda2 = (mc.h * mc.c / (mc.EE_10 * mc.eV))**2 * 1e+4 ## cm^2
    
    v -= mult * lambda2 * PMMA_dict[el] * f1
    k += mult * lambda2 * PMMA_dict[el] * f2


eps1 = v**2 - k**2
eps2 = 2 * v * k

Im_PMMA = eps2 / (eps1**2 + eps2**2)    


plt.loglog(mc.EE_10[249:], Im_PMMA[249:])

# plt.xlim(1e+2, 1e+3)
# plt.ylim(1e-5, 1)

Ritsko_Im = np.loadtxt('ritsko_Im.txt')
plt.loglog(Ritsko_Im[:, 0], Ritsko_Im[:, 1])

Dapor_Im = np.loadtxt('Dapor_Im.txt')
plt.loglog(Dapor_Im[:, 0], Dapor_Im[:, 1])

