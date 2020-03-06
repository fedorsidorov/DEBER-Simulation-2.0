#%% Import
import numpy as np
import os
import importlib
from scipy import integrate

import my_constants as mc
import my_utilities as mu
import Gryzinski as gryz

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)
gryz = importlib.reload(gryz)


os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'MuElec'
        ))


#%%
s_el = np.loadtxt('microelec/sigma_elastic_e_Si.dat')

plt.loglog(s_el[:, 0], s_el[:, 1])

plt.xlim(1e+1, 1e+4)


#%%
s_inel = np.loadtxt('microelec/sigma_inelastic_e_Si.dat')
s_inel_tot = np.zeros(len(s_inel))

EE = s_inel[:, 0]


for i in range(1, len(s_inel[0])):
    
    print(i)
    
    now_s_inel = s_inel[:, i]
    s_inel_tot += now_s_inel
    
    plt.loglog(EE, s_inel[:, i], 'o')


plt.loglog(EE, s_inel_tot)

plt.xlim(1e+1, 1e+4)

sigma_MuElec = np.loadtxt('../diel_responce/curves/Si_MuElec_sigma.txt')
plt.loglog(sigma_MuElec[:, 0], sigma_MuElec[:, 1] * 1e-18 / 1e+5 * mc.n_Si, '--', label='MuElec')


#%%
