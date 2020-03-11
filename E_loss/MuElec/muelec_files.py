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
s_inel = np.loadtxt('microelec/sigma_inelastic_e_Si.dat')
s_inel_tot = np.zeros(len(s_inel))

EE = s_inel[:, 0]

s_6osc_raw = s_inel[:, 1:] * 1e-18


for n in range(6):
    
    now_s_inel = s_6osc_raw[:, n]
    s_inel_tot += now_s_inel
    
    plt.loglog(EE, now_s_inel, 'o', label=str(n))


plt.loglog(EE, s_inel_tot, '.')

sigma_MuElec = np.loadtxt('../diel_responce/curves/Si_MuElec_sigma.txt')
plt.loglog(sigma_MuElec[:, 0], sigma_MuElec[:, 1]*1e-18, '--', label='MuElec')

plt.legend()
plt.grid()

plt.xlim(1e+1, 1e+4)
plt.ylim(1e-21, 1e-15)


#%%
u_6osc_raw = s_6osc_raw * mc.n_Si


for n in range(6):
    
    plt.loglog(EE, u_6osc_raw[:, n], 'o', label=str(n))


l_Chan = np.loadtxt('../diel_responce/curves/Chan_Si_l.txt')

plt.loglog(l_Chan[:, 0], 1/l_Chan[:, 1], label='Chan')

plt.xlim(1e+1, 1e+4)


#%%
u_6osc_raw[np.where(u_6osc_raw == 0)] = 1e-100

u_6osc = np.ones((len(mc.EE), 6)) * 1e-100


beg_ind = np.where(mc.EE >= EE[0])[0][0]


for n in range(6):
    
    u_6osc[beg_ind:, n] = mu.log_interp1d(EE, u_6osc_raw[:, n])(mc.EE[beg_ind:])


u_6osc[np.where(u_6osc == 1e-100)] = 0


u_6osc[np.where(u_6osc < 1)] = 0


#%%
u_ee = np.sum(u_6osc, axis=1)


for n in range(6):
    plt.loglog(mc.EE, u_6osc[:, n], '-.', label=str(n))


plt.loglog(mc.EE, np.sum(u_6osc, axis=1), '-', label='sum')
plt.loglog(l_Chan[:, 0], 1/l_Chan[:, 1], '--', label='Chan')

plt.loglog(mc.EE, u_ee, label='total')


plt.xlim(1e+1, 1e+4)
#plt.ylim(1e+2, 1e+8)

plt.legend()
plt.grid()


#%%
u_6osc_norm = np.zeros(np.shape(u_6osc))


for i in range(len(mc.EE)):
    
    if np.all(u_6osc[i, :] == 0):
        continue
    
    u_6osc_norm[i, :] = u_6osc[i, :] / np.sum(u_6osc[i, :])
    


