#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(mc.sim_path_MAC + 'E_loss')


#%%
Palik_diff_U = np.load('diel_responce/Palik/Si_diff_U_Palik_Ashley.npy') / 1e+2 * mc.eV
#Palik_diff_U = np.load('diel_responce/Palik/Si_diff_U_Palik_Chan.npy') / 1e+2 * mc.eV

E_Palik = np.load('diel_responce/E_PALIK.npy')

Si_1S_DIFF_U = np.load('Gryzinski/Si/Si_1S_DIFF_U.npy')
Si_2S_DIFF_U = np.load('Gryzinski/Si/Si_2S_DIFF_U.npy')
Si_2P_DIFF_U = np.load('Gryzinski/Si/Si_2P_DIFF_U.npy')
Si_3S_DIFF_U = np.load('Gryzinski/Si/Si_3S_DIFF_U.npy')
Si_3P_DIFF_U = np.load('Gryzinski/Si/Si_3P_DIFF_U.npy')

Gryzinski_diff_U = Si_1S_DIFF_U + Si_2S_DIFF_U + Si_2P_DIFF_U + Si_3S_DIFF_U + Si_3P_DIFF_U

E = mc.EE


#%%
## E = 200 eV
ind_P = 3486
ind = 522

plt.loglog(E_Palik, Palik_diff_U[ind_P, :], label='Palik')
plt.loglog(E, Gryzinski_diff_U[ind, :], label='Gryzinski')

plt.grid()
plt.legend()
plt.show()


#%%
## E = 1000 eV
ind_P = 4545
ind = 682

plt.loglog(E_Palik, Palik_diff_U[ind_P, :], label='Palik')
plt.loglog(E, Gryzinski_diff_U[ind, :], label='Gryzinski')

plt.grid()
plt.legend()
plt.show()


#%%
## E = 2000 eV
ind_P = 4999
ind = 750

plt.loglog(E_Palik, Palik_diff_U[ind_P, :], label='Palik')
plt.loglog(E, Gryzinski_diff_U[ind, :], label='Gryzinski')

plt.grid()
plt.legend()
plt.show()




