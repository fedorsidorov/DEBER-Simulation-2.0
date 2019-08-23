#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import E_loss_functions as elf

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
elf = importlib.reload(elf)

os.chdir(mc.sim_path_MAC + 'E_loss/Gryzinski')


#%%
Si_1S_DIFF_U = elf.get_Si_Gryzinski_1S_diff_U(mc.EE)
Si_2S_DIFF_U = elf.get_Si_Gryzinski_2S_diff_U(mc.EE)
Si_2P_DIFF_U = elf.get_Si_Gryzinski_2P_diff_U(mc.EE)
Si_3S_DIFF_U = elf.get_Si_Gryzinski_3S_diff_U(mc.EE)
Si_3P_DIFF_U = elf.get_Si_Gryzinski_3P_diff_U(mc.EE)

Si_total_diff_U = Si_1S_DIFF_U + Si_2S_DIFF_U + Si_2P_DIFF_U + Si_3S_DIFF_U + Si_3P_DIFF_U


#%%
Si_diff_inel_U = np.load(mc.sim_path_MAC + 'E_loss/diel_responce/Palik/Si_diff_U_Palik.npy')
EE = mc.EE


#%%
ind = 900 ## 200 eV

plt.loglog(EE, Si_diff_inel_U[ind, :], label='Palik')
plt.loglog(EE, Si_total_diff_U[ind, :], label='Gryzinski')

plt.show()

