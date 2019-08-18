#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import my_constants as mc
import E_loss_functions as elf

import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mc = importlib.reload(mc)
elf = importlib.reload(elf)

os.chdir(mv.sim_path_MAC + 'E_loss')


#%% PMMA
PMMA_C_1S_INT_U = elf.get_PMMA_C_Gryzinski_1S_int_U(mv.EE)
PMMA_O_1S_INT_U = elf.get_PMMA_O_Gryzinski_1S_int_U(mv.EE)

np.save('Gryzinski/PMMA_C_1S_INT_U.npy', PMMA_C_1S_INT_U)
np.save('Gryzinski/PMMA_O_1S_INT_U.npy', PMMA_O_1S_INT_U)

PMMA_C_1S_U = elf.get_PMMA_C_Gryzinski_1S_U(mv.EE)
PMMA_O_1S_U = elf.get_PMMA_O_Gryzinski_1S_U(mv.EE)

np.save('Gryzinski/PMMA_C_1S_U.npy', PMMA_C_1S_U)
np.save('Gryzinski/PMMA_O_1S_U.npy', PMMA_O_1S_U)


#%%
PMMA_core_SP = elf.get_PMMA_Gryzinski_core_SP(mv.EE)
np.save('Gryzinski/PMMA_core_SP.npy', PMMA_core_SP)


#%% Si
Si_1S_DIFF_U = elf.get_Si_Gryzinski_1S_diff_U(mv.EE)
Si_2S_DIFF_U = elf.get_Si_Gryzinski_2S_diff_U(mv.EE)
Si_2P_DIFF_U = elf.get_Si_Gryzinski_2P_diff_U(mv.EE)
Si_3S_DIFF_U = elf.get_Si_Gryzinski_3S_diff_U(mv.EE)
Si_3P_DIFF_U = elf.get_Si_Gryzinski_3P_diff_U(mv.EE)

np.save('Gryzinski/Si_1S_DIFF_U.npy', Si_1S_DIFF_U)
np.save('Gryzinski/Si_2S_DIFF_U.npy', Si_2S_DIFF_U)
np.save('Gryzinski/Si_2P_DIFF_U.npy', Si_2P_DIFF_U)
np.save('Gryzinski/Si_3S_DIFF_U.npy', Si_3S_DIFF_U)
np.save('Gryzinski/Si_3P_DIFF_U.npy', Si_3P_DIFF_U)

Si_1S_INT_U = elf.get_Si_Gryzinski_1S_int_U(mv.EE)
Si_2S_INT_U = elf.get_Si_Gryzinski_2S_int_U(mv.EE)
Si_2P_INT_U = elf.get_Si_Gryzinski_2P_int_U(mv.EE)
Si_3S_INT_U = elf.get_Si_Gryzinski_3S_int_U(mv.EE)
Si_3P_INT_U = elf.get_Si_Gryzinski_3P_int_U(mv.EE)

np.save('Gryzinski/Si_1S_INT_U.npy', Si_1S_INT_U)
np.save('Gryzinski/Si_2S_INT_U.npy', Si_2S_INT_U)
np.save('Gryzinski/Si_2P_INT_U.npy', Si_2P_INT_U)
np.save('Gryzinski/Si_3S_INT_U.npy', Si_3S_INT_U)
np.save('Gryzinski/Si_3P_INT_U.npy', Si_3P_INT_U)

Si_1S_U = elf.get_Si_Gryzinski_1S_U(mv.EE)
Si_2S_U = elf.get_Si_Gryzinski_2S_U(mv.EE)
Si_2P_U = elf.get_Si_Gryzinski_2P_U(mv.EE)
Si_3S_U = elf.get_Si_Gryzinski_3S_U(mv.EE)
Si_3P_U = elf.get_Si_Gryzinski_3P_U(mv.EE)

np.save('Gryzinski/Si_1S_U.npy', Si_1S_U)
np.save('Gryzinski/Si_2S_U.npy', Si_2S_U)
np.save('Gryzinski/Si_2P_U.npy', Si_2P_U)
np.save('Gryzinski/Si_3S_U.npy', Si_3S_U)
np.save('Gryzinski/Si_3P_U.npy', Si_3P_U)


#%%
Si_VAL_INT_U = elf.get_Si_Gryzinski_VAL_int_U(mv.EE)

np.save('Gryzinski/Si_VAL_INT_U.npy', Si_VAL_INT_U)

Si_VAL_U = elf.get_Si_Gryzinski_VAL_U(mv.EE)

np.save('Gryzinski/Si_VAL_U.npy', Si_VAL_U)

Si_TOTAL_U_SS = Si_1S_U + Si_2S_U + Si_2P_U + Si_3S_U + Si_3P_U
Si_TOTAL_U_VAL = Si_1S_U + Si_2S_U + Si_2P_U + Si_VAL_U

np.save('Gryzinski/Si_TOTAL_U_SS.npy', Si_TOTAL_U_SS)
np.save('Gryzinski/Si_TOTAL_U_VAL.npy', Si_TOTAL_U_VAL)


#%%
Si_core_SP = elf.get_Si_Gryzinski_core_SP(mv.EE)
np.save('Gryzinski/Si_core_SP.npy', Si_core_SP)

Si_Gryzinski_total_SP = elf.get_Si_Gryzinski_total_SP(mv.EE)
np.save('Gryzinski/Si_Gryzinski_total_SP.npy', Si_Gryzinski_total_SP)


#%%
plt.loglog(mv.EE, Si_TOTAL_U_SS, label='Subshell sum')
plt.loglog(mv.EE, Si_TOTAL_U_VAL, label='Mean E$_{val}$')

plt.legend()
plt.grid()
plt.show()

#%% Test for Si
Si_DIFF_U = Si_1S_DIFF_U + Si_2S_DIFF_U + Si_2P_DIFF_U + Si_3S_DIFF_U + Si_3P_DIFF_U

#%%
EE = mv.EE

#%% 682
CHAN_1000_Si = np.loadtxt('Chan_diel_Si_1000.txt')

plt.loglog(EE, Si_DIFF_U[682, :])
plt.loglog(CHAN_1000_Si[:, 0], CHAN_1000_Si[:, 1])

plt.show()

