#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

import my_constants as mc
mc = importlib.reload(mc)

import E_loss_functions as elf
elf = importlib.reload(elf)


#%%
MMA_bonds = {}

MMA_bonds['Op-Cp'] = 815, 4
MMA_bonds['O-Cp'] = 420, 2
MMA_bonds['H-C3'] = 418, 12
MMA_bonds['H-C2'] = 406, 4
MMA_bonds['Cp-Cg'] = 383, 2
MMA_bonds['O-C3'] = 364, 4
MMA_bonds['C-C3'] = 356, 2
MMA_bonds['C-C2'] = 354, 4


#%%
MMA_diff_CS = {}

MMA_diff_CS['Op-Cp'] = get_Gryzinski_diff_CS(EE, MMA_bonds['Op-Cp'][1], WW=mc.EE)


#%%
plt.xkcd()
plt.plot(np.array((1, 2)), np.array((1, 2)))

