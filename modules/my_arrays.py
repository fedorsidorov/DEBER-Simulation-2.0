#%% Import
import numpy as np
#import os
import importlib
import my_functions as mf
import my_variables as mv
import my_constants as mc
#import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mc = importlib.reload(mc)


#%% Elastic scattering




#%% Total inelastic U
## PMMA
PMMA_total_U = np.load(mv.sim_path_MAC + 'E_loss/diel_responce/Dapor/PMMA_U_Dapor.npy')
PMMA_int_U = np.load(mv.sim_path_MAC + 'E_loss/diel_responce/Dapor/PMMA_int_U_Dapor.npy')

## Si
Si_U_total = np.load(mv.sim_path_MAC + 'E_loss/diel_responce/Palik/Si_U_Palik.npy')
Si_int_U = np.load(mv.sim_path_MAC + 'E_loss/diel_responce/Palik/Si_int_U_Palik.npy')


#%% Core electron U components
## PMMA
PMMA_C_1S_total_U = np.load(mv.sim_path_MAC + 'E_loss/Gryzinski/PMMA/PMMA_C_1S_U.npy')
PMMA_C_1S_int_U = np.load(mv.sim_path_MAC + 'E_loss/Gryzinski/PMMA/PMMA_C_1S_int_U.npy')

PMMA_O_1S_total_U = np.load(mv.sim_path_MAC + 'E_loss/Gryzinski/PMMA/PMMA_O_1S_U.npy')
PMMA_O_1S_int_U = np.load(mv.sim_path_MAC + 'E_loss/Gryzinski/PMMA/PMMA_O_1S_int_U.npy')

## Si
Si_1S_total_U = np.load(mv.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_1S_U.npy')
Si_1S_int_U = np.load(mv.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_1S_int_U.npy')

Si_2S_total_U = np.load(mv.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_2S_U.npy')
Si_2S_int_U = np.load(mv.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_2S_int_U.npy')

Si_2P_total_U = np.load(mv.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_2P_U.npy')
Si_2P_int_U = np.load(mv.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_2P_int_U.npy')


#%%








