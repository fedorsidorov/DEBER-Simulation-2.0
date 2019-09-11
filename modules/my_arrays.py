#%% Import
import numpy as np
import my_constants as mc
import my_utilities as mu

import importlib
mc = importlib.reload(mc)
mu = importlib.reload(mu)

#import matplotlib.pyplot as plt


#%%
EE = mc.EE
THETA = mc.THETA


#%% Elastic scattering
PMMA_el_U = np.load(mc.sim_folder + 'elastic/PMMA_elastic_U.npy')
PMMA_el_int_U = np.load(mc.sim_folder + 'elastic/PMMA_elastic_int_CS.npy')

Si_el_U = np.load(mc.sim_folder + 'elastic/Si_elastic_U.npy')
Si_el_int_U = np.load(mc.sim_folder + 'elastic/Si_elastic_int_CS.npy')


#%% Total inelastic U
## PMMA
PMMA_total_inel_U = np.load(mc.sim_folder + 'E_loss/diel_responce/Dapor/PMMA_U_Dapor.npy')
PMMA_diff_inel_U = np.load(mc.sim_folder +\
                           'E_loss/diel_responce/Dapor/PMMA_diff_U_Dapor_Ashley.npy')
PMMA_int_inel_U = np.load(mc.sim_folder +\
                           'E_loss/diel_responce/Dapor/PMMA_int_U_Dapor_Ashley.npy')

## Si
Si_total_inel_U = np.load(mc.sim_folder + 'E_loss/diel_responce/Palik/Si_U_Palik.npy')
Si_diff_inel_U = np.load(mc.sim_folder +\
                         'E_loss/diel_responce/Palik/Si_diff_U_Palik_Ashley.npy')


#%% Core electron U components
## PMMA
PMMA_C_1S_total_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/PMMA/PMMA_C_1S_U.npy')
PMMA_C_1S_diff_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/PMMA/PMMA_C_1S_diff_U.npy')
PMMA_C_1S_int_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/PMMA/PMMA_C_1S_int_U.npy')

PMMA_O_1S_total_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/PMMA/PMMA_O_1S_U.npy')
PMMA_O_1S_diff_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/PMMA/PMMA_O_1S_diff_U.npy')
PMMA_O_1S_int_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/PMMA/PMMA_O_1S_int_U.npy')


#%%
PMMA_val_total_U = np.load(mc.sim_folder + 'E_loss/diel_responce/Dapor/PMMA_val_tot_U_D+G.npy')
PMMA_val_int_U = np.load(mc.sim_folder + 'E_loss/diel_responce/Dapor/PMMA_val_int_U_D+G+A.npy')


#%%
## Si
Si_1S_total_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/Si/Si_1S_U.npy')
Si_1S_diff_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/Si/Si_1S_diff_U.npy')
Si_1S_int_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/Si/Si_1S_int_U.npy')

Si_2S_total_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/Si/Si_2S_U.npy')
Si_2S_diff_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/Si/Si_2S_diff_U.npy')
Si_2S_int_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/Si/Si_2S_int_U.npy')

Si_2P_total_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/Si/Si_2P_U.npy')
Si_2P_diff_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/Si/Si_2P_diff_U.npy')
Si_2P_int_U = np.load(mc.sim_folder + 'E_loss/Gryzinski/Si/Si_2P_int_U.npy')


#%%
Si_val_total_U = np.load(mc.sim_folder + 'E_loss/diel_responce/Palik/Si_val_tot_U_P+G.npy')
Si_val_int_U = np.load(mc.sim_folder + 'E_loss/diel_responce/Palik/Si_val_int_U_P+G+A.npy')


#%% PMMA phonons and polarons
PMMA_phonon_U = np.load(mc.sim_folder + 'E_loss/phonons_polarons/PMMA_phonon_U.npy')
PMMA_polaron_U = np.load(mc.sim_folder + 'E_loss/phonons_polarons/PMMA_polaron_U.npy')


#%% Combine it all for PMMA
## elastic, valence, core_C, core_O, phonons, polarons
PMMA_processes_U_list = [PMMA_el_U, PMMA_val_total_U, PMMA_C_1S_total_U, PMMA_O_1S_total_U,\
                    PMMA_phonon_U, PMMA_polaron_U]

PMMA_processes_U = np.zeros((len(mc.EE), len(PMMA_processes_U_list)))

for i in range(len(PMMA_processes_U_list)):

    PMMA_processes_U[:, i] = PMMA_processes_U_list[i]


#%%
PMMA_processes_int_U = [PMMA_el_int_U, PMMA_val_int_U, PMMA_C_1S_int_U, PMMA_O_1S_int_U]


#%% Combine it all for Si
## elastic, valence, core_1S, core_2S, core_2P
Si_processes_U_list = [Si_el_U, Si_val_total_U, Si_1S_total_U, Si_2S_total_U, Si_2P_total_U]

Si_processes_U = np.zeros((len(mc.EE), len(Si_processes_U_list)))

for i in range(len(Si_processes_U_list)):

    Si_processes_U[:, i] = Si_processes_U_list[i]


#%%
Si_processes_int_U = [Si_el_int_U, Si_val_int_U, Si_1S_int_U, Si_2S_int_U, Si_2P_int_U]


#%%
processes_U = [PMMA_processes_U, Si_processes_U]


processes_int_U = [PMMA_processes_int_U, Si_processes_int_U]


#%%
PMMA_el_E_bind = np.zeros(len(EE)) ## dummy!!!
PMMA_val_E_bind = np.load(mc.sim_folder + 'E_loss/E_bind_PMMA/PMMA_E_bind.npy')
PMMA_C_1S_E_bind = np.ones(len(EE)) * mc.binding_C_1S
PMMA_O_1S_E_bind = np.ones(len(EE)) * mc.binding_O_1S

PMMA_E_bind = [PMMA_el_E_bind, PMMA_val_E_bind, PMMA_C_1S_E_bind, PMMA_O_1S_E_bind]


Si_el_E_bind = np.zeros(len(EE)) ## dummy!!!
Si_val_E_bind = np.load(mc.sim_folder + 'E_loss/E_bind_Si/Si_E_bind.npy')
Si_1S_E_bind = np.ones(len(EE)) * mc.binding_Si[0]
Si_2S_E_bind = np.ones(len(EE)) * mc.binding_Si[1]
Si_2P_E_bind = np.ones(len(EE)) * mc.binding_Si[2]

Si_E_bind = [Si_el_E_bind, Si_val_E_bind, Si_1S_E_bind, Si_2S_E_bind, Si_2P_E_bind]


E_bind = [PMMA_E_bind, Si_E_bind]

