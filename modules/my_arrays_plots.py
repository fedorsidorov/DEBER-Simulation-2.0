#%% Import
import numpy as np
import my_constants as mc
import my_utilities as mu

import importlib
mc = importlib.reload(mc)
mu = importlib.reload(mu)

import matplotlib.pyplot as plt


#%%
EE = mc.EE
THETA = mc.THETA


#%% Elastic scattering
PMMA_el_U = np.load(mc.sim_path_MAC + 'elastic/PMMA_elastic_U.npy')
PMMA_el_int_U = np.load(mc.sim_path_MAC + 'elastic/PMMA_elastic_int_CS.npy')

Si_el_U = np.load(mc.sim_path_MAC + 'elastic/Si_elastic_U.npy')
Si_el_int_U = np.load(mc.sim_path_MAC + 'elastic/Si_elastic_int_CS.npy')


#%% Total inelastic U
## PMMA
PMMA_total_inel_U = np.load(mc.sim_path_MAC + 'E_loss/diel_responce/Dapor/PMMA_U_Dapor.npy')
PMMA_diff_inel_U = np.load(mc.sim_path_MAC +\
                           'E_loss/diel_responce/Dapor/PMMA_diff_U_Dapor_Ashley.npy')

## Si
Si_total_inel_U = np.load(mc.sim_path_MAC + 'E_loss/diel_responce/Palik/Si_U_Palik.npy')
Si_diff_inel_U = np.load(mc.sim_path_MAC +\
                         'E_loss/diel_responce/Palik/Si_diff_U_Palik_Ashley.npy')


#%% Core electron U components
## PMMA
PMMA_C_1S_total_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/PMMA/PMMA_C_1S_U.npy')
PMMA_C_1S_diff_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/PMMA/PMMA_C_1S_diff_U.npy')
PMMA_C_1S_int_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/PMMA/PMMA_C_1S_int_U.npy')

PMMA_O_1S_total_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/PMMA/PMMA_O_1S_U.npy')
PMMA_O_1S_diff_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/PMMA/PMMA_O_1S_diff_U.npy')
PMMA_O_1S_int_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/PMMA/PMMA_O_1S_int_U.npy')


#%%
#PMMA_core_diff_U = PMMA_C_1S_diff_U + PMMA_O_1S_diff_U
#PMMA_val_diff_U = PMMA_diff_inel_U - PMMA_core_diff_U
#
#PMMA_val_diff_U[np.where(PMMA_val_diff_U < 0)] = 0
#
#PMMA_val_int_U = mu.diff2int(PMMA_val_diff_U)


#%%
PMMA_val_total_U = np.load(mc.sim_path_MAC + 'E_loss/diel_responce/Dapor/PMMA_val_tot_U_D+G.npy')
PMMA_val_int_U = np.load(mc.sim_path_MAC + 'E_loss/diel_responce/Dapor/PMMA_val_int_U_D+G+A.npy')


#%%
## Si
Si_1S_total_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_1S_U.npy')
Si_1S_diff_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_1S_diff_U.npy')
Si_1S_int_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_1S_int_U.npy')

Si_2S_total_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_2S_U.npy')
Si_2S_diff_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_2S_diff_U.npy')
Si_2S_int_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_2S_int_U.npy')

Si_2P_total_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_2P_U.npy')
Si_2P_diff_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_2P_diff_U.npy')
Si_2P_int_U = np.load(mc.sim_path_MAC + 'E_loss/Gryzinski/Si/Si_2P_int_U.npy')


#%%
Si_core_diff_U = Si_1S_diff_U + Si_2S_diff_U + Si_2P_diff_U
Si_val_diff_U = Si_diff_inel_U - Si_core_diff_U

Si_val_diff_U[np.where(Si_val_diff_U < 0)] = 0

Si_val_int_U = mu.diff2int(Si_val_diff_U)


#%%
#ind = 800
#
#plt.loglog(EE, Si_core_diff_U[ind, :], label='core')
#plt.loglog(EE, Si_diff_inel_U[ind, :], label='total')
#plt.grid()
#plt.legend()
#plt.show()


Si_val_total_U = np.load(mc.sim_path_MAC + 'E_loss/diel_responce/Palik/Si_val_tot_U_P+G.npy')
#Si_val_int_U = np.load(mc.sim_path_MAC + 'E_loss/diel_responce/Palik/Si_val_int_U_P+G.npy')




#%%
#plt.loglog(EE, PMMA_diff_inel_U[-1, :])
#plt.loglog(EE, PMMA_C_1S_diff_U[-1, :] + PMMA_O_1S_diff_U[-1, :])


#%%
#ind = 500

#plt.loglog(EE, Si_diff_inel_U[ind, :])
#plt.loglog(EE, Si_1S_diff_U[ind, :] + Si_2S_diff_U[ind, :] + Si_2P_diff_U[ind, :])


#%%
#ind = 500
#
#plt.loglog(EE, PMMA_diff_inel_U[ind, :], label='PMMA')
#plt.loglog(EE, Si_diff_inel_U[ind, :], label='Si')
#
#plt.legend()
#plt.grid()
#plt.show()


#%% PMMA phonons and polarons
PMMA_phonon_U = np.load(mc.sim_path_MAC + 'E_loss/phonons_polarons/PMMA_phonon_U.npy')
PMMA_polaron_U = np.load(mc.sim_path_MAC + 'E_loss/phonons_polarons/PMMA_polaron_U.npy')


#%% Combine it all for PMMA
## elastic, valence, core_C, core_O, phonons, polarons
PMMA_processes_U_list = [PMMA_el_U, PMMA_val_total_U, PMMA_C_1S_total_U, PMMA_O_1S_total_U,\
                    PMMA_phonon_U, PMMA_polaron_U]

PMMA_processes_U = np.zeros((len(mc.EE), len(PMMA_processes_U_list)))

for i in range(len(PMMA_processes_U_list)):

    PMMA_processes_U[:, i] = PMMA_processes_U_list[i]


#for U in PMMA_processes_U_list:
#    
#    plt.loglog(EE, U)
#
#plt.xlabel('E, eV')
#plt.ylabel('U, cm$^{-1}$')
#
#plt.ylim(1e+1, 1e+9)
#
#plt.grid()


#%%
PMMA_processes_int_U = [PMMA_el_int_U, PMMA_val_int_U, PMMA_C_1S_int_U, PMMA_O_1S_int_U]


#%% Combine it all for Si
## elastic, valence, core_1S, core_2S, core_2P
Si_processes_U_list = [Si_el_U, Si_val_total_U, Si_1S_total_U, Si_2S_total_U, Si_2P_total_U]

Si_processes_U = np.zeros((len(mc.EE), len(Si_processes_U_list)))

for i in range(len(Si_processes_U_list)):

    Si_processes_U[:, i] = Si_processes_U_list[i]

#for U in Si_processes_U_list:
#    
#    plt.loglog(EE, U)
#
#plt.xlabel('E, eV')
#plt.ylabel('U, cm$^{-1}$')
#
#plt.ylim(1e+1, 1e+9)
#
#plt.grid()


#%%
Si_processes_int_U = [Si_el_int_U, Si_val_int_U, Si_1S_int_U, Si_2S_int_U, Si_2P_int_U]


#%%
processes_U = [PMMA_processes_U, Si_processes_U]

processes_int_U = [PMMA_processes_int_U, Si_processes_int_U]


#%%
PMMA_val_E_bind = np.load(mc.sim_path_MAC + 'E_loss/E_bind_PMMA/PMMA_E_bind.npy')
PMMA_C_1S_E_bind = np.ones(len(EE)) * mc.binding_C_1S
PMMA_O_1S_E_bind = np.ones(len(EE)) * mc.binding_O_1S

PMMA_E_bind = [PMMA_val_E_bind, PMMA_C_1S_E_bind, PMMA_O_1S_E_bind]

Si_val_E_bind = np.load(mc.sim_path_MAC + 'E_loss/E_bind_Si/Si_E_bind.npy')
Si_1S_E_bind = np.ones(len(EE)) * mc.binding_Si[0]
Si_2S_E_bind = np.ones(len(EE)) * mc.binding_Si[1]
Si_2P_E_bind = np.ones(len(EE)) * mc.binding_Si[2]

Si_E_bind = [Si_val_E_bind, Si_1S_E_bind, Si_2S_E_bind, Si_2P_E_bind]

E_bind = [PMMA_E_bind, Si_E_bind]







