#%% Import
import numpy as np
import os

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
Si_el_U = np.load(os.path.join(mc.sim_folder,
        'elastic', 'Si_elastic_U.npy'
        ))

Si_el_int_U = np.load(os.path.join(mc.sim_folder,
        'elastic', 'Si_elastic_int_CS.npy'
        ))


#%% Total inelastic U
Si_ee_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Palik_2020', 'Si_ee_U_Palik_Ashley.npy'
        ))


#%%
## Si
Si_1S_total_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'Si_1S_U.npy'
        ))

Si_1S_int_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'Si_1S_int_U.npy'
        ))

Si_2S_total_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'Si_2S_U.npy'
        ))

Si_2S_int_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'Si_2S_int_U.npy'
        ))

Si_2P_total_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'Si_2P_U.npy'
        ))

Si_2P_int_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'Si', 'Si_2P_int_U.npy'
        ))


#%%
Si_val_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Palik_2020', 'Si_val_U_Palik_Ashley-Gryzinski.npy'
        ))

Si_val_int_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Palik_2020', 'Si_val_int_U_Palik_Ashley-Gryzinski.npy'
        ))


#%%
#plt.loglog(EE, Si_el_U, label='elastic')
#plt.loglog(EE, Si_ee_U, label='ee')
#plt.loglog(EE, Si_1S_total_U, label='1S')
#plt.loglog(EE, Si_2S_total_U, label='2S')
#plt.loglog(EE, Si_2P_total_U, label='2P')
#plt.loglog(EE, Si_val_U, '--', label='valence')
#
#plt.legend()


#%% Combine it all for Si
## elastic, valence, core_1S, core_2S, core_2P
Si_processes_U_list = [Si_el_U, Si_val_U, Si_1S_total_U, Si_2S_total_U, Si_2P_total_U]

Si_processes_U = np.zeros((len(mc.EE), len(Si_processes_U_list)))

for i in range(len(Si_processes_U_list)):

    Si_processes_U[:, i] = Si_processes_U_list[i]


#%%
Si_processes_int_U = [Si_el_int_U, Si_val_int_U, Si_1S_int_U, Si_2S_int_U, Si_2P_int_U]


Si_el_E_bind = np.zeros(len(EE)) ## dummy!!!
Si_val_E_bind = np.load(os.path.join(mc.sim_folder, 'E_loss', 'E_bind_Si', 'Si_E_bind_2020.npy'))
Si_1S_E_bind = np.ones(len(EE)) * mc.binding_Si[0]
Si_2S_E_bind = np.ones(len(EE)) * mc.binding_Si[1]
Si_2P_E_bind = np.ones(len(EE)) * mc.binding_Si[2]

Si_E_bind = [Si_el_E_bind, Si_val_E_bind, Si_1S_E_bind, Si_2S_E_bind, Si_2P_E_bind]

