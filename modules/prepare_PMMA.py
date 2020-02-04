#%% Import
import numpy as np
import my_constants as mc
import my_utilities as mu
import scission_functions as sf
import os

import importlib
mc = importlib.reload(mc)
mu = importlib.reload(mu)
sf = importlib.reload(sf)

#import matplotlib.pyplot as plt


#%%
EE = mc.EE
THETA = mc.THETA


#%% Elastic scattering
PMMA_el_U = np.load(os.path.join(mc.sim_folder, 
        'elastic', 'PMMA_elastic_U.npy'
        ))

PMMA_el_int_U = np.load(os.path.join(mc.sim_folder,
        'elastic', 'PMMA_elastic_int_CS.npy'
        ))


#%% Total inelastic U
## PMMA
PMMA_ee_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Dapor_2020', 'PMMA_ee_U_Dapor_Ashley.npy'
        ))

PMMA_ee_int_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Dapor_2020', 'PMMA_ee_int_U_Dapor_Ashley.npy'
        ))


#%% Core electron U components
## PMMA
PMMA_C_1S_total_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'PMMA_C_1S_U.npy'
        ))

PMMA_C_1S_int_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'PMMA_C_1S_int_U.npy'
        ))

PMMA_O_1S_total_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'PMMA_O_1S_U.npy'
        ))

PMMA_O_1S_int_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'PMMA_O_1S_int_U.npy'
        ))


#%%
PMMA_val_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Dapor_2020', 'PMMA_val_U_Dapor_Ashley-Gryzinski.npy'
        ))

PMMA_val_int_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Dapor_2020', 'PMMA_val_int_U_Dapor_Ashley-Gryzinski.npy'
        ))


#%%
#plt.loglog(mc.EE, PMMA_el_U, label='elastic')
#plt.loglog(mc.EE, PMMA_ee_U, label='ee')
#plt.loglog(mc.EE, PMMA_C_1S_total_U, label='C 1S')
#plt.loglog(mc.EE, PMMA_O_1S_total_U, label='O 1S')
#plt.loglog(mc.EE, PMMA_val_U, '--', label='val')

#plt.legend()


#%%
#plt.loglog(mc.EE, PMMA_ee_U-PMMA_val_U, label='core ee')

#plt.legend()


#%% PMMA phonons and polarons
PMMA_phonon_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'phonons_polarons', 'PMMA_phonon_U.npy'
        ))

PMMA_polaron_U = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'phonons_polarons', 'PMMA_polaron_U.npy'
        ))


#%% Combine it all for PMMA
## elastic, valence, core_C, core_O, phonons, polarons
PMMA_processes_U_list = [PMMA_el_U, PMMA_val_U, PMMA_C_1S_total_U, PMMA_O_1S_total_U,\
                    PMMA_phonon_U, PMMA_polaron_U]

PMMA_processes_U = np.zeros((len(mc.EE), len(PMMA_processes_U_list)))

for i in range(len(PMMA_processes_U_list)):

    PMMA_processes_U[:, i] = PMMA_processes_U_list[i]


#%%
PMMA_processes_int_U = [PMMA_el_int_U, PMMA_val_int_U, PMMA_C_1S_int_U, PMMA_O_1S_int_U]


#%%
PMMA_Eb_val = sf.PMMA_Eb_mean

PMMA_E_bind = [np.zeros(len(EE)), np.ones(len(EE)) * PMMA_Eb_val]


PMMA_el_E_bind = np.zeros(len(EE)) ## dummy!!!
PMMA_val_E_bind = np.ones(len(EE)) * PMMA_Eb_val
PMMA_C_1S_E_bind = np.ones(len(EE)) * mc.binding_C_1S
PMMA_O_1S_E_bind = np.ones(len(EE)) * mc.binding_O_1S

PMMA_E_bind = [PMMA_el_E_bind, PMMA_val_E_bind, PMMA_C_1S_E_bind, PMMA_O_1S_E_bind]


#%%
#plt.figure()
#
#plt.loglog(mc.EE, PMMA_el_U, label='elastic')
#plt.loglog(mc.EE, PMMA_val_U, label='valence ionization')
#plt.loglog(mc.EE, PMMA_phonon_U, label='electron-phonon')
#plt.loglog(mc.EE[:600], PMMA_polaron_U[:600], label='electron-polaron')
#
#plt.loglog(mc.EE, PMMA_C_1S_total_U, label='C 1S')
#plt.loglog(mc.EE, PMMA_O_1S_total_U, label='O 1S')
#
#plt.title('IMFP for processes in PMMA')
#plt.xlabel('E, eV')
#plt.ylabel('U, $\AA^{-1}$')
#
#plt.legend()
#
#plt.xlim(1e+0, 1e+4)
#plt.ylim(1e+1, 1e+9)
#
#plt.grid()

##plt.savefig('PMMA_processes.png', dpi=300)

