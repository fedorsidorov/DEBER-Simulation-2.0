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

os.chdir(mc.sim_folder + 'all_Gryzinski')


#%%
MMA_bonds = {}

kJmol_2_eV = 1e+3 / (mc.Na * mc.eV)

MMA_bonds['Op-Cp'] = 815 * kJmol_2_eV, 4
MMA_bonds['O-Cp'] = 420  * kJmol_2_eV, 2
MMA_bonds['H-C3'] = 418  * kJmol_2_eV, 12
MMA_bonds['H-C2'] = 406  * kJmol_2_eV, 4
MMA_bonds['Cp-Cg'] = 383 * kJmol_2_eV, 2
MMA_bonds['O-C3'] = 364  * kJmol_2_eV, 4
MMA_bonds['C-C3'] = 356  * kJmol_2_eV, 2
MMA_bonds['C-C2'] = 354  * kJmol_2_eV, 4


#%%
PMMA_total_inel_U = np.load(mc.sim_folder + 'E_loss/diel_responce/Dapor/PMMA_U_Dapor.npy')
#PMMA_diff_inel_U = np.load(mc.sim_folder +\
#                           'E_loss/diel_responce/Dapor/PMMA_diff_U_Dapor_Ashley.npy')

PMMA_SP = np.load(mc.sim_folder + 'E_loss/diel_responce/Dapor/PMMA_SP_Dapor.npy')


#%% Go Fryzinski
total_U = np.zeros(len(mc.EE))
total_SP = np.zeros(len(mc.EE))


for bond in MMA_bonds:
    
    total_U += elf.get_Gryzinski_CS(mc.EE, MMA_bonds[bond][0]) * MMA_bonds[bond][1] * mc.n_PMMA_mon
    total_SP += elf.get_Gryzinski_SP(mc.EE, MMA_bonds[bond][0], mc.n_PMMA_mon, MMA_bonds[bond][1])


#%% U
plt.loglog(mc.EE, PMMA_total_inel_U, label='Dapor')
plt.loglog(mc.EE, total_U, label='Gryzinski + BDE')

plt.title('PMMA Dapor and Gryz+BDE U')
plt.xlabel('E, eV')
plt.ylabel('U, cm$^{-1}$')

plt.legend()
plt.grid()

#plt.savefig('PMMA_Dapor_Gryz+BDE_U.png', dpi=300)


#%% SP
plt.loglog(mc.EE, PMMA_SP, label='Dapor')
plt.loglog(mc.EE, total_SP, label='Gryzinski + BDE')

plt.xlabel('E, eV')
plt.ylabel('SP, eV/cm')

plt.legend()
plt.grid()

#plt.savefig('PMMA_Dapor_Gryz+BDE_SP.png', dpi=300)


#%% Gryzinski stairway
gryz_bond_U = np.zeros((len(MMA_bonds), len(mc.EE)))


for i in range(len(MMA_bonds)):
    
    gryz_bond_U[i, :] = elf.get_Gryzinski_CS(mc.EE, MMA_bonds[list(MMA_bonds.keys())[i]][0]) *\
        MMA_bonds[list(MMA_bonds.keys())[i]][1] * mc.n_PMMA_mon
    
    plt.loglog(mc.EE, gryz_bond_U[i, :], label=list(MMA_bonds.keys())[i])


plt.title('PMMA Dapor and Gryz+BDE bond CS for each bond')
plt.xlabel('E, eV')
plt.ylabel('U, cm$^{-1}$')

plt.ylim(1e+5, 1e+8)

plt.legend()
plt.grid()

#plt.savefig('PMMA_Dapor_Gryz+BDE_U_bonds.png', dpi=300)


#%%
def get_w_scission(EE):
    
    result = np.zeros(len(EE))
    
    result = np.ones(len(EE)) * 4/40
    result[np.where(EE < 815 * 0.0103)] = 4/(40 - 8)
    result[np.where(EE < 420 * 0.0103)] = 4/(40 - 8 - 4)
    result[np.where(EE < 418 * 0.0103)] = 4/(40 - 8 - 4 - 12)
    result[np.where(EE < 406 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4)
    result[np.where(EE < 383 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2)
    result[np.where(EE < 364 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4)
    result[np.where(EE < 356 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4 - 2)
    result[np.where(EE < 354 * 0.0103)] = 0
    
    return result


probs_easy = get_w_scission(mc.EE)


#%%
probs = np.zeros(len(mc.EE))


for i in range(len(probs)):
    
    if np.sum(gryz_bond_U[:, i]) == 0:
        continue
    
    probs[i] = np.sum(gryz_bond_U[-2:, i]) / np.sum(gryz_bond_U[:, i])
    

end_ind = 200

plt.plot(mc.EE[:end_ind], probs_easy[:end_ind], label='basic')
plt.plot(mc.EE[:end_ind], probs[:end_ind], label='Gryzinsky')

plt.title('Scission probability')
plt.xlabel('E, eV')
plt.ylabel('p')

plt.legend()
plt.grid()

plt.savefig('scission_probs.png', dpi=300)

