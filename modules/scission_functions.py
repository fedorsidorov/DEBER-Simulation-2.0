#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy
import numpy.random as rnd
import my_arrays_Dapor as ma

ma = importlib.reload(ma)

import my_constants as mc
mc = importlib.reload(mc)

import E_loss_functions as elf
elf = importlib.reload(elf)


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
def scission_probs_gryz(EE):
    
    gryz_bond_U = np.zeros((len(MMA_bonds), len(EE)))
    
    
    for i in range(len(MMA_bonds)):
        
        gryz_bond_U[i, :] = elf.get_Gryzinski_CS(EE, MMA_bonds[list(MMA_bonds.keys())[i]][0]) *\
            MMA_bonds[list(MMA_bonds.keys())[i]][1] * mc.n_PMMA_mon
        
#        plt.loglog(EE, gryz_bond_U[i, :], label=list(MMA_bonds.keys())[i])
    
    
    probs = np.zeros(len(EE))


    for i in range(len(probs)):
        
        if np.sum(gryz_bond_U[:, i]) == 0:
            continue
        
        probs[i] = np.sum(gryz_bond_U[-2:, i]) / np.sum(gryz_bond_U[:, i])
    
    
    return probs


#%%
def scission_probs_ones(EE):
    
    return np.ones(len(EE))


def scission_probs_2CC(EE):
    
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


def scission_probs_2CC_ester(EE):

    result = np.ones(len(EE)) * 6/40
    result[np.where(EE < 815 * 0.0103)] = 6/(40 - 8)
    result[np.where(EE < 420 * 0.0103)] = 6/(40 - 8 - 4)
    result[np.where(EE < 418 * 0.0103)] = 6/(40 - 8 - 4 - 12)
    result[np.where(EE < 406 * 0.0103)] = 6/(40 - 8 - 4 - 12 - 4)
    result[np.where(EE < 383 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2)
    result[np.where(EE < 364 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4)
    result[np.where(EE < 356 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4 - 2)
    result[np.where(EE < 354 * 0.0103)] = 0
    
    return result


def scission_probs_2CC_3H(EE):
    
    result = np.ones(len(EE)) * (4 + 6)/40
    result[np.where(EE < 815 * 0.0103)] = (4 + 6)/(40 - 8)
    result[np.where(EE < 420 * 0.0103)] = (4 + 6)/(40 - 8 - 4)
    result[np.where(EE < 418 * 0.0103)] = 4/(40 - 8 - 4 - 12)
    result[np.where(EE < 406 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4)
    result[np.where(EE < 383 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2)
    result[np.where(EE < 364 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4)
    result[np.where(EE < 356 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4 - 2)
    result[np.where(EE < 354 * 0.0103)] = 0
    
    return result


## 160 C
def scission_probs_2CC_2H(EE):
        
    result =            np.ones(len(EE))* (4 + 4)/40
    result[np.where(EE < 815 * 0.0103)] = (4 + 4)/(40 - 8)
    result[np.where(EE < 420 * 0.0103)] = (4 + 4)/(40 - 8 - 4)
    result[np.where(EE < 418 * 0.0103)] =      4 / (40 - 8 - 4 - 12)
    result[np.where(EE < 406 * 0.0103)] =      4 / (40 - 8 - 4 - 12 - 4)
    result[np.where(EE < 383 * 0.0103)] =      4 / (40 - 8 - 4 - 12 - 4 - 2)
    result[np.where(EE < 364 * 0.0103)] =      4 / (40 - 8 - 4 - 12 - 4 - 2 - 4)
    result[np.where(EE < 356 * 0.0103)] =      4 / (40 - 8 - 4 - 12 - 4 - 2 - 4 - 2)
    result[np.where(EE < 354 * 0.0103)] = 0
    
    return result


def scission_probs_2CC_1p5H(EE):

    result = np.ones(len(EE)) * (4 + 3)/40
    result[np.where(EE < 815 * 0.0103)] = (4 + 3)/(40 - 8)
    result[np.where(EE < 420 * 0.0103)] = (4 + 3)/(40 - 8 - 4)
    result[np.where(EE < 418 * 0.0103)] = 4/(40 - 8 - 4 - 12)
    result[np.where(EE < 406 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4)
    result[np.where(EE < 383 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2)
    result[np.where(EE < 364 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4)
    result[np.where(EE < 356 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4 - 2)
    result[np.where(EE < 354 * 0.0103)] = 0
    
    return result


#%%
#end_ind = 200
#
#plt.plot(ma.EE[:end_ind], scission_probs_2CC_ester_H(ma.EE[:end_ind]))
#
#plt.title('Scission probability')
#plt.xlabel('E, eV')
#plt.ylabel('scission probability')
#
#plt.grid()


