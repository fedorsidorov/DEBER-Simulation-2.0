3#%% Import
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

#kJmol_2_eV = 1e+3 / (mc.Na * mc.eV)
kJmol_2_eV = 0.0103

MMA_bonds['Op-Cp'] = 815 * kJmol_2_eV,  4
MMA_bonds['O-Cp']  = 420 * kJmol_2_eV,  2
MMA_bonds['H-C3']  = 418 * kJmol_2_eV, 12
MMA_bonds['H-C2']  = 406 * kJmol_2_eV,  4
MMA_bonds['Cp-C']  = 373 * kJmol_2_eV,  2 ## 383-10 !!!!
MMA_bonds['O-C3']  = 364 * kJmol_2_eV,  2
MMA_bonds['C-C3']  = 356 * kJmol_2_eV,  2
MMA_bonds['C-C2']  = 354 * kJmol_2_eV,  4
MMA_bonds['justO'] = 13.62,             8

Eb_Nel = np.array(list(MMA_bonds.values()))


#%% mean binding energy = 6.42 eV
total_E = 0
total_N = 0

for key in MMA_bonds.keys():
    total_E += MMA_bonds[key][0]*MMA_bonds[key][1]
    total_N += MMA_bonds[key][1]

PMMA_Eb_mean = total_E / total_N


#%%
def get_stairway(b_map_sc, EE=mc.EE):

#    EE = mc.EE
#    b_map_sc = {'Op-Cp': 2}
    
    Eb_Nel_sc_list = []
    
    for val in b_map_sc.keys():
        Eb_Nel_sc_list.append([MMA_bonds[val][0], b_map_sc[val]])
    
    Eb_Nel_sc = np.array(Eb_Nel_sc_list)
    
    probs = np.zeros(len(EE))
    
    
    nums = np.zeros(len(EE))
    dens = np.zeros(len(EE))
    
    
    for i, e in enumerate(EE):
        
        num = 0
            
        for st in Eb_Nel_sc:
            if e >= st[0]:
                num += st[1]
        
        if num == 0:
            continue
        
        nums[i] = num
        
        den = 0
        
        for st in Eb_Nel:
            if e >= st[0]:
                den += st[1]
        
        dens[i] = den
        
        probs[i] = num / den
        
    
    return probs


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
#EE_low = np.linspace(3, 15, 1000)
#
#plt.figure()
#
#plt.plot(EE_low, get_stairway({'C-C2': 4, 'Cp-Cg': 0}, EE_low),\
#         label='room', linewidth=3)
#
#plt.plot(EE_low, get_stairway({'C-C2': 4, 'Cp-Cg': 2}, EE_low),\
#         '--', label='160$^\circ$', linewidth=3)
#
#plt.title('Scission probability')
#plt.xlabel('E, eV')
#plt.ylabel('scission probability')
#
#plt.xlim(3, 15)
#
#plt.legend()
#
#plt.grid()
#plt.show()

#plt.savefig('two_stairways.png', dpi=300)


#%%
def get_Gs_charlesby(T):
    
    inv_T = 1000 / (T + 273)
    
    k = -0.448036
    b = 1.98906
#    b = 2.14
    
    return np.exp(k*inv_T + b)

