3#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import matplotlib
import copy

import my_constants as mc
mc = importlib.reload(mc)

import E_loss_functions_2020 as elf
elf = importlib.reload(elf)


#%%
#kJmol_2_eV = 1e+3 / (mc.Na * mc.eV)
kJmol_2_eV = 0.0103

MMA_bonds = {}

#MMA_bonds["Oval"] = 13.62,             8
MMA_bonds["C'-O'"] = 815 * kJmol_2_eV,  12
MMA_bonds["C'-O"]  = 420 * kJmol_2_eV,  2
MMA_bonds["C3-H"]  = 418 * kJmol_2_eV, 12
MMA_bonds["C2-H"]  = 406 * kJmol_2_eV,  4
MMA_bonds["C-C'"]  = 383 * kJmol_2_eV,  2 ## 383 !!!!
MMA_bonds["O-C3"]  = 364 * kJmol_2_eV,  2
MMA_bonds["C-C3"]  = 356 * kJmol_2_eV,  2
MMA_bonds["C-C2"]  = 354 * kJmol_2_eV,  4

#bond_names = list(MMA_bonds.keys())
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
    
    gryz_bond_U = np.zeros((len(EE), len(MMA_bonds)))
    
    
    for i in range(len(MMA_bonds)):
        
        gryz_bond_U[:, i] = elf.get_Gryzinski_CS(EE, MMA_bonds[list(MMA_bonds.keys())[i]][0],\
                WW=mc.WW_ext) * MMA_bonds[list(MMA_bonds.keys())[i]][1] * mc.n_PMMA_mon
        
#        plt.loglog(EE, gryz_bond_U[:, i], label=list(MMA_bonds.keys())[i])
    
    gryz_probs = np.zeros(np.shape(gryz_bond_U))
    
    
    for i in range(len(gryz_probs)):
        
        now_sum = np.sum(gryz_bond_U[i])
        
        if now_sum == 0:
            continue
        
        gryz_probs[i] = gryz_bond_U[i] / now_sum
        
        
    return gryz_probs


#%%
def get_stairway_Gryzinski(bond_dict_sc, EE=mc.EE):
    
    gryz_probs = scission_probs_gryz(EE)
    
    probs = np.zeros(len(EE))
    
    
    for b in bond_dict_sc:
        
        bond_ind = np.where(Eb_Nel[:, 0] == MMA_bonds[b][0])[0][0]    
        probs += gryz_probs[:, bond_ind] * bond_dict_sc[b] / MMA_bonds[b][1]
        

    return probs


#%%
matplotlib.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=[3.1496, 3.1496])
font_size = 10

bond_dict_sc_room = {"C-C2": 4}
bond_dict_sc_160 = {"C-C2": 4, "C3-H": 4}

end_ind = 237

#EE = np.linspace(1, 10, 10000)
#EE = mc.

plt.plot(mc.EE[:end_ind], get_stairway(bond_dict_sc_room, mc.EE[:end_ind]), label='room temperature')
plt.plot(mc.EE[:end_ind], get_stairway(bond_dict_sc_160, mc.EE[:end_ind]), 'r--',
         label='160 $^\circ$C')

plt.title('PMMA chain scission probability')
plt.xlabel('incident electron erengy, eV')
plt.ylabel('scission probability')

plt.grid()

plt.legend(fontsize=font_size)

plt.xlim(2.5, 10)
plt.ylim(0, 1)

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)


#plt.savefig('sci_prob_abstract.tiff', bbox_inches='tight', dpi=1000)


#%%
#gryz_bond_U = scission_probs_gryz(mc.EE)
#
#
#for i in range(len(gryz_bond_U[0])):
#    
#    plt.loglog(mc.EE, gryz_bond_U[:, i], label=list(MMA_bonds.keys())[i])
#
#
#plt.title('Gryzinski cross-sections for PMMA valence electrons')
#plt.xlabel('E, eV')
#plt.ylabel('U, $\AA^{-1}$')
#
#plt.xlim(1, 1e+4)
#plt.ylim(1e+4, 1e+8)
#
#plt.legend()
#plt.grid()

#plt.savefig('Gryzinski_val.png', dpi=300)


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

