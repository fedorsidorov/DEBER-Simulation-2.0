3#%% Import
import numpy as np
#import os
import importlib
import matplotlib.pyplot as plt
import matplotlib

import my_constants as mc
mc = importlib.reload(mc)

import my_utilities as mu
mu = importlib.reload(mu)

import Gryzinski as gryz
gryz = importlib.reload(gryz)


#%%
kJmol_2_eV = 1e+3 / (mc.Na * mc.eV)
#kJmol_2_eV = 0.0103

MMA_bonds = {}

MMA_bonds["Oval"]  = 13.62           ,  8
MMA_bonds["C'-O'"] = 815 * kJmol_2_eV,  4
MMA_bonds["C'-O"]  = 420 * kJmol_2_eV,  2
MMA_bonds["C3-H"]  = 418 * kJmol_2_eV, 12
MMA_bonds["C2-H"]  = 406 * kJmol_2_eV,  4
MMA_bonds["C-C'"]  = 373 * kJmol_2_eV,  2 ## 383-10 !!!!
MMA_bonds["O-C3"]  = 364 * kJmol_2_eV,  2
MMA_bonds["C-C3"]  = 356 * kJmol_2_eV,  2
MMA_bonds["C-C2"]  = 354 * kJmol_2_eV,  4

n_bonds = len(MMA_bonds)
bond_inds = list(range(n_bonds))

bond_names = list(MMA_bonds.keys())

BDE_array = np.array(list(MMA_bonds.values()))

bonds_BDE = BDE_array[:, 0]
bonds_occ = BDE_array[:, 1]

#bond_names = list(MMA_bonds.keys())
Eb_Nel = np.array(list(MMA_bonds.values()))


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
def get_bonds_u(EE):
    
    bonds_u = np.zeros((len(EE), n_bonds))
    
    
    for n in range(n_bonds):
        
        mu.pbar(n, n_bonds)
        
        Eb = bonds_BDE[n]
        occ = bonds_occ[n]
        
        for i in range(len(EE)):
            bonds_u[i, n] = gryz.get_Gr_u(Eb, EE[i], mc.n_PMMA_mon, occ)
        
        plt.loglog(EE, bonds_u[:, n], label=list(MMA_bonds.keys())[n])
    
    return bonds_u
    

#%%
def get_scission_probs_gryz_single_E(E):
    
    gryz_bond_u = np.zeros(len(MMA_bonds))
    
    
    for i in range(len(MMA_bonds)):
        gryz_bond_u[i] = gryz.get_Gr_u(
                MMA_bonds[list(MMA_bonds.keys())[i]][0],
                E,
                mc.n_PMMA_mon,
                MMA_bonds[list(MMA_bonds.keys())[i]][1]
                )
    
    
    gryz_probs = np.zeros(np.shape(gryz_bond_u))
    
    
    for i in range(len(gryz_probs)):
        now_sum = np.sum(gryz_bond_u)
        
        if now_sum == 0:
            continue
        
        gryz_probs[i] = gryz_bond_u[i] / now_sum
        
    
    return gryz_probs


#%%
#scission_probs = np.zeros((len(mc.EE), n_bonds))
#
#for i, E in enumerate(mc.EE):
#    
#    mu.pbar(i, len(mc.EE))
#    
#    scission_probs[i, :] = get_scission_probs_gryz_single_E(E)


#%%
#font_size = 14
#
#plt.figure(figsize=[5, 5])
#
#matplotlib.rcParams['font.family'] = 'Times New Roman'
#
#plt.legend(fontsize=font_size, loc='lower right')
#
#ax = plt.gca()
#for tick in ax.xaxis.get_major_ticks():
#    tick.label.set_fontsize(font_size)
#for tick in ax.yaxis.get_major_ticks():
#    tick.label.set_fontsize(font_size)
#
#
#
#for i in range(1, len(MMA_bonds)):
#    plt.loglog(mc.EE, scission_probs[:, i], label=bond_names[i])
#
#
#plt.xlabel('E, eV', fontsize=font_size)
#plt.ylabel('bond weight', fontsize=font_size)
#
#plt.xlim(1e+0, 1e+4)
#plt.ylim(1e-2, 1)
#
#plt.legend(loc='upper right', fontsize=font_size)
#plt.grid()
#
#plt.savefig('probs.png', dpi=600)


#%%
#def get_stairway_Gryzinski(bond_dict_sc, EE=mc.EE):
#    
#    gryz_probs = scission_probs_gryz(EE)
#    
#    probs = np.zeros(len(EE))
#    
#    
#    for b in bond_dict_sc:
#        
#        bond_ind = np.where(Eb_Nel[:, 0] == MMA_bonds[b][0])[0][0]    
#        probs += gryz_probs[:, bond_ind] * bond_dict_sc[b] / MMA_bonds[b][1]
#        
#
#    return probs


#%%
#bond_dict_sc = {"C-C2": 4}
#bond_dict_sc = {"C-C2": 4, "C-C'": 2}

#end_ind = 300

#EE = np.linspace(1, 10, 10000)
#EE = mc.

#plt.plot(EE, get_stairway_Gryzinski(bond_dict_sc, EE), '--')



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

