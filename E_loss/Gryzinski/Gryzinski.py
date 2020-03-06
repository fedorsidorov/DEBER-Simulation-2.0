#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt

import my_utilities as mu
import my_constants as mc
import scission_functions_2020 as sf

from scipy import integrate

mu = importlib.reload(mu)
mc = importlib.reload(mc)
sf = importlib.reload(sf)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski'
        ))


#%%
## PMMA           C,   O
MMA_core_Eb  = [296, 538]
MMA_core_occ = [  2,   2]

N_val_MMA = 40

N_H_MMA = 8
N_C_MMA = 5
N_O_MMA = 2

n_MMA =  mc.rho_PMMA * mc.Na/mc.u_MMA


## Si            1s,  2s,  2p
Si_core_Eb  = [1844, 154, 104]
Si_core_occ = [   2,   2,   6]

## Si             1s,  2s,  2p,    3s,      3p
Si_total_Eb  = [1844, 154, 104, 13.46, 8.15]
Si_total_occ = [   2,   2,   6,     2,    2]

Si_MuElec_Eb = [16.65, 6.52, 13.63, 107.98, 151.55, 1828.5]
Si_MuElec_occ = [4, 2, 2, 6, 2, 2]

#  energyConstant.push_back(16.65*eV);
#  energyConstant.push_back(6.52*eV); 
#  energyConstant.push_back(13.63*eV);
#  energyConstant.push_back(107.98*eV); 
#  energyConstant.push_back(151.55*eV); 
#  energyConstant.push_back(1828.5*eV);

N_val_Si = 4

n_Si = mc.rho_Si * mc.Na/mc.u_Si


#%% Gryzinski
def get_Gr_diff_cs(Eb, E, hw):
    
    if E < hw or hw < Eb:
        return 0
    
    diff_cs = np.pi * mc.k_el**2 * mc.e**4 / np.power(hw*mc.eV, 3) * Eb/E *\
        np.power(E / (E+Eb), 3/2) * np.power((1 - hw/E), Eb/(Eb + hw)) *\
        (hw/Eb * (1-Eb/E) + 4/3 * np.log(2.7 + np.sqrt((E - hw) / Eb)))
    
    return diff_cs * (100)**2 / (1/mc.eV) ## cm^2 / eV


def get_Gr_tau(Eb, E, hw, conc, n_el):
    return get_Gr_diff_cs(Eb, E, hw) * conc * n_el


def get_Gr_cs(Eb, E):
    
    def get_Y(hw):
        return get_Gr_diff_cs(Eb, E, hw)
    
    return integrate.quad(get_Y, Eb, (E + Eb)/2)[0]


def get_Gr_u(Eb, E, conc, n_el):
    return get_Gr_cs(Eb, E) * conc * n_el


def get_Gr_S(Eb, E, conc, n_el):
    
    def get_Y(hw):
        return get_Gr_diff_cs(Eb, E, hw) * hw
    
    return conc * n_el * integrate.quad(get_Y, Eb, (E + Eb)/2)[0]


#%%
## PMMA tau
def get_Gr_PMMA_C_core_tau(E, hw):
    return get_Gr_tau(MMA_core_Eb[0], E, hw, n_MMA, N_C_MMA * MMA_core_occ[0])


def get_Gr_PMMA_O_core_tau(E, hw):
    return get_Gr_tau(MMA_core_Eb[1], E, hw, n_MMA, N_O_MMA * MMA_core_occ[1])


def get_Gr_PMMA_core_tau(E, hw):
    return get_Gr_PMMA_C_core_tau(E, hw) + get_Gr_PMMA_O_core_tau(E, hw)


## PMMA u
def get_Gr_PMMA_C_core_u(E):
    return get_Gr_u(MMA_core_Eb[0], E, n_MMA, N_C_MMA * MMA_core_occ[0])


def get_Gr_PMMA_O_core_u(E):
    return get_Gr_u(MMA_core_Eb[1], E, n_MMA, N_O_MMA * MMA_core_occ[1])


def get_Gr_PMMA_core_u(E):
    return get_Gr_PMMA_C_core_u(E) + get_Gr_PMMA_O_core_u(E)


## PMMA S
def get_Gr_PMMA_C_core_S(E):
    return get_Gr_S(MMA_core_Eb[0], E, n_MMA, N_C_MMA * MMA_core_occ[0])


def get_Gr_PMMA_O_core_S(E):
    return get_Gr_S(MMA_core_Eb[1], E, n_MMA, N_O_MMA * MMA_core_occ[1])


def get_Gr_PMMA_core_S(E):
    return get_Gr_PMMA_C_core_S(E) + get_Gr_PMMA_O_core_S(E)


#%%
## Si tau
def get_Gr_Si_1s_tau(E, hw):
    return get_Gr_tau(Si_core_Eb[0], E, hw, n_Si, Si_core_occ[0])


def get_Gr_Si_2s_tau(E, hw):
    return get_Gr_tau(Si_core_Eb[1], E, hw, n_Si, Si_core_occ[1])


def get_Gr_Si_2p_tau(E, hw):
    return get_Gr_tau(Si_core_Eb[2], E, hw, n_Si, Si_core_occ[2])


def get_Gr_Si_core_tau(E, hw):
    return get_Gr_Si_1s_tau(E, hw) + get_Gr_Si_2s_tau(E, hw) + get_Gr_Si_2p_tau(E, hw)


## Si u
def get_Gr_Si_1s_u(E):
    return get_Gr_u(Si_core_Eb[0], E, n_Si, Si_core_occ[0])


def get_Gr_Si_2s_u(E):
    return get_Gr_u(Si_core_Eb[1], E, n_Si, Si_core_occ[1])


def get_Gr_Si_2p_u(E):
    return get_Gr_u(Si_core_Eb[2], E, n_Si, Si_core_occ[2])


def get_Gr_Si_core_u(E):
    return get_Gr_Si_1s_u(E) + get_Gr_Si_2s_u(E) + get_Gr_Si_2p_u(E)


def get_Gr_Si_total_u(E):
    return get_Gr_Si_core_u(E) +\
        get_Gr_u(Si_total_Eb[3], E, n_Si, Si_total_occ[3]) +\
        get_Gr_u(Si_total_Eb[4], E, n_Si, Si_total_occ[4])


## Si S
def get_Gr_Si_1s_S(E):
    return get_Gr_S(Si_core_Eb[0], E, n_Si, Si_core_occ[0])


def get_Gr_Si_2s_S(E):
    return get_Gr_S(Si_core_Eb[1], E, n_Si, Si_core_occ[1])


def get_Gr_Si_2p_S(E):
    return get_Gr_S(Si_core_Eb[2], E, n_Si, Si_core_occ[2])


def get_Gr_Si_core_S(E):
    return get_Gr_Si_1s_S(E) + get_Gr_Si_2s_S(E) + get_Gr_Si_2p_S(E)


def get_Gr_Si_total_S(E):
    return get_Gr_Si_core_S(E) +\
        get_Gr_S(Si_total_Eb[3], E, n_Si, Si_total_occ[3]) +\
        get_Gr_S(Si_total_Eb[4], E, n_Si, Si_total_occ[4])


#%%
def get_scission_probs_gryz_single_E(E):
    
    gryz_bond_u = np.zeros(len(sf.MMA_bonds))
    
    
    for i in range(len(sf.MMA_bonds)):
        gryz_bond_u[i] = get_Gr_u(
                sf.MMA_bonds[list(sf.MMA_bonds.keys())[i]][0],
                E,
                mc.n_PMMA_mon,
                sf.MMA_bonds[list(sf.MMA_bonds.keys())[i]][1]
                )
    
    
    gryz_probs = np.zeros(np.shape(gryz_bond_u))
    
    
    for i in range(len(gryz_probs)):
        now_sum = np.sum(gryz_bond_u)
        
        if now_sum == 0:
            continue
        
        gryz_probs[i] = gryz_bond_u[i] / now_sum
        
    
    return gryz_probs


#%%
#scission_probs = np.zeros((len(mc.EE), sf.MMA_n_bonds))
#
#for i, E in enumerate(mc.EE):
#    
#    mu.pbar(i, len(mc.EE))
#    
#    scission_probs[i, :] = get_scission_probs_gryz_single_E(E)


#%%
#EE = mc.EE

#tau_1s = np.zeros((len(mc.EE), len(mc.EE)))
#tau_2s = np.zeros((len(mc.EE), len(mc.EE)))
#tau_2p = np.zeros((len(mc.EE), len(mc.EE)))

#u_C_PMMA = np.zeros(len(EE))
#u_O_PMMA = np.zeros(len(EE))
#S_C_PMMA = np.zeros(len(EE))
#S_O_PMMA = np.zeros(len(EE))

#u1_Si = np.zeros(len(EE))
#u2_Si = np.zeros(len(EE))
#u3_Si = np.zeros(len(EE))
#u4_Si = np.zeros(len(EE))
#u5_Si = np.zeros(len(EE))
#u6_Si = np.zeros(len(EE))


#for i, E in enumerate(mc.EE):

#    mu.pbar(i, len(mc.EE))
    
#    u_C_PMMA[i] = get_Gr_PMMA_C_core_u(E)
#    u_O_PMMA[i] = get_Gr_PMMA_O_core_u(E)
#    S_C_PMMA[i] = get_Gr_PMMA_C_core_S(E)
#    S_O_PMMA[i] = get_Gr_PMMA_O_core_S(E)
    
#    u1_Si[i] = get_Gr_S(Si_MuElec_Eb[0], E, n_Si, Si_MuElec_occ[0])
#    u2_Si[i] = get_Gr_S(Si_MuElec_Eb[1], E, n_Si, Si_MuElec_occ[1])
#    u3_Si[i] = get_Gr_S(Si_MuElec_Eb[2], E, n_Si, Si_MuElec_occ[2])
#    u4_Si[i] = get_Gr_S(Si_MuElec_Eb[3], E, n_Si, Si_MuElec_occ[3])
#    u5_Si[i] = get_Gr_S(Si_MuElec_Eb[4], E, n_Si, Si_MuElec_occ[4])
#    u6_Si[i] = get_Gr_S(Si_MuElec_Eb[5], E, n_Si, Si_MuElec_occ[5])
    
#    for j, hw in enumerate(mc.EE):
#        tau_1s[i, j] = get_Gr_Si_1s_tau(E, hw)
#        tau_2s[i, j] = get_Gr_Si_2s_tau(E, hw)
#        tau_2p[i, j] = get_Gr_Si_2p_tau(E, hw)


#%%
#EE = np.logspace(0, 4.4, 100)
#
#ans = np.zeros(len(EE))
#bns = np.zeros(len(EE))
#cns = np.zeros(len(EE))
#
#
#for i, E in enumerate(EE):
#    
#    mu.pbar(i, len(EE))
#    
#    ans[i] = get_Gr_PMMA_core_u(E)
##    bns[i] = get_Gr_Si_2p_S(E)
#    
#    cns[i] = elf.get_PMMA_Gryzinski_core_U(np.array([E]))
##    cns[i] = elf.get_PMMA_C_1S_U(np.array([E]))
#
#
#plt.loglog(EE, ans, '.', label='now core')
#plt.loglog(EE, cns, '--', label='old core')
#
##plt.loglog(EE, ans - bns)
#
#plt.legend()

