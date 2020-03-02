#%% Import
import numpy as np
#import os
import importlib

import my_utilities as mu
import my_constants as mc
import E_loss_functions_2020 as elf

import matplotlib.pyplot as plt

from scipy import integrate

#ma = importlib.reload(ma)
mu = importlib.reload(mu)
mc = importlib.reload(mc)
elf = importlib.reload(elf)

#print('E_loss_functions are loaded')


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

#Si_total_Eb  = [1849, 159, 109, 18, 12.67]
#Si_total_occ = [   2,   2,   6,  2,     2]

N_val_Si = 4

n_Si = mc.rho_Si * mc.Na/mc.u_Si


#%% Gryzinski
def get_Gr_diff_cs(Eb, E, hw):
    
    if E < hw:
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
    return get_Gr_tau(MMA_core_Eb[0], E, hw, n_MMA, MMA_core_occ[0])


def get_Gr_PMMA_O_core_tau(E, hw):
    return get_Gr_tau(MMA_core_Eb[1], E, hw, n_MMA, MMA_core_occ[1])


## PMMA u
def get_Gr_PMMA_C_core_u(E):
    return get_Gr_u(MMA_core_Eb[0], E, n_MMA, MMA_core_occ[0])


def get_Gr_PMMA_O_core_u(E):
    return get_Gr_u(MMA_core_Eb[1], E, n_MMA, MMA_core_occ[1])


def get_Gr_PMMA_core_u(E):
    return get_Gr_PMMA_C_core_u(E) + get_Gr_PMMA_O_core_u(E)


## PMMA S
def get_Gr_PMMA_C_core_S(E):
    return get_Gr_S(MMA_core_Eb[0], E, n_MMA, MMA_core_occ[0])


def get_Gr_PMMA_O_core_S(E):
    return get_Gr_S(MMA_core_Eb[1], E, n_MMA, MMA_core_occ[1])


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


## Si u
def get_Gr_Si_1s_u(E):
    return get_Gr_u(Si_core_Eb[0], E, n_Si, Si_core_occ[0])


def get_Gr_Si_2s_u(E):
    return get_Gr_u(Si_core_Eb[1], E, n_Si, Si_core_occ[1])


def get_Gr_Si_2p_u(E):
    return get_Gr_u(Si_core_Eb[2], E, n_Si, Si_core_occ[2])


def get_Gr_Si_core_u(E):
    get_Gr_Si_1s_u(E) + get_Gr_Si_2s_u(E) + get_Gr_Si_2p_u(E)


## Si S
def get_Gr_Si_1s_S(E):
    return get_Gr_S(Si_core_Eb[0], E, n_Si, Si_core_occ[0])


def get_Gr_Si_2s_S(E):
    return get_Gr_S(Si_core_Eb[1], E, n_Si, Si_core_occ[1])


def get_Gr_Si_2p_S(E):
    return get_Gr_S(Si_core_Eb[2], E, n_Si, Si_core_occ[2])


def get_Gr_Si_core_S(E):
    get_Gr_Si_1s_S(E) + get_Gr_Si_2s_S(E) + get_Gr_Si_2p_S(E)


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
##    ans[i] = get_Gr_Si_total_S(E)
#    bns[i] = get_Gr_Si_core_u(E)
#    
#    ans[i] = elf.get_Si_Gryzinski_core_U(np.array([E]))
#
#
##%%
#chan_S = np.loadtxt('Chan_Si_l.txt')
#
#plt.loglog(EE, ans, '.', label='my total')
#plt.loglog(EE, bns, '.', label='my core')
#
##plt.loglog(EE, ans - bns)
#
#plt.loglog(chan_S[:, 0], 1/chan_S[:, 1], '.', label='chan')
#
#plt.legend()

