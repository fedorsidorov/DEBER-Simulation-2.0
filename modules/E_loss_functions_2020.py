#%% Import
import numpy as np
#import os
import importlib
#import my_arrays as ma
import my_utilities as mu
import my_constants as mc
import matplotlib.pyplot as plt

#ma = importlib.reload(ma)
mu = importlib.reload(mu)
mc = importlib.reload(mc)

#print('E_loss_functions are loaded')


#%% Gryzinski
def get_Gryzinski_diff_CS_single(E, Ui, WW=mc.EE):
    
    Gryzinski_diff_CS_single = np.zeros(len(WW))
    inds = np.where(np.logical_and(WW>=Ui, WW <= (E+Ui)/2))
    DE = WW[inds]
    
    if len(inds) == 0:
        return Gryzinski_diff_CS_single
    
    diff_CS = np.pi * mc.k_el**2 * mc.e**4 / np.power(DE*mc.eV, 3) * Ui/E *\
        np.power(E / (E + Ui), 3/2) * np.power((1 - DE/E), Ui/(Ui+DE)) *\
        (DE/Ui * (1 - Ui/E) + 4/3 * np.log(2.7 + np.sqrt((E - DE)/Ui))) ## m^2 / J
    
    Gryzinski_diff_CS_single[inds] = diff_CS * (100)**2 * mc.eV ## cm^2 / eV
        
    return Gryzinski_diff_CS_single


def get_Gryzinski_diff_CS(EE, Ui, WW=mc.EE):
    
    Gryzinski_diff_CS = np.zeros((len(EE), len(WW)))
    
    for i in range(len(EE)):
        Gryzinski_diff_CS[i, :] = get_Gryzinski_diff_CS_single(EE[i], Ui, WW)
        
    return Gryzinski_diff_CS


def get_Gryzinski_CS(EE, Ui, WW=mc.EE):
    
    Gryzinski_CS = np.zeros(len(EE))
    
    for i in range(len(EE)):
        diff_CS = get_Gryzinski_diff_CS_single(EE[i], Ui, WW)
        Gryzinski_CS[i] = np.trapz(diff_CS, x=WW)
        
    return Gryzinski_CS


def get_Gryzinski_SP_single(E, Ui, conc, n_el, WW=mc.EE):
    
    diff_CS = get_Gryzinski_diff_CS_single(E, Ui, WW)
    Gryzinski_SP_single = conc * n_el * np.trapz(diff_CS * WW, x=WW)
    
    return Gryzinski_SP_single


def get_Gryzinski_SP(EE, Ui, conc, n_el, WW=mc.EE):
    
    Gryzinski_SP = np.zeros(len(EE))
    diff_CS = get_Gryzinski_diff_CS(EE, Ui, WW)
    
    for i in range(len(EE)):
        Gryzinski_SP[i] = conc * n_el * np.trapz(diff_CS[i, :] * WW, x=WW)
    
    return Gryzinski_SP


#%% PMMA
## Differential
def get_PMMA_C_Gryzinski_1S_diff_U(EE, WW):
    return get_Gryzinski_diff_CS(EE, mc.binding_C_1S, WW) * mc.occupancy_1S *\
        mc.n_PMMA_mon * mc.N_C_PMMA


def get_PMMA_O_Gryzinski_1S_diff_U(EE, WW):
    return get_Gryzinski_diff_CS(EE, mc.binding_O_1S, WW) * mc.occupancy_1S *\
        mc.n_PMMA_mon * mc.N_O_PMMA


## Integral
def get_PMMA_C_Gryzinski_1S_int_U(EE):
    return mu.diff2int(get_PMMA_C_Gryzinski_1S_diff_U(EE))


def get_PMMA_O_Gryzinski_1S_int_U(EE):
    return mu.diff2int(get_PMMA_O_Gryzinski_1S_diff_U(EE))
    

## Total
def get_PMMA_C_Gryzinski_1S_U(EE):
    return get_Gryzinski_CS(EE, mc.binding_C_1S) * mc.occupancy_1S * mc.n_PMMA_mon *\
        mc.N_C_PMMA


def get_PMMA_O_Gryzinski_1S_U(EE):
    return get_Gryzinski_CS(EE, mc.binding_O_1S) * mc.occupancy_1S * mc.n_PMMA_mon *\
        mc.N_O_PMMA


## Stopping power
def get_PMMA_C_1S_SP(EE):
    return get_Gryzinski_SP(EE, mc.binding_C_1S, mc.n_PMMA_mon*mc.N_C_PMMA, mc.occupancy_1S)


def get_PMMA_O_1S_SP(EE):
    return get_Gryzinski_SP(EE, mc.binding_O_1S, mc.n_PMMA_mon*mc.N_O_PMMA, mc.occupancy_1S)


#%% Si
## Differential
def get_Si_Gryzinski_core_diff_U(EE):
    
    core_diff_U = np.zeros((len(EE), len(EE)))
    
    for i in range(3):    
        core_diff_U += get_Gryzinski_diff_CS(EE, mc.binding_Si[i]) * mc.occupancy_Si[i] * mc.n_Si
    
    return core_diff_U


## Detailed
def get_Si_Gryzinski_1S_diff_U(EE):
    return get_Gryzinski_diff_CS(EE, mc.binding_Si[0]) * mc.occupancy_Si[0] * mc.n_Si


def get_Si_Gryzinski_2S_diff_U(EE):
    return get_Gryzinski_diff_CS(EE, mc.binding_Si[1]) * mc.occupancy_Si[1] * mc.n_Si


def get_Si_Gryzinski_2P_diff_U(EE):
    return get_Gryzinski_diff_CS(EE, mc.binding_Si[2]) * mc.occupancy_Si[2] * mc.n_Si


## Integral
def get_Si_Gryzinski_1S_int_U(EE):
    return mu.diff2int(get_Si_Gryzinski_1S_diff_U(EE))


def get_Si_Gryzinski_2S_int_U(EE):
    return mu.diff2int(get_Si_Gryzinski_2S_diff_U(EE))


def get_Si_Gryzinski_2P_int_U(EE):
    return mu.diff2int(get_Si_Gryzinski_2P_diff_U(EE))


## Total
def get_Si_Gryzinski_1S_U(EE):
    return get_Gryzinski_CS(EE, mc.binding_Si[0]) * mc.occupancy_Si[0] * mc.n_Si


def get_Si_Gryzinski_2S_U(EE):
    return get_Gryzinski_CS(EE, mc.binding_Si[1]) * mc.occupancy_Si[1] * mc.n_Si


def get_Si_Gryzinski_2P_U(EE):
    return get_Gryzinski_CS(EE, mc.binding_Si[2]) * mc.occupancy_Si[2] * mc.n_Si


#%%
def get_PMMA_Gryzinski_core_U(EE):    
    return get_PMMA_C_Gryzinski_1S_U(EE) + get_PMMA_O_Gryzinski_1S_U(EE)


def get_PMMA_Gryzinski_core_SP(EE):
    return get_PMMA_C_1S_SP(EE) + get_PMMA_O_1S_SP(EE)
    

#%% Si
def get_Si_Gryzinski_core_U(EE):
    
    Si_core_U = np.zeros(len(EE))
    
    for i in range(3):
        Si_core_U += get_Gryzinski_CS(EE, mc.binding_Si[i]) * mc.occupancy_Si[i] * mc.n_Si
    
    return Si_core_U


def get_Si_Gryzinski_core_SP(EE):

    Si_core_SP = np.zeros(len(EE))
    
    for i in range(3):        
        Si_core_SP += get_Gryzinski_SP(EE, mc.binding_Si[i], mc.n_Si, mc.occupancy_Si[i])
    
    return Si_core_SP


#%% Bethe
def get_Bethe_SP(E, Z, rho, A, J):
    
    K = 0.734 * Z**0.037
    dEds = 785 * rho*Z / (A*E) * np.log(1.166 * (E + K*J) / J) * 1e+8 ## eV/cm
    
    return dEds


def get_PMMA_Bethe_SP(E):
    
    Z_PMMA = mc.Z_PMMA
    rho_PMMA = mc.rho_PMMA
    A_PMMA = mc.u_PMMA / 15
    J_PMMA = 65.6
    
    return get_Bethe_SP(E, Z_PMMA, rho_PMMA, A_PMMA, J_PMMA)


def get_Si_Bethe_SP(E):
    
    Z_Si = mc.Z_Si
    rho_Si = mc.rho_Si
    A_Si = mc.u_Si
    J_Si = 173
    
    return get_Bethe_SP(E, Z_Si, rho_Si, A_Si, J_Si)


#%%
def Ashley_L(x):
    f = (1-x)*np.log(4/x) - 7/4*x + x**(3/2) - 33/32*x**2
    return f

def Ashley_S(x):
    f = np.log(1.166/x) - 3/4*x - x/4*np.log(4/x) + 1/2*x**(3/2) - x**2/16*np.log(4/x) -\
        31/48*x**2
    return f


#%% Greeneich E_loss, z in cm
def get_e_density(E0, Q, z):
    
    q = 1.6e-19
    rho = 1.19
    
    RG = 4.6e-6 / rho * (E0 / 1e+3)**1.75
    f = z / RG
    lambda_f = 0.74 + 4.7*f - 8.9*f**2 + 3.5*f**3
    
    eps = Q/q * E0/RG * lambda_f
    
    return eps ## eV/cc


#%% Calculate Harris E_loss per 1 cm^2 for 500nm PMMA layer, 10 keV, 100 uC/cm^2
#z = np.linspace(0, 500e-7, 1000)
#dE = get_e_density(10e+3, 1e-4, z)

#plt.plot(z, dE)

#E_loss_cm2 = np.trapz(dE, x=z)





