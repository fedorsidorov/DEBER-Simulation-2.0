#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import my_constants as mc
import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mc = importlib.reload(mc)

#os.chdir(mv.sim_path_MAC + 'E_loss')

#%% Moller
def get_Moller_diff_CS(E, W):
    
    eps_c = 0.01
#    eps_c = 0.001
    
    Moller_diff_CS = np.zeros(len(W))
    
    inds = np.where(np.logical_and(W >= eps_c*E, W <= E/2))
    eps = W[inds] / E
    
    diff_CS = np.pi * mc.k_el**2 * mc.e**4 / np.power(E*mc.eV, 3) *\
        (1/eps**2 + 1/(1-eps)**2 - 1/(eps*(1-eps))) ## m^2 / J
    
    Moller_diff_CS[inds] = diff_CS * (100)**2 * mc.eV  ## cm^2 / eV
    
    return Moller_diff_CS


def get_Moller_CS(E):
    
    Moller_CS = np.zeros(len(E))
    DE = np.logspace(0, 4.4, 1000)
    
    for i in range(len(E)):
        
        diff_cs = get_Moller_diff_CS(E[i], DE)
        Moller_CS[i] = np.trapz(diff_cs, x=DE)
        
    return Moller_CS


def get_Moller_SP(E, conc, n_el):
    
    W = np.logspace(0, 4.4, 1000)
    
    Moller_SP = np.zeros(len(E))
    
    for i in range(len(E)):
        
        Moller_SP[i] = conc * n_el * np.trapz(get_Moller_diff_CS(E[i], W) * W, x=W)
    
    return Moller_SP


#%% Gryzinski
def get_Gryzinski_diff_CS(E, Ui, W):
    
    Gryzinski_diff_CS = np.zeros(len(W))
    
    inds = np.where(np.logical_and(W>=Ui, W <= (E+Ui)/2))
    
    dE = W[inds]
    
    diff_cs = np.pi * mc.k_el**2 * mc.e**4 / np.power(dE*mc.eV, 3) * Ui/E *\
        np.power(E / (E + Ui), 3/2) * np.power((1 - dE/E), Ui/(Ui+dE)) *\
        (dE/Ui * (1 - Ui/E) + 4/3 * np.log(2.7 + np.sqrt((E - dE)/Ui))) ## m^2 / J
    
    Gryzinski_diff_CS[inds] = diff_cs * (100)**2 * mc.eV ## cm^2 / eV
    
    return Gryzinski_diff_CS


def get_Gryzinsky_CS(E, Ui):
    
    Gryzinsky_CS = np.zeros(len(E))
    
    DE = np.logspace(0, 4.4, 1000)
    
    for i in range(len(E)):
        
        diff_CS = get_Gryzinski_diff_CS(E[i], Ui, DE)
        Gryzinsky_CS[i] = np.trapz(diff_CS, x=DE)
        
    return Gryzinsky_CS


def get_Gryzinski_SP(E, Ui, conc, n_el):
    
    W = np.logspace(-2, 4.4, 10000)
    
    Gryzinski_SP = np.zeros(len(E))
    
    for i in range(len(E)):
        
        Gryzinski_SP[i] = conc * n_el * np.trapz(get_Gryzinski_diff_CS(E[i], Ui, W) * W, x=W)
    
    return Gryzinski_SP


#%% Vriens
def get_Vriens_diff_CS(E, Ui, W):
    
    R = 13.6 ## eV
    Rn = R / E
    Uin = Ui / E
    
    inds = np.where(np.logical_and(W>=Ui, W <= (E+Ui)/2))
    eps = W[inds] / E
    
    PHI = np.cos( -np.sqrt((Rn/(1 + Uin))) * np.log(Uin) )
    
    diff_cs = np.pi * mc.k_el**2 * mc.e**4 / ((E*mc.eV)**3*(1 + 2*Uin)) *\
        ( (1/eps**2 + 4*Uin/(3*eps**3)) + (1/(1 + Uin - eps)**2 +\
           4*Uin/(3*(1 + Uin - eps)**2)) - PHI / (eps*(1 + Uin - eps))) ## m^2 / J
    
    vriens_diff_CS = np.zeros(len(W))
    vriens_diff_CS[inds] = diff_cs * (100)**2 * mc.eV ## cm^2 / eV
    
    return vriens_diff_CS


def get_Vriens_CS(E_arr, Ui):
    
    R = 13.6 ## eV
    Rn = R / E
    Uin = Ui / E
    
    PHI = np.cos( -np.sqrt((Rn/(1 + Uin))) * np.log(Uin) )
    
    total_Vriens_cs = np.pi * mc.k_el**2 * mc.e**4 / ((E_arr*mc.eV)**2 * (1 + 2*Uin)) *\
        (5/(3*Uin) - 1 - 2/3*Uin + PHI*np.log(Uin)/(1 + Uin))  * (100)**2 ## cm^2
    
    return total_Vriens_cs


#%% PMMA and Si
##              1s 2s 2p 3s 3p
binding_H    = [13.6]
occupancy_H  = [1]
binding_C    = [296, 16.59, 11.26]
occupancy_C  = [2, 2, 2]
binding_O    = [538, 28.48, 13.62]
occupancy_O  = [2, 2, 4]
binding_Si   = [1844, 154, 104, 13.46, 8.15]
occupancy_Si = [2, 2, 6, 2, 2]

n_val_PMMA = 40
n_val_Si = 4
Ui_PMMA = 10
Ui_Si = 4


## PMMA
def get_PMMA_Gryzinski_core_U(E):
    
    CS_C_1S = get_Gryzinsky_CS(E, binding_C[0]) * occupancy_C[0]
    U_C_K = CS_C_1S * mc.n_PMMA*5
    
    CS_O_1S = get_Gryzinsky_CS(E, binding_O[0]) * occupancy_O[0]
    U_O_K = CS_O_1S * mc.n_PMMA*2
    
    return U_C_K + U_O_K
    

def get_PMMA_Gryzinski_valence_U(E):
    
    CS_PMMA_val = get_Gryzinsky_CS(E, Ui_PMMA) * n_val_PMMA
    U_PMMA_val = CS_PMMA_val * mc.n_PMMA
    
    return U_PMMA_val


def get_PMMA_Moller_valence_U(E):
    
    CS_PMMA_val = get_Moller_CS(E) * n_val_PMMA
    U_PMMA_val = CS_PMMA_val * mc.n_PMMA
    
    return U_PMMA_val


def get_PMMA_Gryzinski_core_SP(E):
    
    SP_C_1S = get_Gryzinski_SP(E, binding_C[0], mc.n_PMMA*5, occupancy_C[0])
    SP_O_1S = get_Gryzinski_SP(E, binding_O[0], mc.n_PMMA*2, occupancy_O[0])
    
    return SP_C_1S + SP_O_1S
    

def get_PMMA_Gryzinski_valence_SP(E):
    
    SP_PMMA_val = get_Gryzinski_SP(E, Ui_PMMA, mc.n_PMMA, n_val_PMMA)
    
    return SP_PMMA_val
    

def get_PMMA_Moller_valence_SP(E):
    
    SP_PMMA_val = get_Moller_SP(E, mc.n_PMMA, n_val_PMMA)

    return SP_PMMA_val
    

## Si
def get_Si_Gryzinski_core_U(E):
    
    CS_Si_1S = get_Gryzinsky_CS(E, binding_Si[0]) * occupancy_Si[0]
    U_Si_K = CS_Si_1S * mc.n_Si
    
    CS_Si_2S = get_Gryzinsky_CS(E, binding_Si[1]) * occupancy_Si[1]
    CS_Si_2P = get_Gryzinsky_CS(E, binding_Si[2]) * occupancy_Si[2]
    U_Si_L = (CS_Si_2S + CS_Si_2P) * mc.n_Si
    
    return U_Si_K + U_Si_L
    

def get_Si_Gryzinski_valence_U(E):
    
    CS_Si_val = get_Gryzinsky_CS(E, Ui_Si) * n_val_Si
    U_Si_val = CS_Si_val * mc.n_Si
    
    return U_Si_val


def get_Si_Moller_valence_U(E):
    
    CS_Si_val = get_Moller_CS(E) * n_val_Si
    U_Si_val = CS_Si_val * mc.n_Si
    
    return U_Si_val


def get_Si_Gryzinski_core_SP(E):
    
    SP_Si_1S = get_Gryzinski_SP(E, binding_Si[0], mc.n_Si, occupancy_Si[0])
    SP_Si_2S = get_Gryzinski_SP(E, binding_Si[1], mc.n_Si, occupancy_Si[1])
    SP_Si_3S = get_Gryzinski_SP(E, binding_Si[2], mc.n_Si, occupancy_Si[2])
    
    return SP_Si_1S + SP_Si_2S + SP_Si_3S
    

def get_Si_Gryzinski_valence_SP(E):
    
    SP_Si_val = get_Gryzinski_SP(E, Ui_Si, mc.n_Si, n_val_Si)
    
    return SP_Si_val


def get_Si_Moller_valence_SP(E):
    
    SP_Si_val = get_Moller_SP(E, mc.n_Si, n_val_Si)
    
    return SP_Si_val


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


#%% PMMA
#E = np.logspace(1, 4.4, 1000)
#
#SP_PMMA_GG = get_PMMA_Gryzinski_core_SP(E) + get_PMMA_Gryzinski_valence_SP(E)
#SP_PMMA_GM = get_PMMA_Gryzinski_core_SP(E) + get_PMMA_Moller_valence_SP(E)
#
#SP_PMMA_B = get_PMMA_Bethe_SP(E)
#
#plt.loglog(E, SP_PMMA_GG, label='Gryzinski + Gryzinski')
#plt.loglog(E, SP_PMMA_GM, label='Gryzinski + Moller')
#plt.loglog(E, SP_PMMA_B, label='Bethe')
#
#plt.title('Stopping Power for PMMA')
#plt.xlabel('E, eV')
#plt.ylabel('SP, eV/cm')
#
#plt.legend()
#plt.grid()
#plt.show()
#plt.savefig('SP_PMMA.png', dpi=300)

#%% Si
#E = np.logspace(1, 4.4, 1000)
#
#SP_Si_GG = get_Si_Gryzinski_core_SP(E) + get_Si_Gryzinski_valence_SP(E)
#SP_Si_GM = get_Si_Gryzinski_core_SP(E) + get_Si_Moller_valence_SP(E)
#
#SP_Si_B = get_Si_Bethe_SP(E)
#
#plt.loglog(E, SP_Si_GG, label='Gryzinski + Gryzinski')
#plt.loglog(E, SP_Si_GM, label='Gryzinski + Moller')
#plt.loglog(E, SP_Si_B, label='Bethe')
#
#plt.title('Stopping Power for Si')
#plt.xlabel('E, eV')
#plt.ylabel('SP, eV/cm')
#
#plt.legend()
#plt.grid()
#plt.show()
#plt.savefig('SP_Si.png', dpi=300)
