#%% Import
import numpy as np
#import os
import importlib
import my_functions as mf
import my_variables as mv
import my_constants as mc
#import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mc = importlib.reload(mc)

#print('E_loss_functions are loaded')


#%% Gryzinski
def get_Gryzinski_diff_CS_single(E, Ui, WW=mv.EE):
    
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


def get_Gryzinski_diff_CS(EE, Ui, WW=mv.EE):
    
    Gryzinski_diff_CS = np.zeros((len(EE), len(WW)))
    
    for i in range(len(EE)):
        
        Gryzinski_diff_CS[i, :] = get_Gryzinski_diff_CS_single(EE[i], Ui, WW)
        
    return Gryzinski_diff_CS


def get_Gryzinski_CS(EE, Ui, WW=mv.EE):
    
    Gryzinski_CS = np.zeros(len(EE))
    
    for i in range(len(EE)):
        
        diff_CS = get_Gryzinski_diff_CS_single(EE[i], Ui, WW)
        Gryzinski_CS[i] = np.trapz(diff_CS, x=WW)
        
    return Gryzinski_CS


def get_Gryzinski_SP_single(E, Ui, conc, n_el, WW=mv.EE):
        
    diff_CS = get_Gryzinski_diff_CS_single(E, Ui, WW)
    
    Gryzinski_SP_single = conc * n_el * np.trapz(diff_CS * WW, x=WW)
    
    return Gryzinski_SP_single


def get_Gryzinski_SP(EE, Ui, conc, n_el, WW=mv.EE):
    
    Gryzinski_SP = np.zeros(len(EE))
        
    diff_CS = get_Gryzinski_diff_CS(EE, Ui, WW)
    
    for i in range(len(EE)):
        
        Gryzinski_SP[i] = conc * n_el * np.trapz(diff_CS[i, :] * WW, x=WW)
    
    return Gryzinski_SP


#%% PMMA and Si
binding_C_1S = 296
binding_O_1S = 538

occupancy_1S = 2

##              1s    2s   2p   3s     3p
binding_Si   = [1844, 154, 104, 13.46, 8.15]
occupancy_Si = [2,    2,   6,   2,     2]

n_val_PMMA = 40
n_val_Si = 4
#Ui_PMMA = 10
#Ui_Si = 4


#%% Integrals of spectra
def diff2int(DIFF, EE=mv.EE):
    
    INT = np.zeros((len(EE), len(EE)))

    for i in range(len(EE)):

        integral = np.trapz(DIFF[i, :], x=EE)
        
        if integral == 0:
            continue
        
        for j in range(1, len(EE)):
            INT[i, j] = np.trapz(DIFF[i, :j], x=EE[:j]) / integral
    
    return INT


#%% PMMA
## Differential
def get_PMMA_C_Gryzinski_1S_diff_U(EE):
    return get_Gryzinski_diff_CS(EE, binding_C_1S) * occupancy_1S * mc.n_PMMA_mon * mc.n_C_PMMA


def get_PMMA_O_Gryzinski_1S_diff_U(EE):
    return get_Gryzinski_diff_CS(EE, binding_O_1S) * occupancy_1S * mc.n_PMMA_mon * mc.n_O_PMMA


## Integral
def get_PMMA_C_Gryzinski_1S_int_U(EE):
    return diff2int(get_PMMA_C_Gryzinski_1S_diff_U(EE))


def get_PMMA_O_Gryzinski_1S_int_U(EE):
    return diff2int(get_PMMA_O_Gryzinski_1S_diff_U(EE))
    

## Total
def get_PMMA_C_Gryzinski_1S_U(EE):
    return get_Gryzinski_CS(EE, binding_C_1S) * occupancy_1S * mc.n_PMMA_mon * mc.n_C_PMMA


def get_PMMA_O_Gryzinski_1S_U(EE):
    return get_Gryzinski_CS(EE, binding_O_1S) * occupancy_1S * mc.n_PMMA_mon * mc.n_O_PMMA


#%% Si
## Differential
def get_Si_Gryzinski_1S_diff_U(EE):
    return get_Gryzinski_diff_CS(EE, binding_Si[0]) * occupancy_Si[0] * mc.n_Si


def get_Si_Gryzinski_2S_diff_U(EE):
    return get_Gryzinski_diff_CS(EE, binding_Si[1]) * occupancy_Si[1] * mc.n_Si


def get_Si_Gryzinski_2P_diff_U(EE):
    return get_Gryzinski_diff_CS(EE, binding_Si[2]) * occupancy_Si[2] * mc.n_Si


def get_Si_Gryzinski_3S_diff_U(EE):
    return get_Gryzinski_diff_CS(EE, binding_Si[3]) * occupancy_Si[3] * mc.n_Si


def get_Si_Gryzinski_3P_diff_U(EE):
    return get_Gryzinski_diff_CS(EE, binding_Si[4]) * occupancy_Si[4] * mc.n_Si


#def get_Si_Gryzinski_VAL_diff_U(EE):
#    return get_Gryzinski_diff_CS(EE, Ui_Si) * n_val_Si * mc.n_Si


## Integral
def get_Si_Gryzinski_1S_int_U(EE):
    return diff2int(get_Si_Gryzinski_1S_diff_U(EE))


def get_Si_Gryzinski_2S_int_U(EE):
    return diff2int(get_Si_Gryzinski_2S_diff_U(EE))


def get_Si_Gryzinski_2P_int_U(EE):
    return diff2int(get_Si_Gryzinski_2P_diff_U(EE))


def get_Si_Gryzinski_3S_int_U(EE):
    return diff2int(get_Si_Gryzinski_3S_diff_U(EE))


def get_Si_Gryzinski_3P_int_U(EE):
    return diff2int(get_Si_Gryzinski_3P_diff_U(EE))


#def get_Si_Gryzinski_VAL_int_U(EE):
#    return diff2int(get_Si_Gryzinski_VAL_diff_U(EE))


## Total
def get_Si_Gryzinski_1S_U(EE):
    return get_Gryzinski_CS(EE, binding_Si[0]) * occupancy_Si[0] * mc.n_Si


def get_Si_Gryzinski_2S_U(EE):
    return get_Gryzinski_CS(EE, binding_Si[1]) * occupancy_Si[1] * mc.n_Si


def get_Si_Gryzinski_2P_U(EE):
    return get_Gryzinski_CS(EE, binding_Si[2]) * occupancy_Si[2] * mc.n_Si


def get_Si_Gryzinski_3S_U(EE):
    return get_Gryzinski_CS(EE, binding_Si[3]) * occupancy_Si[3] * mc.n_Si


def get_Si_Gryzinski_3P_U(EE):
    return get_Gryzinski_CS(EE, binding_Si[4]) * occupancy_Si[4] * mc.n_Si


#def get_Si_Gryzinski_VAL_U(EE):
#    return get_Gryzinski_CS(EE, Ui_Si) * n_val_Si * mc.n_Si


#%% Additions
#%% Additions
#%% Additions
#%% Moller
def get_Moller_diff_CS(EE, W):
    
    eps_c = 0.01
#    eps_c = 0.001
    
    Moller_diff_CS = np.zeros(len(W))
    
    inds = np.where(np.logical_and(W >= eps_c*EE, W <= EE/2))
    eps = W[inds] / EE
    
    diff_CS = np.pi * mc.k_el**2 * mc.e**4 / np.power(EE*mc.eV, 3) *\
        (1/eps**2 + 1/(1-eps)**2 - 1/(eps*(1-eps))) ## m^2 / J
    
    Moller_diff_CS[inds] = diff_CS * (100)**2 * mc.eV  ## cm^2 / eV
    
    return Moller_diff_CS


def get_Moller_CS(EE):
    
    Moller_CS = np.zeros(len(EE))
    DE = np.logspace(0, 4.4, 1000)
    
    for i in range(len(EE)):
        
        diff_cs = get_Moller_diff_CS(EE[i], DE)
        Moller_CS[i] = np.trapz(diff_cs, x=DE)
        
    return Moller_CS


def get_Moller_SP(EE, conc, n_el):
    
    W = np.logspace(0, 4.4, 1000)
    
    Moller_SP = np.zeros(len(EE))
    
    for i in range(len(EE)):
        
        Moller_SP[i] = conc * n_el * np.trapz(get_Moller_diff_CS(EE[i], W) * W, x=W)
    
    return Moller_SP


#%% Vriens
def get_Vriens_diff_CS(EE, Ui, W):
    
    R = 13.6 ## eV
    Rn = R / EE
    Uin = Ui / EE
    
    inds = np.where(np.logical_and(W>=Ui, W <= (EE+Ui)/2))
    eps = W[inds] / EE
    
    PHI = np.cos( -np.sqrt((Rn/(1 + Uin))) * np.log(Uin) )
    
    diff_cs = np.pi * mc.k_el**2 * mc.e**4 / ((EE*mc.eV)**3*(1 + 2*Uin)) *\
        ( (1/eps**2 + 4*Uin/(3*eps**3)) + (1/(1 + Uin - eps)**2 +\
           4*Uin/(3*(1 + Uin - eps)**2)) - PHI / (eps*(1 + Uin - eps))) ## m^2 / J
    
    vriens_diff_CS = np.zeros(len(W))
    vriens_diff_CS[inds] = diff_cs * (100)**2 * mc.eV ## cm^2 / eV
    
    return vriens_diff_CS


def get_Vriens_CS(EE, Ui):
    
    R = 13.6 ## eV
    Rn = R / EE
    Uin = Ui / EE
    
    PHI = np.cos( -np.sqrt((Rn/(1 + Uin))) * np.log(Uin) )
    
    total_Vriens_cs = np.pi * mc.k_el**2 * mc.e**4 / ((EE*mc.eV)**2 * (1 + 2*Uin)) *\
        (5/(3*Uin) - 1 - 2/3*Uin + PHI*np.log(Uin)/(1 + Uin))  * (100)**2 ## cm^2
    
    return total_Vriens_cs


def get_PMMA_Gryzinski_core_U(EE):
    
    CS_C_1S = get_Gryzinski_CS(EE, binding_C_1S) * occupancy_1S
    U_C_K = CS_C_1S * mc.n_PMMA_mon * mc.n_C_PMMA
    
    CS_O_1S = get_Gryzinski_CS(EE, binding_O_1S) * occupancy_1S
    U_O_K = CS_O_1S * mc.n_PMMA_mon * mc.n_O_PMMA
    
    return U_C_K + U_O_K
    

#def get_PMMA_Gryzinski_valence_U(EE):
#    CS_PMMA_val = get_Gryzinski_CS(EE, Ui_PMMA) * n_val_PMMA
#    U_PMMA_val = CS_PMMA_val * mc.n_PMMA_mon
#    return U_PMMA_val


def get_PMMA_Moller_valence_U(EE):
    
    CS_PMMA_val = get_Moller_CS(EE) * n_val_PMMA
    U_PMMA_val = CS_PMMA_val * mc.n_PMMA_mon
    
    return U_PMMA_val


def get_PMMA_Gryzinski_core_SP(EE):
    
    SP_C_1S = get_Gryzinski_SP(EE, binding_C_1S, mc.n_PMMA_mon*mc.n_C_PMMA, occupancy_1S)
    SP_O_1S = get_Gryzinski_SP(EE, binding_O_1S, mc.n_PMMA_mon*mc.n_O_PMMA, occupancy_1S)
    
    return SP_C_1S + SP_O_1S
    

#def get_PMMA_Gryzinski_valence_SP(EE):
#    SP_PMMA_val = get_Gryzinski_SP(EE, Ui_PMMA, mc.n_PMMA_mon, n_val_PMMA)
#    return SP_PMMA_val
    

def get_PMMA_Moller_valence_SP(EE):
    
    SP_PMMA_val = get_Moller_SP(EE, mc.n_PMMA_mon, n_val_PMMA)

    return SP_PMMA_val
    

#%% Si
def get_Si_Gryzinski_core_U(EE):
    
    CS_Si_1S = get_Gryzinski_CS(EE, binding_Si[0]) * occupancy_Si[0]
    U_Si_K = CS_Si_1S * mc.n_Si
    
    CS_Si_2S = get_Gryzinski_CS(EE, binding_Si[1]) * occupancy_Si[1]
    CS_Si_2P = get_Gryzinski_CS(EE, binding_Si[2]) * occupancy_Si[2]
    U_Si_L = (CS_Si_2S + CS_Si_2P) * mc.n_Si
    
    return U_Si_K + U_Si_L
    

#def get_Si_Gryzinski_valence_U(EE):
#    CS_Si_val = get_Gryzinski_CS(EE, Ui_Si) * n_val_Si
#    U_Si_val = CS_Si_val * mc.n_Si
#    return U_Si_val


def get_Si_Moller_valence_U(EE):
    
    CS_Si_val = get_Moller_CS(EE) * n_val_Si
    U_Si_val = CS_Si_val * mc.n_Si
    
    return U_Si_val


def get_Si_Gryzinski_core_SP(EE):
    
    SP_Si_1S = get_Gryzinski_SP(EE, binding_Si[0], mc.n_Si, occupancy_Si[0])
    SP_Si_2S = get_Gryzinski_SP(EE, binding_Si[1], mc.n_Si, occupancy_Si[1])
    SP_Si_2P = get_Gryzinski_SP(EE, binding_Si[2], mc.n_Si, occupancy_Si[2])
    
    return SP_Si_1S + SP_Si_2S + SP_Si_2P


def get_Si_Gryzinski_total_SP(EE):
    
    SP_Si_1S = get_Gryzinski_SP(EE, binding_Si[0], mc.n_Si, occupancy_Si[0])
    SP_Si_2S = get_Gryzinski_SP(EE, binding_Si[1], mc.n_Si, occupancy_Si[1])
    SP_Si_2P = get_Gryzinski_SP(EE, binding_Si[2], mc.n_Si, occupancy_Si[2])
    SP_Si_3S = get_Gryzinski_SP(EE, binding_Si[3], mc.n_Si, occupancy_Si[3])
    SP_Si_3P = get_Gryzinski_SP(EE, binding_Si[2], mc.n_Si, occupancy_Si[2])
    
    return SP_Si_1S + SP_Si_2S + SP_Si_2P + SP_Si_3S + SP_Si_3P


#def get_Si_Gryzinski_valence_SP(EE):
#    return get_Gryzinski_SP(EE, Ui_Si, mc.n_Si, n_val_Si)


#def get_Si_Moller_valence_SP(EE):
#    return get_Moller_SP(EE, mc.n_Si, n_val_Si)

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
    f = np.log(1.166/x) - 3/4*x - x/4*np.log(4/x) + 1/2*x**(3/2) - x**2/16*np.log(4/x) - 31/48*x**2
    return f


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
