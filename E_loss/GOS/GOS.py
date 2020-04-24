import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import importlib
import os

import my_constants as mc
mc = importlib.reload(mc)

os.chdir(os.path.join(mc.sim_folder, 'E_loss', 'GOS'))


#%%
m_CGS     = 9.109383701e-28 ## g
e_CGS     = 4.803204673e-10 ## sm^3/2 * g^1/2* s^-1
eV_CGS    = 1.602176620e-12 ## erg
h_bar_CGS = 1.054571817e-27 ## erg * s
a0_CGS    = 5.292e-9        ## cm
Ry_CGS   = 13.605693 * eV_CGS


def get_df_dw_1s(k, w, Zs):
    
    n = 1 ## 1s
    
    Q_0 = h_bar_CGS**2 * k**2 / (2 * m_CGS)
    W_0 = h_bar_CGS * w
    
    Q = Q_0 / (Zs**2 * Ry_CGS)
    W = W_0 / (Zs**2 * Ry_CGS)
    
    k2 = W - 1/n**2
    
    
    def A(Q, W):
        
        if k2 > 0:
            
            kappa = np.sqrt(k2)
            
            fC = (1 - np.exp(-2*np.pi/kappa))**(-1)
            
            return 2**5 * (2/n)**3 * W * np.exp(
                -2/kappa * np.arctan(
                    2*kappa/n / (Q - W + 2/n**2)
                    )
                ) * fC
        
        else:
            
            return 2**5 * (2/n)**3 * W * np.exp(
                -1/np.sqrt(-k2) * np.log(
                    (Q - W + 2/n**2 + 2*np.sqrt(-k2)/n) / (Q - W + 2/n**2 - 2*np.sqrt(-k2)/n)
                    )
                )
    
    
    def B(Q, W):
        
        return ((Q - W)**2 + (2/n)**2 * Q)**(-2*(n+1))
    
    
    def C(Q, W):
        
        return Q + W/3
    
    
    df_dW = A(Q, W) * B(Q, W) * C(Q, W)
    
    df_dW_0 = 1 / (Zs**2 * Ry_CGS) * df_dW
    
    
    return df_dW_0


def get_PMMA_ELG_GOS(q, hw_eV):
    
    w = hw_eV * eV_CGS / h_bar_CGS
    k = q / h_bar_CGS
    
    C_alpha = mc.N_C_MMA
    O_alpha = mc.N_O_MMA
    
    C_Zs = 5.7
    O_Zs = 7.7
    
    PMMA_ELF_GOS = 2 * np.pi**2 * mc.n_MMA / w *\
        C_alpha * get_df_dw_1s(k, w, C_Zs) *\
        O_alpha * get_df_dw_1s(k, w, O_Zs)
    
    
    return PMMA_ELF_GOS


def get_PMMA_DIIMFP_GOS(E_eV, hw_eV, exchange=False):
    
    if hw_eV > E_eV:
        return 0
    
    E = E_eV * eV_CGS
    hw = hw_eV * eV_CGS
    
    
    def get_Y(k):
        return get_PMMA_ELG_GOS(k * h_bar_CGS, hw_eV) / k
    
    
    kp = np.sqrt(2*m_CGS / h_bar_CGS**2) * (np.sqrt(E) + np.sqrt(E - hw))
    km = np.sqrt(2*m_CGS / h_bar_CGS**2) * (np.sqrt(E) - np.sqrt(E - hw))
    
    integral = integrate.quad(get_Y, km, kp)[0]
    
    
    return 1 / (np.pi * a0_CGS * E_eV) * integral ## cm^-1 * eV^-1
    
    
#%%



