#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu
# from scipy import integrate

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA'
        ))


#%%
m_sgs     = 9.109383701e-28 ## g
e_sgs     = 4.803204673e-10 ## sm^3/2 * g^1/2* s^-1
eV_sgs    = 1.602176620e-12 ## erg
h_bar_sgs = 1.054571817e-27 ## erg * s



def get_kF_vF(wp): ## SGS
    
    n = wp**2 * m_sgs / (4 * np.pi * e_sgs**2) ## SGS
    
    kF = ( 3 * np.pi**2 * n )**(1/3)
    vF = h_bar_sgs * kF / m_sgs
    
    return kF, vF


def get_eps_L_book_gamma(q, hw_eV, Epl_eV, gamma_eV): ## gamma is energy!
    
    kF, vF = get_kF_vF(Epl_eV * eV_sgs / h_bar_sgs)
    qF = h_bar_sgs * kF
    EF = m_sgs * vF**2 / 2
    
    z = q/ (2 * qF)
    x = (hw_eV + 1j*gamma_eV) * eV_sgs / EF
    
    chi_2 = e_sgs**2 / (np.pi * h_bar_sgs * vF)
    
    
    def f(x, z):
        
        res = 1/2 + 1/(8*z) * (1 - (z - x/(4*z))**2) *\
        np.log( (z - x/(4*z) + 1) / (z - x/(4*z) - 1) ) +\
        1/(8*z) * (1 - (z + x/(4*z))**2) *\
        np.log( (z + x/(4*z) + 1) / (z + x/(4*z) - 1) )
        
        return res
    
    
    return 1 + chi_2/z**2 * f(x, z)


def get_eps_L_book(q, hw_eV, Epl_eV):
    
    kF, vF = get_kF_vF(Epl_eV * eV_sgs / h_bar_sgs)
    
    # print(vF)
    
    qF = h_bar_sgs * kF
    EF = m_sgs * vF**2 / 2
    
    # print(EF)
    
    z = q/ (2 * qF)
    x = hw_eV * eV_sgs / EF
    
    # print(x, z)
    
    chi_2 = e_sgs**2 / (np.pi * h_bar_sgs * vF)
    # chi_2 = (4 / (9 * np.pi**4))**(1/3) * 1.78
    
    print(chi_2)
    
    
    def f1(x, z):
        
        br_1 = 1 - (z - x/(4*z))**2
        br_2 = 1 - (z + x/(4*z))**2
        
        log_1 = np.log( np.abs( (z - x/(4*z) + 1) / (z - x/(4*z) - 1) ) )
        log_2 = np.log( np.abs( (z + x/(4*z) + 1) / (z + x/(4*z) - 1) ) )
        
        res = 1/2 + 1/(8*z)*br_1*log_1 + 1/(8*z)*br_2*log_2
        
        return res
    
    
    def f2(x, z):
        
        u = x / (4 * z)
        
        # if 0 < x and x < 4 * z * (1 - z):
        if z + u < 1:
            return np.pi * x / (8 * z)
    
        # elif 4 * z * (z - 1) < x and x < 4 * z * (z + 1) and x > 4 * z * (1 - z):
        elif np.abs(z - u) < 1 and 1 < z + u:
            return (np.pi / 8) * (1 - (x / (4 * z))**2)
        
        else:
            return 0
    
    
    return 1 + chi_2/z**2 * ( f1(x, z) + 1j*f2(x, z) )


#%%
p_au = 1.9928519141e-24 * 1e+5

q = p_au

k = q / h_bar_sgs

Epl_20 = 20

ww = np.linspace(0, 30, 100)
# ww = np.linspace(3, 7, 2)

eps_L_03 = np.zeros(len(ww), dtype=complex)
eps_L_05 = np.zeros(len(ww), dtype=complex)
eps_L_07 = np.zeros(len(ww), dtype=complex)

eps_L_b_03 = np.zeros(len(ww), dtype=complex)
eps_L_b_05 = np.zeros(len(ww), dtype=complex)
eps_L_b_07 = np.zeros(len(ww), dtype=complex)


for i, hw in enumerate(ww):
    
    # eps_L_b_03[i] = get_eps_L_book(0.3*q, hw, Epl_20)
    # eps_L_b_05[i] = get_eps_L_book(0.5*q, hw, Epl_20)
    # eps_L_b_07[i] = get_eps_L_book(0.7*q, hw, Epl_20)
    
    eps_L_b_03[i] = get_eps_L_book_gamma(0.3*q, hw, Epl_20, 1e-100)
    eps_L_b_05[i] = get_eps_L_book_gamma(0.5*q, hw, Epl_20, 1e-100)
    eps_L_b_07[i] = get_eps_L_book_gamma(0.7*q, hw, Epl_20, 1e-100)


plt.plot(ww, np.imag(eps_L_b_03))
plt.plot(ww, np.imag(eps_L_b_05))
plt.plot(ww, np.imag(eps_L_b_07))


book = np.loadtxt('book_L_im.txt')

plt.plot(book[:, 0], book[:, 1], '.')


# plt.xlim(0, 30)
# plt.ylim(-5, 25)

