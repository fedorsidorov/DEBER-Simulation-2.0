#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu
from scipy import integrate

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
a0_sgs    = 5.292e-9        ## cm

## hwi    hgi       Ai
PMMA_params = [
    [19.13,  9.03, 2.59e-1],
    [25.36, 14.34, 4.46e-1],
    [70.75, 48.98, 4.44e-3]
    ]

PMMA_hw_th = 2.99

p_au = 1.9928519141e-19

inv_A = 1e+8 ## cm^-1


#%%
def get_eps_L(q, hw_eV_complex, Epl_eV): ## gamma is energy!
    
    n = (Epl_eV * eV_sgs / h_bar_sgs)**2 * m_sgs / (4 * np.pi * e_sgs**2) ## SGS
    
    kF = ( 3 * np.pi**2 * n )**(1/3)
    vF = h_bar_sgs * kF / m_sgs
    qF = h_bar_sgs * kF
    EF = m_sgs * vF**2 / 2
    
    z = q/ (2 * qF)
    x = hw_eV_complex * eV_sgs / EF
    
    chi_2 = e_sgs**2 / (np.pi * h_bar_sgs * vF)
    
    
    def f(x, z):
        
        res = 1/2 + 1/(8*z) * (1 - (z - x/(4*z))**2) *\
        np.log( (z - x/(4*z) + 1) / (z - x/(4*z) - 1) ) +\
        1/(8*z) * (1 - (z + x/(4*z))**2) *\
        np.log( (z + x/(4*z) + 1) / (z + x/(4*z) - 1) )
        
        return res
    
    
    return 1 + chi_2/z**2 * f(x, z)


def get_eps_M(q, hw_eV_complex, E_pl_eV): ## hw_complex = hw + 1j*gamma - in eV
    
    hw = np.real(hw_eV_complex)
    gamma = np.imag(hw_eV_complex)
    
    num = (1 + 1j*gamma/hw) *\
        (get_eps_L(q, hw_eV_complex, E_pl_eV) - 1)
        
    den = 1 +\
        (1j*gamma/hw) * (get_eps_L(q, hw_eV_complex, E_pl_eV) - 1) /\
        (get_eps_L(q, 1e-100 + 1e-100j, E_pl_eV) - 1)
    
    return 1 + num/den


#%% 1 test ELF - OK
# HW = np.linspace(1, 80, 100)


# eps_M_0p1 = np.zeros(len(HW), dtype=complex)
# eps_M_1p0 = np.zeros(len(HW), dtype=complex)
# eps_M_10p = np.zeros(len(HW), dtype=complex)


# for i, hw in enumerate(HW):
    
#     eps_M_0p1[i] = get_eps_M(1.0*p_au, hw + 0.1j, 20)
#     eps_M_1p0[i] = get_eps_M(1.0*p_au, hw +   1j, 20)
#     eps_M_10p[i] = get_eps_M(1.0*p_au, hw +  10j, 20)


# plt.plot(HW, np.imag(-1/eps_M_0p1))
# plt.plot(HW, np.imag(-1/eps_M_1p0))
# plt.plot(HW, np.imag(-1/eps_M_10p))

# book = np.loadtxt('curves/mermin_book.txt')
# plt.plot(book[:, 0], book[:, 1], '.')


#%% 2 test ELF - OK
# HW = np.linspace(1, 80, 100)

# eps_L_0p5 = np.zeros(len(HW), dtype=complex)
# eps_L_1p0 = np.zeros(len(HW), dtype=complex)
# eps_L_1p5 = np.zeros(len(HW), dtype=complex)

# eps_M_0p5 = np.zeros(len(HW), dtype=complex)
# eps_M_1p0 = np.zeros(len(HW), dtype=complex)
# eps_M_1p5 = np.zeros(len(HW), dtype=complex)


# for i, hw in enumerate(HW):
    
#     eps_L_0p5[i] = get_eps_L(0.5*p_au, hw + 1e-100j, 20)
#     eps_L_1p0[i] = get_eps_L(1.0*p_au, hw + 1e-100j, 20)
#     eps_L_1p5[i] = get_eps_L(1.5*p_au, hw + 1e-100j, 20)
    
#     eps_M_0p5[i] = get_eps_M(0.5*p_au, hw + 5j, 20)
#     eps_M_1p0[i] = get_eps_M(1.0*p_au, hw + 5j, 20)
#     eps_M_1p5[i] = get_eps_M(1.5*p_au, hw + 5j, 20)


# plt.plot(HW, np.imag(-1/eps_L_0p5))
# plt.plot(HW, np.imag(-1/eps_L_1p0))
# plt.plot(HW, np.imag(-1/eps_L_1p5))

# plt.plot(HW, np.imag(-1/eps_M_0p5))
# plt.plot(HW, np.imag(-1/eps_M_1p0))
# plt.plot(HW, np.imag(-1/eps_M_1p5))

# book_L = np.loadtxt('curves/book_L.txt')
# book_M = np.loadtxt('curves/book_M.txt')

# plt.plot(book_L[:, 0], book_L[:, 1], '.')
# plt.plot(book_M[:, 0], book_M[:, 1], '.')


#%%
def get_PMMA_ELF_L(q, hw_eV, params_hw_hg_A): ## SGS
    
    PMMA_ELF_L = 0
    
    
    for line in params_hw_hg_A:
        
        E_pl_eV, _, A = line
        now_eps_L = get_eps_M(q, hw_eV + 1e-100j, E_pl_eV)

        PMMA_ELF_L += A * np.imag(-1/now_eps_L)
    
    
    return PMMA_ELF_L


def get_PMMA_ELF_M(q, hw_eV, params_hw_hg_A): ## SGS
    
    PMMA_ELF_M = 0
    
    
    for line in params_hw_hg_A:
        
        E_pl_eV, gamma_eV, A = line
        now_eps_M = get_eps_M(q, hw_eV + 1j*gamma_eV, E_pl_eV)

        PMMA_ELF_M += A * np.imag(-1/now_eps_M)
    
    
    return PMMA_ELF_M


def get_PMMA_ELF(q, hw_eV, params_hw_hg_A, kind): ## SGS
    
    PMMA_ELF = 0
    
    
    for line in params_hw_hg_A:
        
        E_pl_eV, gamma_eV, A = line
        
        if  kind == 'L':
            gamma_eV = 1e-100
        
        elif kind == 'M':
            gamma_eV = gamma_eV
        
        else:
            print('Specify ELF kind!')
            return 0 + 0j
        
        now_eps = get_eps_M(q, hw_eV + 1j*gamma_eV, E_pl_eV)

        PMMA_ELF += A * np.imag(-1/now_eps)
    
    
    return PMMA_ELF


def get_PMMA_OLF_D(hw, params_hw_hg_A):
    
    PMMA_OLF_D = 0
    
    
    for line in params_hw_hg_A:
        
        hw_eV, hg_eV, A = line
        PMMA_OLF_D += A * hw_eV**2 * hg_eV * hw / ((hw_eV**2 - (hw)**2)**2 + (hg_eV*hw)**2)
    
    
    return PMMA_OLF_D


#%% test PMMA OLF - OK
# EE = np.linspace(1, 100, 100)

# OLF_M = np.zeros(len(EE))
# OLF_D = np.zeros(len(EE))


# for i, E in enumerate(EE):
    
#     OLF_M[i] = get_PMMA_ELF(5e-3 * inv_A * h_bar_sgs, E, PMMA_params, kind='M')
#     OLF_D[i] = get_PMMA_OLF_D(E, PMMA_params)


# plt.plot(EE, OLF_M, '*')
# plt.plot(EE, OLF_D)

# ritsko = np.loadtxt('Ritsko_Henke/Ritsko_dashed.txt')
# plt.plot(ritsko[:, 0], ritsko[:, 1], '.', label='Ritsko')


#%% test PMMA ELF - OK
# EE = np.linspace(1, 80, 100)

# ELF_2_M = np.zeros(len(EE))
# ELF_4_M = np.zeros(len(EE))


# for i, E in enumerate(EE):
    
#     ELF_2_M[i] = get_PMMA_ELF(2 * inv_A * h_bar_sgs, E, PMMA_params, 'M')
#     ELF_4_M[i] = get_PMMA_ELF(4 * inv_A * h_bar_sgs, E, PMMA_params, 'M')


# plt.plot(EE, ELF_2_M, label='my Mermin 2 A-1')
# plt.plot(EE, ELF_4_M, label='my Mermin 4 A-1')


# DM2 = np.loadtxt('curves/Dapor_M_2A-1.txt')
# DM4 = np.loadtxt('curves/Dapor_M_4A-1.txt')

# plt.plot(DM2[:, 0], DM2[:, 1], 'o')
# plt.plot(DM4[:, 0], DM4[:, 1], 'o')


#%%
def get_PMMA_DIIMFP(E_eV, hw_eV):
    
    if hw_eV > E_eV:
        return 0
    
    E = E_eV * eV_sgs
    hw = hw_eV * eV_sgs
    
    
    def get_Y(k):
        return get_PMMA_ELF(k * h_bar_sgs, hw_eV, PMMA_params, 'M') / k
    
    kp = np.sqrt(2*m_sgs / h_bar_sgs**2) * (np.sqrt(E) + np.sqrt(E - hw))
    km = np.sqrt(2*m_sgs / h_bar_sgs**2) * (np.sqrt(E) - np.sqrt(E - hw))
    
    integral = integrate.quad(get_Y, km, kp)[0]
    
    
    return 1 / (np.pi * a0_sgs * E_eV) * integral ## cm^-1 * eV^-1


#%% test PMMA DIIMFP - OK
# EE_D = [50, 100, 200, 300, 400, 500, 1000]
# EE = np.linspace(1, 80, 100)

# DIIMFP = np.zeros((len(EE_D), len(EE)))

# for i, E_D in enumerate(EE_D):

#     for j, E in enumerate(EE):
    
#         DIIMFP[i, j] = get_PMMA_DIIMFP(E_D, E)


# for i in range(len(EE_D)):
#     plt.plot(EE, DIIMFP[i, :] * 1e-8)


# Dapor_DIIMFP = np.loadtxt('curves/Dapor_DIIMFP.txt')
# plt.plot(Dapor_DIIMFP[:, 0], Dapor_DIIMFP[:, 1], '.')

# plt.xlim(0, 100)
# plt.ylim(0, 0.008)

