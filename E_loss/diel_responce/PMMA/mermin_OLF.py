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

## hwi    hgi       Ai
PMMA_params = [
    [19.13,  9.03, 2.59e-1],
    [25.36, 14.34, 4.46e-1],
    [70.75, 48.98, 4.44e-3]
    ]

PMMA_hw_th = 2.99


def get_eps_L(k, hw_eV_complex, Epl_eV): ## gamma is energy!
    
    n = (Epl_eV * eV_sgs / h_bar_sgs)**2 * m_sgs / (4 * np.pi * e_sgs**2) ## SGS
    
    kF = ( 3 * np.pi**2 * n )**(1/3)
    vF = h_bar_sgs * kF / m_sgs
    EF = m_sgs * vF**2 / 2
    
    z = k / (2 * kF)
    x = hw_eV_complex * eV_sgs / EF
    
    chi_2 = e_sgs**2 / (np.pi * h_bar_sgs * vF)
    
    
    def f(x, z):
                
        log_arg_1 = (z - x/(4*z) + 1) / (z - x/(4*z) - 1)
        log_arg_2 = (z + x/(4*z) + 1) / (z + x/(4*z) - 1)
        
        res = 1/2 + 1/(8*z) * (1 - (z - x/(4*z))**2) *\
        np.log( log_arg_1 ) +\
        1/(8*z) * (1 - (z + x/(4*z))**2) *\
        np.log( log_arg_2 )
        
        return res
    
    
    return 1 + chi_2/z**2 * f(x, z)


#%%
def get_eps_M(k, hw_eV_complex, E_pl_eV): ## hw_complex = hw + 1j*gamma - in eV
    
    hw = np.real(hw_eV_complex)
    gamma = np.imag(hw_eV_complex)
    
    num = (1 + 1j*gamma/hw) *\
        (get_eps_L(k, hw_eV_complex, E_pl_eV) - 1)
        
    den = 1 +\
        (1j*gamma/hw) * (get_eps_L(k, hw_eV_complex, E_pl_eV) - 1) /\
        (get_eps_L(k, 0 + 0j, E_pl_eV) - 1)
    
    return 1 + num/den


def get_Im(k, hw_eV, params_hw_hg_A): ## SGS
    
    Im_L = 0
    Im_M = 0
    
    
    for line in params_hw_hg_A:
        
        E_pl_eV, gamma_eV, A = line
        
        now_eps_L = get_eps_L(k, hw_eV + 1e-100j, E_pl_eV)
        now_eps_M = get_eps_M(k, hw_eV + 1j*gamma_eV, E_pl_eV)

        now_Im_L = np.imag(-1/now_eps_L)
        now_Im_M = np.imag(-1/now_eps_M)
        
        Im_L += A * now_Im_L
        Im_M += A * now_Im_M
    
    
    return Im_L, Im_M


def drude_Im(hw, params_hw_hg_A):
    
    Im = 0
    
    
    for line in params_hw_hg_A:
        
        hw_eV, hg_eV, A = line
        
        Im += A * hw_eV**2 * hg_eV * hw / ( (hw_eV**2 - (hw)**2)**2 + (hg_eV*hw)**2 )
    
    
    return Im


#%% test OLF, ELF
# ritsko = np.loadtxt('Ritsko_Henke/Ritsko_dashed.txt')
# dapor_2 = np.loadtxt('curves/Dapor_2A-1.txt')
# dapor_2L = np.loadtxt('curves/Dapor_lind_2A-1.txt')

book_L = np.loadtxt('curves/book_L.txt')
# book_M = np.loadtxt('curves/book_M.txt')

plt.plot(book_L[:, 0], book_L[:, 1], 'o', label='book Lindhard')
# plt.plot(book_M[:, 0], book_M[:, 1], 'o', label='book Mermin')

p_au = 1.9928519141e-19

# EE = np.linspace(1, 100, 100)
EE = np.linspace(1, 80, 80)

# Im_2_M = np.zeros(len(EE))
# Im_2_L = np.zeros(len(EE))

ELF_0p5_L = np.zeros(len(EE), dtype=complex)
ELF_1p0_L = np.zeros(len(EE), dtype=complex)
ELF_1p5_L = np.zeros(len(EE), dtype=complex)

ELF_0p5_M = np.zeros(len(EE), dtype=complex)
ELF_1p0_M = np.zeros(len(EE), dtype=complex)
ELF_1p5_M = np.zeros(len(EE), dtype=complex)

inv_A = 1e+8 ## cm^-1


for i, E in enumerate(EE):
    
    # Im_drude[i] = drude_Im(E, PMMA_params)
    # Im[i] = get_Im(4e-3*inv_A, E, PMMA_params)
    # Im_2_L[i], Im_2_M[i] = get_Im(2*inv_A, E, PMMA_params)
    
    ELF_0p5_L[i] = get_eps_L(0.5*p_au/h_bar_sgs, E + 0j, 20)
    ELF_1p0_L[i] = get_eps_L(1.0*p_au/h_bar_sgs, E + 0j, 20)
    ELF_1p5_L[i] = get_eps_L(1.5*p_au/h_bar_sgs, E + 0j, 20)
    
    # ELF_0p5_M[i] = get_eps_M(0.5*p_au/h_bar_sgs, E + 0j, 20)
    # ELF_1p0_M[i] = get_eps_M(1.0*p_au/h_bar_sgs, E + 0j, 20)
    # ELF_1p5_M[i] = get_eps_M(1.5*p_au/h_bar_sgs, E + 0j, 20)


plt.plot(EE, np.imag(-1/ELF_0p5_L))
plt.plot(EE, np.imag(-1/ELF_1p0_L))
plt.plot(EE, np.imag(-1/ELF_1p5_L))

# plt.plot(EE, np.imag(-1/ELF_0p5_M))
# plt.plot(EE, np.imag(-1/ELF_1p0_M))
# plt.plot(EE, np.imag(-1/ELF_1p5_M))

# plt.plot(ritsko[:, 0], ritsko[:, 1], 'o', label='Ritsko')
# plt.plot(dapor_2[:, 0], dapor_2[:, 1], 'o', label='Dapor 2 inv A')
# plt.plot(dapor_2L[:, 0], dapor_2L[:, 1], 'o', label='Dapor Lind 2 inv A')


# plt.plot(EE, Im_drude, label='Drude')

# plt.plot(EE, Im, '*')
# plt.plot(EE, Im_2_M, '*', label='my M')
# plt.plot(EE, Im_2_L, '*', label='my L')
# plt.plot(EE, Im_2, '*', label='my 2 inv A')

plt.grid()
plt.legend()

plt.xlim(0, 80)
# plt.ylim(0, 1.2)


#%% test ELF - OK
p_au = 1.9928519141e-24 * 1e+5

Epl_20 = 20

HW = np.linspace(1, 80, 100)


eps_M_p1 = np.zeros(len(HW), dtype=complex)
eps_M_01 = np.zeros(len(HW), dtype=complex)
eps_M_10 = np.zeros(len(HW), dtype=complex)


for i, hw in enumerate(HW):
    
    eps_M_p1[i] = get_eps_M(1.0*p_au, hw + 0.1j, Epl_20)
    eps_M_01[i] = get_eps_M(1.0*p_au, hw +   1j, Epl_20)
    eps_M_10[i] = get_eps_M(1.0*p_au, hw +  10j, Epl_20)


plt.plot(HW, np.imag(-1/eps_M_p1))
plt.plot(HW, np.imag(-1/eps_M_01))
plt.plot(HW, np.imag(-1/eps_M_10))

book = np.loadtxt('curves/mermin_book.txt')
plt.plot(book[:, 0], book[:, 1], '.')


#%%
# k = 2e+8 ## 2 A^-1 - cm^-1
# p_au = 1.9928519141e-24 * 1e+5

# xx = np.linspace(1, 30, 100)
# yy = np.zeros(len(xx))


# for i in range(len(xx)):
    
#     yy[i] = np.imag(1/get_eps_L_gamma(0.3*p_au, xx[i], 20, 0))


# plt.plot(xx, yy)


# lindhard = np.loadtxt('Lindharh_ELF_book.txt')

# plt.plot(lindhard[:, 0], lindhard[:, 1], '.')




#%%
# EE = np.linspace(1, 100, 100)

# Im = np.zeros(len(EE))


# for i, E in enumerate(EE):
    
#     Im[i] = get_Im(h_bar_sgs*k, E, PMMA_params, hw_threshold_eV=0)


# plt.plot(EE, Im)


#%%
# def get_DIIMFP(E_eV, hw_eV):
        
#     E = E_eV * mc.eV
#     hw = hw_eV * mc.eV
    
#     w = hw / mc.h_bar
    
#     if hw > E:
#         return 0
    
#     def get_Y(k):
#         # return get_Im(k, w, PMMA_params, PMMA_hw_th) / k
#         return get_Im(k, w, PMMA_params, 0) / k
    
#     kp = np.sqrt( 2*mc.m / mc.h_bar**2 ) * ( np.sqrt(E) + np.sqrt(E - hw) )
#     km = np.sqrt( 2*mc.m / mc.h_bar**2 ) * ( np.sqrt(E) - np.sqrt(E - hw) )
    
#     integral = integrate.quad(get_Y, km, kp)[0]
    
    
#     return 1 / (np.pi * mc.a0 * E_eV) * integral ## cm^-1 * eV^-1


#%%
# DIIMFP = np.zeros(len(mc.EE))


# for i, E in enumerate(mc.EE):
    
#     DIIMFP[i] = get_DIIMFP(200, mc.EE[i])


#%%
# plt.plot(mc.EE, DIIMFP * 1e-8)

# plt.xlim(0, 100)




