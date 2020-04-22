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
PMMA_params = [
    ## hwi    hgi       Ai
    [19.13,  9.03, 2.59e-1],
    [25.36, 14.34, 4.46e-1],
    [70.75, 48.98, 4.44e-3]
    ]

PMMA_hw_th = 2.99

m_sgs     = 9.109383701e-28
e_sgs     = 4.803204673e-10
eV_sgs    = 1.602176620e-12
h_bar_sgs = 1.054571817e-27



def get_kF_vF(wp): ## SGS
    
    n = wp**2 * m_sgs / (4 * np.pi * e_sgs**2) ## SGS
    
    kF = ( 3 * np.pi * n )**(1/3)
    vF = h_bar_sgs * kF / m_sgs
    
    return kF, vF


## DEPRECATED
def get_eps_L(k, w, wp, gamma): ## SGS gamma is energy!!!
    
    
    def g(x):
    
        return (1 - x**2) * np.log(np.abs((x + 1)/(x  -1)))


    def f1(u, z):
    
        return 1/2 + 1/(8*z) * (g(z-u) + g(z+u))


    def f2(u, z):
    
        if z + u < 1:
            return np.pi/2 * u
    
        elif np.abs(z - u) < 1 and 1 < z + u:
            return np.pi/(8*z) * (1 - (z - u)**2)
        
        elif np.abs(z - u) > 1:
            return 0
        
        else:
            print('f2 WTF')
    
    
    def f(u, z): ## u is complex !!!
    
        f = 1/2 + 1/(8*z) * (
            (1 - (z - u)**2) *\
            # np.log(np.abs((z - x/(4*z) + 1)/(z - x/(4*z) - 1))) +\
            np.log((z - u + 1)/(z - u - 1)) +\
            (1 - (z + u)**2) *\
            # np.log(np.abs((z + x/(4*z) + 1)/(z + x/(4*z) - 1)))
            np.log((z + u + 1)/(z + u - 1))
            )
        
        return f
    
    
    kF, vF = get_kF_vF(wp)
    EF = m_sgs * vF**2 / 2
    wF = EF / h_bar_sgs
    
    z = k/(2 * kF)
    
    # chi_2 = e_sgs**2 / (np.pi * h_bar_sgs * vF)
    # mult = chi_2 / z**2
    
    mult = 3 *  wF**2 / (k**2 * vF**2)
    
    
    if gamma > 0:
        
        # u = (h_bar_sgs*w + 1j*h_bar_sgs*gamma) / EF
        u = (w + 1j*gamma / h_bar_sgs) / (k * vF)
        
        # print(u, z)
        
        return 1 + mult * f(u, z)
    
    else:
        
        u = w / (k * vF)
        
        # print(u, z)
        
        return 1 + mult * (f1(u, z) + 1j*f2(u, z))


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
    
    # chi_2 = e_sgs**2 / (np.pi * h_bar_sgs * vF)
    chi_2 = (4 / (9 * np.pi**4))**(1/3) * 1.78
    
    
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
    
    eps_L_b_03[i] = get_eps_L_book(0.3*q, hw, Epl_20)
    
    # eps_L_b_03[i] = get_eps_L_book_gamma(0.3*q, hw, Epl_20, 1e-100)
    # eps_L_b_05[i] = get_eps_L_book_gamma(0.5*q, hw, Epl_20, 1e-100)
    # eps_L_b_07[i] = get_eps_L_book_gamma(0.7*q, hw, Epl_20, 1e-100)
    
    eps_L_b_03[i] = get_eps_L_book(0.3*q, hw, Epl_20)
    eps_L_b_05[i] = get_eps_L_book(0.5*q, hw, Epl_20)
    eps_L_b_07[i] = get_eps_L_book(0.7*q, hw, Epl_20)


plt.plot(ww, np.imag(eps_L_b_03))
plt.plot(ww, np.imag(eps_L_b_05))
plt.plot(ww, np.imag(eps_L_b_07))


book = np.loadtxt('book_L_im.txt')

plt.plot(book[:, 0], book[:, 1], '.')


# plt.xlim(0, 30)
# plt.ylim(-5, 25)


#%%
def get_eps_M(k, w, wp, gamma):
    
    num = (1 + 1j*gamma/w) * (get_eps_L(k, w + 1j*gamma, wp, gamma) - 1)
    den = 1 + (1j * gamma/w) * (get_eps_L(k, w + 1j*gamma, wp, gamma) - 1) /\
        (get_eps_L(k, 0, wp, gamma) - 1)
    
    eps_M = 1 + num/den
    
    # print('Im[eps_M] =', np.real(eps_M))
    # print('Im[eps_M] =', np.imag(eps_M))
    
    return eps_M


def get_Im(k, w, params_hw_hg_A, hw_threshold=0): ## SI !!!
    
    Im = 0
    
    
    for line in params_hw_hg_A:
        
        hw_eV, hg_eV, A = line
        
        wp = hw_eV*mc.eV / mc.h_bar
        gamma = hg_eV*mc.eV / mc.h_bar
        
        now_eps_M = get_eps_M(k, w, wp, gamma)

        now_Im = np.imag( -1 / now_eps_M )
        
        
        if hw_threshold > 0:
            now_Im *= np.heaviside(w - hw_threshold * mc.eV / mc.h_bar, 1)
        
        
        Im += A * now_Im
    
    
    return Im


def drude_Im(hw, params_hw_hg_A):
    
    Im = 0
    
    
    for line in params_hw_hg_A:
        
        hw_eV, hg_eV, A = line
        
        Im += A * hw_eV**2 * hg_eV * hw / ( (hw_eV**2 - (hw)**2)**2 + (hg_eV*hw)**2 )
    
    
    return Im
        

#%%
# k = 2e+10 ## 2 A^-1
k = 1e+9
# w = 11*mc.eV / mc.h_bar

xx = np.linspace(1, 100, 100)
yy = np.zeros(len(xx))


for i in range(len(xx)):
    
    yy[i] = get_Im(k, xx[i]*mc.eV / mc.h_bar, PMMA_params, hw_threshold=0)


plt.plot(xx, yy)

#%%
EE = np.linspace(1, 100, 100)

Im = np.zeros(len(EE))


for i in range(len(EE)):
    
    w = EE[i]*mc.eV / mc.h_bar
    
    Im[i] = get_Im(2e+10, w, PMMA_params, hw_threshold=0)


plt.plot(EE, Im)


#%%
def get_DIIMFP(E_eV, hw_eV):
        
    E = E_eV * mc.eV
    hw = hw_eV * mc.eV
    
    w = hw / mc.h_bar
    
    if hw > E:
        return 0
    
    def get_Y(k):
        # return get_Im(k, w, PMMA_params, PMMA_hw_th) / k
        return get_Im(k, w, PMMA_params, 0) / k
    
    kp = np.sqrt( 2*mc.m / mc.h_bar**2 ) * ( np.sqrt(E) + np.sqrt(E - hw) )
    km = np.sqrt( 2*mc.m / mc.h_bar**2 ) * ( np.sqrt(E) - np.sqrt(E - hw) )
    
    integral = integrate.quad(get_Y, km, kp)[0]
    
    
    return 1 / (np.pi * mc.a0 * E_eV) * integral ## cm^-1 * eV^-1


#%%
DIIMFP = np.zeros(len(mc.EE))


for i, E in enumerate(mc.EE):
    
    DIIMFP[i] = get_DIIMFP(200, mc.EE[i])


#%%
plt.plot(mc.EE, DIIMFP * 1e-8)

plt.xlim(0, 100)


#%% test OLF
ritsko = np.loadtxt('Ritsko_Henke/Ritsko_dashed.txt')

EE = np.linspace(1, 100, 100)

Im = np.zeros(len(EE))


for i, E in enumerate(EE):
    
    Im[i] += drude_Im(EE[i], PMMA_params)


plt.plot(EE, Im)
plt.plot(ritsko[:, 0], ritsko[:, 1])

