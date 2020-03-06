#%% Import
import numpy as np
import os
import importlib
from scipy import integrate

import my_constants as mc
import my_utilities as mu
import Gryzinski as gryz

#from itertools import product


import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)
gryz = importlib.reload(gryz)


os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce'
        ))


#%%
popt = np.load('popt_Akkerman.npy')
params = popt.reshape((6, 3))


#%%
def get_oscillator(E, An, En, wn, q):
    
    if En > 50 and E < En:
        return 0
    
    Enq = En + q**2 / (2*mc.m)
    
    return An*wn*E / ((E**2 - Enq**2)**2 + (wn*E)**2)


def get_ELF(E, q):
    
    ELF = 0
    
    for arr in params:
        An, En, wn = arr
        ELF += get_oscillator(E, An, En, wn, q)
    
    return ELF


def get_ELF_1(E, q):
    An, En, wn = params[0, :]
    return get_oscillator(E, An, En, wn, q)


def get_ELF_2(E, q):
    An, En, wn = params[1, :]
    return get_oscillator(E, An, En, wn, q)


def get_ELF_3(E, q):
    An, En, wn = params[2, :]
    return get_oscillator(E, An, En, wn, q)


def get_ELF_4(E, q):
    An, En, wn = params[3, :]
    if E < En:
        return 0
    return get_oscillator(E, An, En, wn, q)


def get_ELF_5(E, q):
    An, En, wn = params[4, :]
    if E < En:
        return 0
    return get_oscillator(E, An, En, wn, q)


def get_ELF_6(E, q):
    An, En, wn = params[5, :]
    if E < En:
        return 0
    return get_oscillator(E, An, En, wn, q)


def get_tau(E, hw):
    
    if hw > E:
        return 0
    
    def get_Y(q):
        return get_ELF(hw, q) / q
    
    qp = np.sqrt(2*mc.m)*(np.sqrt(E) + np.sqrt(E - hw))
    qm = np.sqrt(2*mc.m)*(np.sqrt(E) - np.sqrt(E - hw))
    
    return mc.h2si * 1/(np.pi * E) * integrate.quad(get_Y, qm, qp)[0] / 1e+2 ## m -> cm


def get_tau_ELF(E, hw, get_ELF_func):
    
    if hw > E:
        return 0
    
    def get_Y(q):
        return get_ELF_func(hw, q) / q
    
    qp = np.sqrt(2*mc.m)*(np.sqrt(E) + np.sqrt(E - hw))
    qm = np.sqrt(2*mc.m)*(np.sqrt(E) - np.sqrt(E - hw))
    
    return mc.h2si * 1/(np.pi * E) * integrate.quad(get_Y, qm, qp)[0] / 1e+2 ## m -> cm


def get_u(E):
    def get_Y(hw):
        return get_tau(E, hw)
    return integrate.quad(get_Y, 0, E/2)[0]


def get_S(E):
    def get_Y(hw):
        return get_tau(E, hw) * hw
    return integrate.quad(get_Y, 0, E/2)[0]


def get_u1(E):
    def get_Y(hw):
        return get_tau_ELF(E, hw, get_ELF_1)
#    return integrate.quad(get_Y, params[0, 1], (E+params[0, 1])/2)[0]
    return integrate.quad(get_Y, 0, (E)/2)[0]


def get_u2(E):
    def get_Y(hw):
        return get_tau_ELF(E, hw, get_ELF_2)
#    return integrate.quad(get_Y, params[1, 1], (E+params[1, 1])/2)[0]
    return integrate.quad(get_Y, 8.15, (E+8.15)/2)[0]


def get_u3(E):
    def get_Y(hw):
        return get_tau_ELF(E, hw, get_ELF_3)
#    return integrate.quad(get_Y, params[2, 1], (E+params[2, 1])/2)[0]
    return integrate.quad(get_Y, 13.46, (E+13.46)/2)[0]


def get_u4(E):
    def get_Y(hw):
        return get_tau_ELF(E, hw, get_ELF_4)
#    return integrate.quad(get_Y, params[3, 1], (E+params[3, 1])/2)[0]
    return integrate.quad(get_Y, 104, (E+104)/2)[0]


def get_u5(E):
    def get_Y(hw):
        return get_tau_ELF(E, hw, get_ELF_5)
#    return integrate.quad(get_Y, params[4, 1], (E+params[4, 1])/2)[0]
    return integrate.quad(get_Y, 154, (E+154)/2)[0]


def get_u6(E):
    def get_Y(hw):
        return get_tau_ELF(E, hw, get_ELF_6)
#    return integrate.quad(get_Y, params[5, 1], (E+params[5, 1])/2)[0]
    return integrate.quad(get_Y, 1844, (E+1844)/2)[0]


#%%
#EE = mc.EE
#
#tau = np.zeros((len(EE), len(EE)))
#
#
#for i, E in enumerate(EE):
#    
#    mu.pbar(i, len(EE))
#    
#    for j, hw in enumerate(EE):
#        
#        tau[i, j] = get_tau(E, hw)


#%%
EE = np.logspace(1, 4.4, 100)


u1 = np.zeros(len(EE))
u2 = np.zeros(len(EE))
u3 = np.zeros(len(EE))
u4 = np.zeros(len(EE))
u5 = np.zeros(len(EE))
u6 = np.zeros(len(EE))


for i, E in enumerate(EE):
    
    mu.pbar(i, len(EE))
    
    u1[i] = get_u1(E)
    u2[i] = get_u2(E)
    u3[i] = get_u3(E)
    u4[i] = get_u4(E)
    u5[i] = get_u5(E)
    u6[i] = get_u6(E)


#%%
u1g = np.zeros(len(EE))
u2g = np.zeros(len(EE))
u3g = np.zeros(len(EE))
u4g = np.zeros(len(EE))
u5g = np.zeros(len(EE))
u6g = np.zeros(len(EE))


for i, E in enumerate(EE):
    
    mu.pbar(i, len(EE))
    
    u1g[i] = gryz.get_Gr_u(16.7, E, gryz.n_Si, 4)
    u2g[i] = gryz.get_Gr_u(8.15, E, gryz.n_Si, gryz.Si_total_occ[4])
    u3g[i] = gryz.get_Gr_u(13.46, E, gryz.n_Si, gryz.Si_total_occ[3])
    u4g[i] = gryz.get_Gr_u(104, E, gryz.n_Si, gryz.Si_total_occ[2])
    u5g[i] = gryz.get_Gr_u(154, E, gryz.n_Si, gryz.Si_total_occ[1])
    u6g[i] = gryz.get_Gr_u(1844, E, gryz.n_Si, gryz.Si_total_occ[0])
    
#    u1g[i] = gryz.get_Gr_u(params[0, 1], E, gryz.n_Si, 4)
#    u2g[i] = gryz.get_Gr_u(gryz.Si_total_Eb[4], E, gryz.n_Si, gryz.Si_total_occ[4])
#    u3g[i] = gryz.get_Gr_u(gryz.Si_total_Eb[3], E, gryz.n_Si, gryz.Si_total_occ[3])
#    u4g[i] = gryz.get_Gr_u(gryz.Si_total_Eb[2], E, gryz.n_Si, gryz.Si_total_occ[2])
#    u5g[i] = gryz.get_Gr_u(gryz.Si_total_Eb[1], E, gryz.n_Si, gryz.Si_total_occ[1])
#    u6g[i] = gryz.get_Gr_u(gryz.Si_total_Eb[0], E, gryz.n_Si, gryz.Si_total_occ[0])


#%%
#u = np.zeros(len(EE))
u = u1 + u2 + u3 + u4 + u5 + u6

#for i, E in enumerate(EE):
    
#    mu.pbar(i, len(EE))    
#    u[i] = get_u(E)


#%%
plt.loglog(EE, u, label='total')
plt.loglog(EE, u1, label='1')
plt.loglog(EE, u2, label='2')
plt.loglog(EE, u3, label='3')
plt.loglog(EE, u4, label='4')
plt.loglog(EE, u5, label='5')
plt.loglog(EE, u6, label='6')

plt.loglog(EE, u1g, '--', label='1g')
plt.loglog(EE, u2g, '--', label='2g')
plt.loglog(EE, u3g, '--', label='3g')
plt.loglog(EE, u4g, '--', label='4g')
plt.loglog(EE, u5g, '--', label='5g')
plt.loglog(EE, u6g, '--', label='6g')

#plt.xlim(1, 1e+4)
#plt.ylim(1e+5, 1e+9)

plt.legend()
plt.grid()


#%%
plt.loglog(EE, u, label='total')
plt.loglog(EE, u1+u2+u3, label='M')
plt.loglog(EE, u4+u5, label='L')
plt.loglog(EE, u6, label='K')

plt.loglog(EE, u1g+u2g+u3g, '--', label='Mg')
plt.loglog(EE, u4g+u5g, '--', label='Lg')
plt.loglog(EE, u6g, '--', label='Kg')

plt.xlim(1e+1, 1e+4)
plt.ylim(1e+1, 1e+8)

plt.xlabel('E, eV')
plt.ylabel('u, cm$^{-1}$')

plt.legend()
plt.grid()

#plt.savefig('Si_valentin2012_quad_S.png', dpi=300)


#%%
u = np.load('Si/u.npy')

plt.loglog(mc.EE, u, label='my')

sigma_MuElec = np.loadtxt('curves/Si_MuElec_sigma.txt')
plt.loglog(sigma_MuElec[:, 0], sigma_MuElec[:, 1] * 1e-18 * mc.n_Si, '--', label='MuElec')

plt.xlabel('E, eV')
plt.ylabel('u, cm$^{-1}$')

plt.xlim(1e+1, 1e+4)
plt.ylim(1e+5, 1e+8)

plt.legend()
plt.grid()

#plt.savefig('Si_valentin2012_quad_u.png', dpi=300)


#%%
S = np.load('Si/S.npy')

plt.loglog(mc.EE, S, label='mu') ## IS BETTER
#plt.semilogx(EE_eV, S_exc / mc.eV / 1e+2, label='exchange')

S_Chan = np.loadtxt('curves/Chan_Si_S.txt')
#plt.loglog(S_Chan[:, 0], S_Chan[:, 1], label='Chan')

S_MuElec = np.loadtxt('curves/Si_MuElec_S.txt')
plt.loglog(S_MuElec[:, 0], S_MuElec[:, 1] * 1e+7, label='MuElec')

plt.xlim(1e+1, 1e+4)
plt.ylim(1e+6, 1e+9)

plt.legend()
plt.grid()

#plt.savefig('Si_valentin2012_quad_S.png', dpi=300)

