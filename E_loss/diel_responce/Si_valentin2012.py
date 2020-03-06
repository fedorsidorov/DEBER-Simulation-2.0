#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu
#from itertools import product
from scipy import integrate

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)

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


def get_tau(E, hw):
    
    if hw > E:
        return 0
    
    def get_ELF_q(q):
        return get_ELF(hw, q) / q
    
    qp = np.sqrt(2*mc.m)*(np.sqrt(E) + np.sqrt(E - hw))
    qm = np.sqrt(2*mc.m)*(np.sqrt(E) - np.sqrt(E - hw))
    
    return mc.h2si * 1/(np.pi * E) * integrate.quad(get_ELF_q, qm, qp)[0] / 1e+2 ## m -> cm


def get_S(E):
    
    def get_tau_hw_S(hw):
        return get_tau(E, hw) * hw
    
    return integrate.quad(get_tau_hw_S, 0, E/2)[0]


def get_u(E):
    
    def get_tau_u(hw):
        return get_tau(E, hw)
    
    return integrate.quad(get_tau_u, 0, E/2)[0]


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
#EE = mc.EE
#
#S = np.zeros(len(EE))
#u = np.zeros(len(EE))
#
#
#for i, E in enumerate(EE):
#    
#    mu.pbar(i, len(EE))
#    
#    S[i] = get_S(E)
#    u[i] = get_u(E)


#%%
#plt.semilogx(EE, S, label='my')
#
#S_Chan = np.loadtxt('curves/Chan_Si_S.txt')
##plt.loglog(S_Chan[:, 0], S_Chan[:, 1], label='Chan')
#
#S_MuElec = np.loadtxt('curves/Si_MuElec_S.txt')
#plt.loglog(S_MuElec[:, 0], S_MuElec[:, 1] * 1e+7, '--', label='MuElec')
#
##plt.xlim(1, 1e+4)
##plt.ylim(1e+5, 1e+9)
#
#plt.legend()
#plt.grid()

#plt.savefig('Si_valentin2012_quad_S.png', dpi=300)


#%%
#plt.semilogx(EE, u, label='no exchange') ## IS BETTER
##plt.semilogx(EE_eV, u_exc / 1e+2, label='exchange')
#
#l_Chan = np.loadtxt('curves/Chan_Si_l.txt')
##plt.loglog(l_Chan[:, 0], 1 / l_Chan[:, 1], label='Chan')
#
#sigma_MuElec = np.loadtxt('curves/Si_MuElec_sigma.txt')
#plt.loglog(sigma_MuElec[:, 0], sigma_MuElec[:, 1] * 1e-18 * mc.n_Si, '--', label='MuElec')
#
#plt.xlim(1, 1e+4)
#plt.ylim(1e+5, 1e+8)
#
#plt.legend()
#plt.grid()

#plt.savefig('Si_valentin2012_quad_u.png', dpi=300)

