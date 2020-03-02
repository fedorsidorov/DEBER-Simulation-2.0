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
EE_eV = np.logspace(0, 4.4, 100)
#EE_eV = np.logspace(-1, 4.4, 2000)
#EE_eV = np.linspace(0.01, 1e+4, 1000)

EE = EE_eV * mc.eV
qq = np.sqrt(2*mc.m*EE)

a0 = 5.29e-11 ## m

h2si = mc.k_el * mc.m * mc.e**2 / mc.hbar**2

popt = np.load('popt_Akkerman.npy')

params = popt.reshape((6, 3))



#%%
def get_oscillator(E_eV, A, E, w, q_eV):
    
    if E > 50 and E_eV < E:
        return 0
    
    Eq = E + q_eV**2 / (2*mc.m)
    
    return A*w*E_eV / ((E_eV**2 - Eq**2)**2 + (w*E_eV)**2)


def get_ELF(E_eV, q_eV):
    
    ELF = 0
    
    for arr in params:
        A, E, w = arr
        ELF += get_oscillator(E_eV, A, E, w, q_eV)
    
    
    return ELF


#%%
OLF_test = np.zeros(len(EE_eV))

for i in range(len(EE_eV)):
    OLF_test[i] = get_ELF(EE_eV[i], 0)


plt.loglog(EE_eV, OLF_test)  

OLF_file = np.loadtxt('curves/OLF_Akkerman_fit.txt')

EE_file = OLF_file[:, 0]
OLF_file = OLF_file[:, 1]

plt.loglog(EE_file, OLF_file)



#%%
def get_tau(E_eV, hw_eV):
    
    def get_ELF_q(q_eV):
        return get_ELF(hw_eV, q_eV) / q_eV
    
    qp = np.sqrt(2*mc.m)*(np.sqrt(E_eV) + np.sqrt(E_eV - hw_eV))
    qm = np.sqrt(2*mc.m)*(np.sqrt(E_eV) - np.sqrt(E_eV - hw_eV))
    
    return h2si * 1/(np.pi * E_eV) * integrate.quad(get_ELF_q, qm, qp)[0]


def get_S(E_eV):
    
    def get_tau_hw_S(hw_eV):
        return get_tau(E_eV, hw_eV) * hw_eV
    
    return integrate.quad(get_tau_hw_S, 0, E_eV/2)[0]


def get_u(E_eV):
    
    def get_tau_u(hw_eV):
        return get_tau(E_eV, hw_eV)
    
    return integrate.quad(get_tau_u, 0, E_eV/2)[0]


#%%
S = np.zeros(len(EE_eV))
u = np.zeros(len(EE_eV))


for i, E_eV in enumerate(EE_eV):
    
    mu.pbar(i, len(EE_eV))
    
    S[i] = get_S(E_eV)
    u[i] = get_u(E_eV)


#%%
plt.semilogx(EE_eV, S / 1e+2, label='my')

S_Chan = np.loadtxt('curves/Chan_Si_S.txt')
#plt.loglog(S_Chan[:, 0], S_Chan[:, 1], label='Chan')

S_MuElec = np.loadtxt('curves/Si_MuElec_S.txt')
plt.loglog(S_MuElec[:, 0], S_MuElec[:, 1] * 1e+7, '--', label='MuElec')

#plt.xlim(1, 1e+4)
#plt.ylim(1e+5, 1e+9)

plt.legend()
plt.grid()

#plt.savefig('Si_valentin2012_quad_S.png', dpi=300)


#%%
plt.semilogx(EE_eV, u / 1e+2, label='no exchange') ## IS BETTER
#plt.semilogx(EE_eV, u_exc / 1e+2, label='exchange')

l_Chan = np.loadtxt('curves/Chan_Si_l.txt')
#plt.loglog(l_Chan[:, 0], 1 / l_Chan[:, 1], label='Chan')

sigma_MuElec = np.loadtxt('curves/Si_MuElec_sigma.txt')
plt.loglog(sigma_MuElec[:, 0], sigma_MuElec[:, 1] * 1e-18 * mc.n_Si, '--', label='MuElec')

plt.xlim(1, 1e+4)
plt.ylim(1e+5, 1e+8)

plt.legend()
plt.grid()

#plt.savefig('Si_valentin2012_quad_u.png', dpi=300)

