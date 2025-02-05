#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu
from scipy import integrate

#from itertools import product

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce'
        ))


#%%
params = [
         [100.0, 19.46, 8.770],
         [286.5, 25.84, 14.75],
         [ 80.0, 300.0, 140.0],
         [ 55.0, 550.0, 300.0],
         ]


#%%
def get_oscillator(E, An, En, wn, q):
    Eq = En + q**2 / (2*mc.m)
    return An*wn*E / ((E**2 - Eq**2)**2 + (wn*E)**2)


def get_ELF(E, q):
    
    ELF = 0
    
    for arr in params:
        An, En, wn = arr
        ELF += get_oscillator(E, An, En, wn, q)
    
    return ELF


def get_tau(E, hw):
    
    if hw > E:
        return 0
    
    def get_Y(q):
        return get_ELF(hw, q) / q
    
    qp = np.sqrt(2*mc.m)*(np.sqrt(E) + np.sqrt(E - hw))
    qm = np.sqrt(2*mc.m)*(np.sqrt(E) - np.sqrt(E - hw))
    
    return mc.h2si * 1/(np.pi*E) * integrate.quad(get_Y, qm, qp)[0] / 1e+2 ## m -> cm


def get_S(E):
    
    def get_Y(hw):
        return get_tau(E, hw) * hw
    
    return integrate.quad(get_Y, 0, E/2)[0]


def get_u(E):
    
    def get_Y(hw):
        return get_tau(E, hw)
    
    return integrate.quad(get_Y, 0, E/2)[0]


#%%
#EE = np.logspace(-1, 4.4, 1000)
EE = mc.EE

tau = np.zeros((len(EE), len(EE)))


for i, E in enumerate(EE):
    
    mu.pbar(i, len(EE))
    
    for j, hw in enumerate(EE):        
        tau[i, j] = get_tau(E, hw)



#%%
#tau_cumulated = mu.diff2int(tau, V=mc.EE, H=mc.EE)

tau = np.load('PMMA/tau.npy')

tau_norm = np.zeros(np.shape(tau))


for i in range(len(mc.EE)):
    
    now_tau = tau[i, :]
    
    if np.all(now_tau == 0):
        continue
    
    tau_norm[i, :] = now_tau / np.sum(now_tau)


#%%
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
u = np.load('PMMA/easy/u.npy')

plt.semilogx(mc.EE, 1/u * 1e+8, label='my')

l_Dapor = np.loadtxt('curves/l_dapor2015.txt')
plt.semilogx(l_Dapor[:, 0], l_Dapor[:, 1], label='dapor2015.pdf')

plt.xlim(20, 1.1e+4)
plt.ylim(0, 250)

plt.legend()
plt.grid()


#%%
S = np.load('PMMA/easy/S.npy')

plt.semilogx(mc.EE, S / 1e+8, label='my')

S_Dapor = np.loadtxt('curves/S_dapor2015.txt')
plt.semilogx(S_Dapor[:, 0], S_Dapor[:, 1], label='dapor2015.pdf')

S_Ciappa = np.loadtxt('curves/dEds_solid.txt')
plt.semilogx(S_Ciappa[:, 0], S_Ciappa[:, 1], label='ciappa2010.pdf')

plt.xlim(1, 1e+4)
plt.ylim(0, 4)

plt.legend()
plt.grid()

#plt.savefig('S 0.2, 4.4, 1000.png', dpi=300)

