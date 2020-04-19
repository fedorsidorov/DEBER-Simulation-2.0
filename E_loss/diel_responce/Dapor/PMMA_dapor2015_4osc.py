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
params = np.array((
         (286.5, 25.84, 14.75), ## plasmon first!!!
         (100.0, 19.46, 8.770),
         ( 80.0, 300.0, 140.0),
         ( 55.0, 550.0, 300.0),
         ))

Ebind = (3.6, 25.84, 288, 543)


#%%
def get_ELF(E, q, n_osc):
    An, En, wn = params[n_osc, :]
    Eq = En + q**2 / (2*mc.m)
    return An*wn*E / ((E**2 - Eq**2)**2 + (wn*E)**2)


def get_tau(E, hw, n_osc):
    
    if hw > E:
        return 0
    
    def get_Y(q):
        return get_ELF(hw, q, n_osc) / q
    
    qp = np.sqrt(2*mc.m)*(np.sqrt(E) + np.sqrt(E - hw))
    qm = np.sqrt(2*mc.m)*(np.sqrt(E) - np.sqrt(E - hw))
    
    return mc.h2si * 1/(np.pi * E) * integrate.quad(get_Y, qm, qp)[0] / 1e+2 ## m -> cm


def get_u(E, n_osc):
    
    def get_Y(hw):
        return get_tau(E, hw, n_osc)
    
    Eb = Ebind[n_osc]
    return integrate.quad(get_Y, Eb, (E + Eb)/2)[0]
#    return integrate.quad(get_Y, 0, (E)/2)[0]


def get_S(E, n_osc):
    
    def get_Y(hw):
        return get_tau(E, hw, n_osc) * hw
    
    Eb = Ebind[n_osc]
    return integrate.quad(get_Y, Eb, (E + Eb)/2)[0]
#    return integrate.quad(get_Y, 0, (E)/2)[0]


#%%
EE = mc.EE
#EE = np.logspace(0, 4.4, 50)


#%%
tau_4osc = np.zeros((4, len(EE), len(EE)))


for n in range(4):

    for i, E in enumerate(EE):
    
        mu.pbar(i, len(EE))
    
        for j, hw in enumerate(EE):
        
            tau_4osc[n, i, j] = get_tau(E, hw, n)


#%%
tau_4osc = np.load('PMMA/tau_4osc.npy')

tau_4osc_int = np.zeros(np.shape(tau_4osc))


for n in range(4):
    
    tau_4osc_int[n, :, :] = mu.diff2int(tau_4osc[n, :, :], V=mc.EE, H=mc.EE)
    


#%%
u = np.zeros((4, len(EE)))
S = np.zeros((4, len(EE)))


for n in range(4):

    for i, E in enumerate(EE):
    
        mu.pbar(i, len(EE))
        
        u[n, i] = get_u(E, n)
        S[n, i] = get_S(E, n)


#%% u
u = np.load('PMMA/u_4osc.npy')


for n in range(4):
    
    plt.loglog(EE, u[:, n], '--', label=str(n))

u_ee = np.sum(u, axis=1)


plt.loglog(EE, u_ee, '-', label='total')
#plt.loglog(EE, u_old, label='old')

l_Dapor = np.loadtxt('curves/l_dapor2015.txt')
plt.semilogx(l_Dapor[:, 0], 1/l_Dapor[:, 1]*1e+8, 'o', label='dapor2015.pdf')


plt.legend()
plt.grid()

plt.xlim(0, 1e+4)
plt.ylim(1e+3, 1e+8)


#%%
u_4osc = np.load('PMMA/u_4osc.npy')

u_4osc_norm = np.zeros(np.shape(u_4osc))


for i in range(len(mc.EE)):
    
    if np.all(u_4osc[i, :] == 0):
        continue
    
    u_4osc_norm[i, :] = u_4osc[i, :] / np.sum(u[i, :])
    


#%% S
S = np.load('PMMA/S_4osc.npy')


for n in range(4):
    
    plt.semilogx(EE, S[n, :], '--', label=str(n))


plt.semilogx(EE, np.sum(S, axis=0), '-', label='total')


S_Dapor = np.loadtxt('curves/S_dapor2015.txt')
plt.semilogx(S_Dapor[:, 0], S_Dapor[:, 1]*1e+8, 'o', label='dapor2015.pdf')


plt.legend()
plt.grid()

plt.xlim(0, 1e+4)
plt.ylim(0, 4e+8)


#%%
u_4osc = np.load('PMMA/u_4osc.npy').transpose()
u = np.load('PMMA/u.npy')

u_4osc[np.where(u_4osc == -0)] = 0



#%%
plt.loglog(mc.EE, np.sum(u_4osc, axis=1))
plt.loglog(mc.EE, u)


#%%
arr = np.load('PMMA/tau_4osc.npy')

result = np.zeros(np.shape(arr))

result[0, :, :] = arr[1, :, :]
result[1, :, :] = arr[0, :, :]
result[2:, :, :] = arr[2:, :, :]

