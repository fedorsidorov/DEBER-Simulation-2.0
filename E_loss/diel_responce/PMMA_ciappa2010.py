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
        'E_loss', 'diel_responce'
        ))


#%%
## En, Gn, An from dapor2015.pdf
params = np.array((
         (286.5, 25.84, 14.75), ## plasmon first!!!
         (100.0, 19.46, 8.770),
         ( 80.0, 300.0, 140.0),
         ( 55.0, 550.0, 300.0),
         ))


#%% Na easichah
def S(x):
    
    f = (1-x)*np.log(4/x) - 7/4*x + x**(3/2) - 33/32*x**2
    
    return f


def G(x):
    
    f = np.log(1.166/x) - 3/4*x - x/4*np.log(4/x) + 1/2*x**(3/2) -\
        x**2/16*np.log(4/x) - 31/48*x**2
        
    return f


def get_oscillator_q0(E, An, En, wn):
    
    return An*wn*E / ((E**2 - En**2)**2 + (wn*E)**2)


def get_OLF(E):
    
    OLF = 0
    
    
    for arr in params:
        
        An, En, wn = arr
        OLF += get_oscillator_q0(E, An, En, wn)
    
    
    return OLF


def get_tau_u(E, hw):
    
    return 1/2 * mc.h2si * 1/(np.pi*E) * get_OLF(hw) * S(hw/E) / 1e+2 ## m -> cm
    

def get_tau_S(E, hw):
    
    return mc.h2si * 1/(np.pi*E) * get_OLF(hw) * G(hw/E) / 1e+2 ## m -> cm


def get_u(E):
    
    def get_Y(hw):
        return get_tau_u(E, hw)
    
    return integrate.quad(get_Y, 0, E/2)[0]


def get_S(E):
    
    def get_Y(hw):
        return get_tau_S(E, hw) * hw
    
    return integrate.quad(get_Y, 0, E/2)[0]


#%%
tau = np.load('PMMA/tau.npy')
u = np.sum(np.load('PMMA/4osc/u_4osc.npy'), axis=1)
SP = np.sum(np.load('PMMA/4osc/S_4osc.npy'), axis=1)

tau_u = np.zeros((len(mc.EE), len(mc.EE)))
tau_S = np.zeros((len(mc.EE), len(mc.EE)))

u_easy = np.zeros(len(mc.EE))
S_easy = np.zeros(len(mc.EE))


for i, E in enumerate(mc.EE):
    
    mu.pbar(i, len(mc.EE))
    
    
    for j, hw in enumerate(mc.EE):
        
        tau_u[i, j] = get_tau_u(E, hw)
        tau_S[i, j] = get_tau_S(E, hw)
    
    
    u_easy[i] = get_u(E)
    S_easy[i] = get_S(E)


#%%
tau_u[np.where(tau_u < 0)] = 0
tau_S[np.where(tau_S < 0)] = 0


#%%
tau_u_norm = np.zeros(np.shape(tau_u))
tau_S_norm = np.zeros(np.shape(tau_S))


for i in range(len(mc.EE)):
    
    if np.sum(tau_u[i, :]) != 0:
        tau_u_norm[i, :] = tau_u[i, :] / np.sum(tau_u[i, :])
    
    if np.sum(tau_S[i, :]) != 0:
        tau_S_norm[i, :] = tau_S[i, :] / np.sum(tau_S[i, :])
        


#%%
ind = 500

plt.loglog(mc.EE, tau[ind, :])
plt.loglog(mc.EE, tau_u[ind, :], '--')
plt.loglog(mc.EE, tau_S[ind, :], '--')

plt.grid()


#%%
plt.loglog(mc.EE, u)
plt.loglog(mc.EE, u_easy, '--')


#%%
plt.loglog(mc.EE, SP)
plt.loglog(mc.EE, S_easy, '--')


#%%
OLF_ref = np.load('OLF_total.npy')

OLF = np.zeros(len(mc.EE))


for i, E in enumerate(mc.EE):
    
    OLF[i] = get_OLF(E)



plt.loglog(mc.EE, OLF_ref)
plt.loglog(mc.EE, OLF)




