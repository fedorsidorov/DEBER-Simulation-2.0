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
        'E_loss', 'diel_responce', 'Dapor'
        ))


#%%
def S(x):
    
    f = (1-x)*np.log(4/x) - 7/4*x + x**(3/2) - 33/32*x**2
    
    return f


def G(x):
    
    f = np.log(1.166/x) - 3/4*x - x/4*np.log(4/x) + 1/2*x**(3/2) -\
        x**2/16*np.log(4/x) - 31/48*x**2
        
    return f


Im_RH = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Ritsko_Henke', 'Ritsko_Henke_Im.npy'))
        # 'E_loss', 'diel_responce', 'Ritsko_Henke', 'Ritsko_Henke_Dapor_Im.npy'))


## cut
# Im_RH[:140] = 0


#%%
tau_u = np.zeros((len(mc.EE), len(mc.EE)))
tau_s = np.zeros((len(mc.EE), len(mc.EE)))

u = np.zeros(len(mc.EE))
s = np.zeros(len(mc.EE))


for i in range(len(mc.EE)):
    
    E = mc.EE[i]
    hw = mc.EE
    
    tau_u[i, :] = 1/2 * mc.h2si * 1/(np.pi*E) * Im_RH * S(hw/E) / 1e+2 ## m -> cm
    tau_s[i, :] =       mc.h2si * 1/(np.pi*E) * Im_RH * G(hw/E) / 1e+2 ## m -> cm
    
    inds = np.where(hw <= E/2)
    
    u[i] = np.trapz(tau_u[i, inds],          x=hw[inds])
    s[i] = np.trapz(tau_s[i, inds]*hw[inds], x=hw[inds])
    

#%%
plt.loglog(mc.EE, 1/u * 1e+8)

# Dapor_l = np.loadtxt('Dapot_thesis_l_ee.txt')
Dapor_l = np.loadtxt('curves/Dapor_book_l_ee.txt')
plt.loglog(Dapor_l[:, 0], Dapor_l[:, 1])

plt.grid()
plt.xlim(1e+1, 2e+4)
plt.ylim(1e+0, 1e+4)

# plt.savefig('u_trapz.png', dpi=300)


#%%
plt.semilogx(mc.EE, s / 1e+8)
Dapor_S = np.loadtxt('curves/Dapor_thesis_S_ee.txt')
plt.semilogx(Dapor_S[:, 0], Dapor_S[:, 1])


#%%
tau_u[np.where(tau_u < 0)] = 0
tau_s[np.where(tau_s < 0)] = 0


#%%
tau_u_norm = np.zeros(np.shape(tau_u))
tau_s_norm = np.zeros(np.shape(tau_s))


for i in range(len(mc.EE)):
    
    if np.sum(tau_u[i, :]) != 0:
        tau_u_norm[i, :] = tau_u[i, :] / np.sum(tau_u[i, :])
    
    if np.sum(tau_s[i, :]) != 0:
        tau_s_norm[i, :] = tau_s[i, :] / np.sum(tau_s[i, :])
        


#%%
def get_OLF(hw):
     
    if hw < mc.EE[0]:
        return 0
    
    else:
        return mu.log_interp1d(mc.EE, Im_RH)(hw)


def get_tau_u(E, hw):
    
    return 1/2 * mc.h2si * 1/(np.pi*E) * get_OLF(hw) * S(hw/E) / 1e+10 ## m -> Å
    

def get_tau_S(E, hw):
    
    return mc.h2si * 1/(np.pi*E) * get_OLF(hw) * G(hw/E) / 1e+10 ## m -> Å


def get_u(E):
    
    def get_Y(hw):
        return get_tau_u(E, hw)
    
    return integrate.quad(get_Y, 0, E/2)[0]


def get_S(E):
    
    def get_Y(hw):
        return get_tau_S(E, hw) * hw
    
    return integrate.quad(get_Y, 0, E/2)[0]


#%%
# tau = np.load('PMMA/tau.npy')
# u = np.sum(np.load('PMMA/4osc/u_4osc.npy'), axis=1)
# SP = np.sum(np.load('PMMA/4osc/S_4osc.npy'), axis=1)

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
plt.loglog(mc.EE, 1/u_easy)

# Dapor_l = np.loadtxt('Dapot_thesis_l_ee.txt')
plt.loglog(Dapor_l[:, 0], Dapor_l[:, 1])

plt.grid()
plt.xlim(1e+1, 2e+4)
plt.ylim(1e+0, 1e+4)

# plt.savefig('u_quad.png', dpi=300)


#%%
plt.semilogx(mc.EE, S_easy)
Dapor_S = np.loadtxt('Dapor_thesis_S_ee.txt')
plt.semilogx(Dapor_S[:, 0], Dapor_S[:, 1])


#%%
ind = 500

plt.loglog(mc.EE, tau_u[ind, :], '--')
plt.loglog(mc.EE, tau_S[ind, :], '--')

plt.grid()


#%%
plt.loglog(mc.EE, u)
plt.loglog(mc.EE, u_easy, '--')


#%%
plt.loglog(mc.EE, S_easy, '--')


#%%
OLF_ref = np.load('OLF_total.npy')

OLF = np.zeros(len(mc.EE))


for i, E in enumerate(mc.EE):
    
    OLF[i] = get_OLF(E)



plt.loglog(mc.EE, OLF_ref)
plt.loglog(mc.EE, OLF)




