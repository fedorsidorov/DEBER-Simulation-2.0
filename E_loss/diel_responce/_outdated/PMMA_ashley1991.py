#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce'
        ))


#%%
#EE_eV = np.logspace(0, 4.4, 1000)
EE_eV = np.logspace(-1, 4.4, 1000)

EE = EE_eV * mc.eV
qq = np.sqrt(2*mc.m*EE)

a0 = 5.29e-11 ## m

h2si = mc.k_el * mc.m * mc.e**2 / mc.hbar**2

#WW_ext = np.logspace(-1, 4.5, 5000) * mc.eV

## En, Gn, An from dapor2015.pdf
params = [
         [19.46, 8.770, 100.0],
         [25.84, 14.75, 286.5],
         [300.0, 140.0, 80.0],
         [550.0, 300.0, 55.0],
         ]


#%%
OLF_1d = np.zeros(len(EE))

for arr in params:
    
    E, G, A, = arr
    OLF_1d += A*G*EE_eV / ((E**2 - EE_eV**2)**2 + (G*EE_eV)**2)


#%%
plt.loglog(EE_eV, OLF_1d, 'ro', label='OLF, q = 0')

plt.xlabel('E, eV')
plt.ylabel('Im[-1/$\epsilon(\omega, 0)$]')

plt.xlim(1, 1e+4)
plt.ylim(1e-9, 1e+1)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('oscillators_Dapor.png', dpi=300)


#%%
tau_exc = np.zeros((len(EE), len(EE)))


for i in range(len(EE)):
    
    mu.pbar(i, len(EE))
    
    E = EE[i]

    for j in range(len(EE)):
        
        w = EE[j]
        
        w_min = 0
        
        y = w/E
        
        if w <= E/2:
            w_min = 0
        elif w <= 3/4*E:
            w_min = E*(2*y - 1)
        else:
            continue
        
        w_max = 2*E*(y - 1 + np.sqrt(1 - y))
        
        inds = np.where(np.logical_and(EE >= w_min, EE <= w_max))[0]
        
        wp = EE[inds]
        
        F = (w*(w - wp))**(-1) + ((E + wp - w)*(E - w))**(-1) +\
            (w*(w - wp)*(E + wp - w)*(E-w))**(-1/2)
        
        tau_exc[i, j] = h2si * 1/(2*np.pi*E) * np.trapz(OLF_1d[inds] * wp * F, x=EE[inds])


#%%
SS = np.zeros(len(EE))
uu = np.zeros(len(EE))


for i in range(len(EE)):
    
    SS[i] = np.trapz(tau_exc[i] * EE * np.heaviside(EE[i]/2 - EE, 1), x=EE)
    uu[i] = np.trapz(tau_exc[i] * np.heaviside(EE[i]/2 - EE, 1), x=EE)


#%% SP
plt.semilogx(EE_eV, SS / mc.eV / 1e+10, 'o', label='my')

SP_D_solid = np.loadtxt('curves/dEds_solid.txt')
plt.semilogx(SP_D_solid[:, 0], SP_D_solid[:, 1], label='Dapor_solid')

plt.xlim(1, 1e+4)
plt.ylim(0, 5)

plt.legend()
plt.grid()


#%%
plt.semilogx(EE_eV, uu / 1e+10, 'o', label='my')

U_D_solid = np.loadtxt('curves/IMFP_solid.txt')
plt.semilogx(U_D_solid[:, 0], 1/U_D_solid[:, 1], label='Dapor_solid')

plt.xlim(1, 1e+4)
plt.ylim(0, 2e-1)

plt.legend()
plt.grid()

