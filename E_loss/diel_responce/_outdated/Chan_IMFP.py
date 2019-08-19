#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import my_constants as mc
import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mc = importlib.reload(mc)

os.chdir(mv.sim_path_MAC + 'diel_responce')

#%%
E = np.load('E_SI_PALIK.npy')
IM = np.load('IM_PALIK.npy')
DIFF_MU = np.load('DIFF_MU_SI_PALIK.npy')

plt.loglog(E, DIFF_MU[4545, :] / 1e+2 * mc.eV, label='My')

#%%
def L(x):
    f = (1-x)*np.log(4/x) - 7/4*x + x**(3/2) - 33/32*x**2
    return f

#%%
MU = np.zeros(len(E))

for i in range(len(E)):
    
    now_E = E[i]
    
    inds = np.where(E <= now_E/2)[0]
    
    MU[i] = np.trapz(DIFF_MU[i, inds], x=E[inds]*mc.eV)

#%%
MU_2 = np.zeros(len(E))
DIFF_MU_2 = np.zeros((len(E), len(E)))

for i in range(len(E)):
    
    now_E = E[i]
    
    inds = np.where(E <= now_E/2)[0]
    
    y_MU_2 = IM[inds] * L(E[inds]/E[i])
    
    MU_2[i] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 * E[i]*mc.eV) *\
        np.trapz(y_MU_2, x=E[inds]*mc.eV)
    
    DIFF_MU_2[i, inds] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 *\
        E[i]*mc.eV) * IM[inds] * L(E[inds]/E[i])

#%%
plt.loglog(E, 1/MU * 1e+2, label='My')
plt.loglog(E, 1/MU_2 * 1e+2, label='My 2')

sun_hui = np.loadtxt('curves/sun hui v chai.txt')

plt.loglog(sun_hui[:, 0], sun_hui[:, 1], label='Chan')

#plt.xlim(10, 10000)
plt.ylim(1e-8, 1e+2)

plt.title('Si electron inelastic MFP')
plt.xlabel('E, eV')
plt.ylabel('IMFP, $\AA$')
plt.legend()
plt.grid()
plt.show()

#%%
E_PALIK = np.load('E_SI_PALIK.npy')
MU_PALIK = np.load('MU_SI_PALIK.npy')

CS_PALIK = MU_PALIK * 1e-2 / mc.n_Si

E_GRYZ = np.load('E_SI_GRYZ.npy')
CS_GRYZ = np.load('SI_CS_TOTAL_GRYZ.npy')

plt.ylim(1e-21, 1e-15)

plt.loglog(E_PALIK, CS_PALIK, label='dielectric responce')
plt.loglog(E_GRYZ, CS_GRYZ, label='Gryzinski')

plt.title('Si electron inelastic CS')
plt.xlabel('E, eV')
plt.ylabel('$\sigma$, cm$^2$')
plt.legend()
plt.grid()
plt.show()
