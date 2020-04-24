#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
from scipy import interpolate

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)


os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'PMMA'
        ))


#%%
DIIMFP_0 = np.load('DIIMFP/DIIMFP_000-824.npy')[:824, :]
DIIMFP_1 = np.load('DIIMFP/DIIMFP_824-950.npy')[:126, :]
DIIMFP_2 = np.load('DIIMFP/DIIMFP_950-999.npy')[:50 , :]

DIIMFP = np.vstack((DIIMFP_0, DIIMFP_1, DIIMFP_2))


plt.loglog(mc.EE_prec, DIIMFP_0[-1, :])
plt.loglog(mc.EE_prec, DIIMFP[823, :], '--')

plt.loglog(mc.EE_prec, DIIMFP_1[-1, :])
plt.loglog(mc.EE_prec, DIIMFP[949, :], '--')

plt.loglog(mc.EE_prec, DIIMFP_2[-1, :])
plt.loglog(mc.EE_prec, DIIMFP[-1, :], '--')


#%% test IMFP - OK
IIMFP = np.zeros(len(mc.EE_prec))


for i, E in enumerate(mc.EE_prec):
    
    inds = np.where(mc.EE_prec <= E/2)
    
    IIMFP[i] = np.trapz(DIIMFP[i, inds], mc.EE_prec[inds])


IIMFP_int = mu.log_log_interp(mc.EE_prec, IIMFP)(mc.EE)

plt.loglog(mc.EE, 1/IIMFP_int * 1e+8, label='my IMFP')

DB = np.loadtxt('curves/Dapor_BOOK_grey.txt')
plt.loglog(DB[:, 0], DB[:, 1], '.', label='Dapor IMFP')

plt.xlim(1e+1, 1e+3)
plt.ylim(1, 100)

plt.grid()
plt.legend()


#%% interpolate it all
DIIMFP[np.where(DIIMFP == 0)] = 1e-100

DIIMFP_int = mu.log_log_interp_2d(mc.EE_prec, mc.EE_prec, DIIMFP)(mc.EE, mc.EE)

DIIMFP[np.where(DIIMFP < 1)] = 0
DIIMFP_int[np.where(DIIMFP_int < 1)] = 0

# plt.loglog(mc.EE_prec, DIIMFP[555, :])
# plt.loglog(mc.EE, DIIMFP_int[455, :], '--')

# plt.loglog(mc.EE_prec, DIIMFP[740, :])
# plt.loglog(mc.EE, DIIMFP_int[681, :], '--')

plt.loglog(mc.EE_prec, DIIMFP[241, :])
plt.loglog(mc.EE, DIIMFP_int[68, :], 'o')


#%%
IIMFP_test = np.zeros(len(mc.EE))


for i, E in enumerate(mc.EE):
    
    inds = np.where(mc.EE <= E/2)
    
    IIMFP_test[i] = np.trapz(DIIMFP_int[i, inds], mc.EE[inds])


# plt.loglog(mc.EE, 1/IIMFP_test * 1e+8, label='my IMFP')
plt.loglog(mc.EE, 1/IIMFP_int * 1e+8, '--', label='my IMFP int')


#%%
DIIMFP_int_norm = np.zeros((len(mc.EE), len(mc.EE)))


for i, _ in enumerate(mc.EE):
    
    if np.all(DIIMFP_int[i, :] == 0):
        continue
    
    DIIMFP_int_norm[i, :] = DIIMFP_int[i, :] / np.sum(DIIMFP_int[i, :])
    
  
IIMFP_int_final = IIMFP_int
IIMFP_int_final[:2] = 0


