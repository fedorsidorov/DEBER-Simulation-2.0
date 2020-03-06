#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import my_constants as mc
import my_utilities as mu
import Gryzinski as gryz

mc = importlib.reload(mc)
mu = importlib.reload(mu)
gryz = importlib.reload(gryz)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'MuElec'
        ))


#%%
diff_file = np.loadtxt('microelec/sigmadiff_inelastic_e_Si.dat')
diff_file = diff_file[np.where(diff_file[:, 0] <= 30e+3)]
diff_file[np.where(diff_file == 0)] = 10**(-100)

diff_EE_unique = np.unique(diff_file[:, 0])

EE = mc.EE

diff_6osc_pre = np.ones((6, len(diff_EE_unique), len(mc.EE))) * 10**(-100)
diff_6osc = np.zeros((6, len(mc.EE), len(mc.EE)))
#diff_6osc = np.ones((6, len(mc.EE), len(mc.EE))) * 10**(-100)


for i, E in enumerate(diff_EE_unique):
    
    mu.pbar(i, len(diff_EE_unique))
    
    inds = np.where(diff_file[:, 0] == E)[0]
    
    HW = diff_file[inds, 1]
    now_diff = diff_file[inds, 2:]
    
    for n in range(6):
        interp_inds = np.where(np.logical_and(EE >= HW.min(), EE <= HW.max()))[0]        
        diff_6osc_pre[n, i, interp_inds] = mu.log_interp1d(HW, now_diff[:, n])(EE[interp_inds])


for n in range(6):
    
    mu.pbar(n, 6)
    
    for j, _ in enumerate(EE):
        interp_inds = np.where(EE > 10)[0]  
        diff_6osc[n, interp_inds, j] =\
            mu.log_interp1d(diff_EE_unique, diff_6osc_pre[n, :, j])(EE[interp_inds])


#%%








