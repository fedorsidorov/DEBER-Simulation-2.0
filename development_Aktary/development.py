#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

import numpy.random as rnd

import my_constants as mc
import my_utilities as mu
import my_arrays_Dapor as ma

mc = importlib.reload(mc)
mu = importlib.reload(mu)
ma = importlib.reload(ma)

os.chdir(mc.sim_folder + 'development_Aktary')

import matplotlib


#%%
mat_3 = np.load('../e-matrix_Aktary/Aktary_e_matrix_val_1nm_100_4.npy')

mat_2 = np.average(mat_3, axis=1)

n_mon_bin = 7.095

sci_prob_mat_2 = mat_2 / n_mon_bin

M_final = 9500 / (1 + 9500 * sci_prob_mat_2) * 100

R = (84 + 3.14e+8   / M_final**1.5) / 10 ## Greeneich, A/min
#R = (0  + 9.33e+14 / M_final**3.86) / 10 ## Han, A/min

R_inv = 1/R

R_inv_1D = np.sum(R_inv, axis=0)


surface = np.zeros(len(R_inv[0]))


total_time = 0.5 ## min
dt = 0.001

steps = int(total_time / dt)

for t_step in range(steps):
    
    mu.pbar(t_step, steps)
    
    for i in range(len(R_inv[0])):
        
        surf_ind = 0
        
        for j in range(len(R_inv[1])):
            
            if R_inv[i, j] > 0:
                surf_ind = j
                break
        
        R_inv[i, surf_ind] -= dt


#%%
R_inv[np.where(R_inv < 0)] = 0
R_inv[np.where(R_inv > 0)] = 1

#plt.figure(figsize=[3.1496, 3.1496])
plt.figure(figsize=[5, 4.])

font_size = 10

matplotlib.rcParams['font.family'] = 'Times New Roman'

plt.imshow(R_inv.transpose())
#plt.colorbar()

plt.xlabel('x, нм')
plt.ylabel('z, нм')

plt.savefig('fig3.png', dpi=600)

