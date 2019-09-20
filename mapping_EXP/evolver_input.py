#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

from itertools import product

import my_constants as mc
import my_utilities as mu
import my_arrays_Dapor as ma

mc = importlib.reload(mc)
mu = importlib.reload(mu)
ma = importlib.reload(ma)

os.chdir(mc.sim_folder + 'mapping_EXP')


#%%
full_mat = np.load('full_mat_dose3.npy')
mono_mat = np.load('mono_mat_dose3.npy')

res_mat = np.zeros(np.shape(full_mat))

for xi, yi, zi in product(range(600), range(5), range(450)):
    
    if full_mat[xi, yi, zi] == 0:
        continue
    
    else:
        res_mat[xi, yi, zi] = (full_mat[xi, yi, zi] - mono_mat[xi, yi, zi]) / full_mat[xi, yi, zi]


#%%
cs_mat = np.average(res_mat, axis=1)

cs_arr = np.sum(cs_mat, axis=1) / 450

x_centers = np.arange(-599, 600, 2)

plt.plot(x_centers, cs_arr*900)

plt.title('Structure profile after monomer diffusion, D = 0.05 $\mu C / cm^3$')
plt.xlabel('x, nm')
plt.ylabel('z, nm')

plt.grid()

plt.savefig('profile_after_diffusion_dose1_FAKE.png', dpi=300)






