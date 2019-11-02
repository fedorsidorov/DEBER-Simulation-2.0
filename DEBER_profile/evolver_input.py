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

os.chdir(mc.sim_folder + 'DEBER_profile')


vols = [0.39, 1.31, 2.05]


#%%
n_dose = 1

full_mat_3 = np.load('../mapping_EXP/2um_CT_0/full_mat_dose' + str(n_dose) + '.npy')
mono_mat_3 = np.load('../mapping_EXP/2um_CT_0/mono_mat_dose' + str(n_dose) + '.npy')

full_mat = np.average(full_mat_3, axis=1)
mono_mat = np.average(mono_mat_3, axis=1)

res_mat = np.zeros(np.shape(full_mat))


for xi, yi, zi in product(range(1000), range(5), range(450)):
    
    if full_mat[xi, yi, zi] == 0:
        res_mat[xi, yi, zi] = 1
        continue
    
    else:
        res_mat[xi, yi, zi] = (full_mat[xi, yi, zi] - mono_mat[xi, yi, zi]) /\
            full_mat[xi, yi, zi]
#        res_mat[xi, yi, zi] = (full_mat[xi, yi, zi] - mono_mat[xi, yi, zi]) / 57


cs_mat = np.average(res_mat, axis=1)
#cs_mat = res_mat[:, 4, :]

cs_arr = np.sum(cs_mat, axis=1) / 450 / 2

profile = cs_arr*0.9

x_centers = np.arange(-999, 1000, 2) / 1000

plt.plot(x_centers, profile, label='simulation')

height = 0.9

volume_um2  = (x_centers[-1] - x_centers[0]) * height - np.trapz(profile, x=x_centers)

plt.title('Structure profile after monomer diffusion, dose ' + str(n_dose) +\
          ', R = ' + str(int(volume_um2 / vols[n_dose-1] * 100) / 100))
plt.xlabel('x, $\mu$m')
plt.ylabel('y, $\mu$m')

plt.ylim(0, 1.2)

plt.legend()
plt.grid()


#plt.savefig('profile_after_diffusion_dose' + str(n_dose) + '_2um_CT_0.png', dpi=300)

