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


#%%
n_dose = 1

full_mat = np.load('../mapping_EXP/2um_mod/full_mat_dose' + str(n_dose) + '.npy')
mono_mat = np.load('../mapping_EXP/2um_mod/mono_mat_dose' + str(n_dose) + '.npy')


res_mat = np.zeros(np.shape(full_mat))


for xi, yi, zi in product(range(1000), range(5), range(450)):
    
    if full_mat[xi, yi, zi] == 0:
        res_mat[xi, yi, zi] = 1
        continue
    
    else:
        res_mat[xi, yi, zi] = (full_mat[xi, yi, zi] - mono_mat[xi, yi, zi]) / full_mat[xi, yi, zi]


#cs_mat = np.average(res_mat, axis=1)
cs_mat = res_mat[:, 2]

cs_arr = np.sum(cs_mat, axis=1) / 450

profile = cs_arr*0.9

x_centers = np.arange(-999, 1000, 2)

plt.plot(x_centers/1000, profile, label='simulation')

plt.title('Structure profile after monomer diffusion, dose ' + str(n_dose))
plt.xlabel('x, $\mu$m')
plt.ylabel('y, $\mu$m')

plt.ylim(0, 1)

plt.legend()
plt.grid()


#plt.savefig('profile_after_diffusion_dose' + str(n_dose) + '_2um.png', dpi=300)


#%%


volume_um2  = (xx[-1] - xx[0]) * height - np.trapz(yy_900, x=xx)



