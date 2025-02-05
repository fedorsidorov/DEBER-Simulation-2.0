#%% Import
import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import copy

from itertools import product

import my_constants as mc
import my_utilities as mu
#import my_arrays_Dapor as ma

mc = importlib.reload(mc)
mu = importlib.reload(mu)
#ma = importlib.reload(ma)

os.chdir(mc.sim_folder + '/DEBER_profile')

vols = [0.39, 1.31, 2.05]
vols_07 = [0.2, 0.64, 1.11]


#%%
plt.figure(figsize=[3.1496, 3.1496])

font_size = 12

matplotlib.rcParams['font.family'] = 'Times New Roman'

doses = ['25 пКл/см', '100 пКл/см', '435 пКл/см']

#n_dose = 2


for n_dose in range(3):

    full_mat_3 = np.load('../mapping_EXP/2um_CT_160C_reducing_3400/full_mat_dose' +\
                         str(n_dose) + '.npy')
    mono_mat_3 = np.load('../mapping_EXP/2um_CT_160C_reducing_3400/mono_mat_dose' +\
                         str(n_dose) + '.npy')
    
    full_mat = np.average(full_mat_3, axis=1)
    mono_mat = np.average(mono_mat_3, axis=1)
    
    res_mat = np.zeros(np.shape(full_mat))
    
    full_arr_pre = np.average(full_mat, axis = 1)
    #full_arr_pre = np.ones(1000) * 57
    
    full_arr_add = np.ones(1500) * 57
    
    full_arr = np.concatenate((full_arr_add, full_arr_pre, full_arr_add))
    
    
    mono_arr_pre = np.average(mono_mat, axis = 1) / 1
    
    mono_arr_add = np.zeros(1500)
    
    mono_arr = np.concatenate((mono_arr_add, mono_arr_pre, mono_arr_add))
    
    
    res_arr = np.zeros(np.shape(full_arr))
    
    
    extra_mons = 0
    
    for xi in reversed(range(2000)):
        
        diff = full_arr[xi] - mono_arr[xi] - extra_mons
        
        if diff < 0:
            res_arr[xi] = 0
            extra_mons += mono_arr[xi] - full_arr[xi]
        
        else:
            res_arr[xi] = (full_arr[xi] - mono_arr[xi] - extra_mons) / full_arr[xi]
            extra_mons = 0
    
    
    extra_mons = 0
    
    for xi in range(2000, 4000):
        
        diff = mono_arr[xi] + extra_mons - full_arr[xi]
        
        if diff > 0:
            res_arr[xi] = 0
            extra_mons += mono_arr[xi] - full_arr[xi]
            
        else:
            res_arr[xi] = - diff / full_arr[xi]
            extra_mons = 0
    
    
    profile = res_arr*0.9
    
    x_centers = np.arange(-3999, 4000, 2) / 1000
    
    plt.plot(x_centers, profile, label=str(doses[n_dose]))

height = 0.9

volume_um2 = (x_centers[-1] - x_centers[0]) * height - np.trapz(profile, x=x_centers)

vol_ratio = int(volume_um2 / vols_07[n_dose] * 100) / 100

#plt.title('Simulated DEBER structure profile after monomer diffusion', fontsize=font_size)
plt.xlabel(r'x, мкм', fontsize=font_size)
plt.ylabel(r'z, мкм', fontsize=font_size)

plt.xlim(-1.5, 1.5)
plt.ylim(0, 1)

plt.grid()

print(volume_um2)

plt.legend(fontsize=font_size, loc='lower right')

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)


#plt.savefig('profile_AD_paper_RUS.tiff', bbox_inches='tight', dpi=500)
#    plt.savefig('profile_AD_paper.pdf', bbox_inches='tight')

