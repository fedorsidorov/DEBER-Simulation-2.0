#%% Import
import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
#import copy

#from itertools import product

import my_constants as mc
import my_utilities as mu
#import my_arrays_Dapor as ma

mc = importlib.reload(mc)
mu = importlib.reload(mu)
#ma = importlib.reload(ma)

os.chdir(mc.sim_folder + '/DEBER_profile')

vols = [0.39, 1.31, 2.05]
vols_07 = [0.2, 0.64, 1.11]

#from scipy.optimize import curve_fit


#%%
plt.figure(figsize=[3.1496, 3.1496])
font_size = 10

n_dose = 1

shifts = [2.837, 6.97, 6.85]

#SE_A = [0.13, 0.46, 0.69]
#SE_S = [1.14, 0.40, 0.63]

SE_A = [0.136, 0.47, 0.645]
SE_S = [1.14, 0.40, 0.625]

#popts = [0.192429, 1.28817], [0.660531, 1.21191], [0.99424, 1.4249]
popts = [0.192429, 1.9], [0.67, 0.68], [0.922, 1.0]
popt = popts[n_dose]

doses = [25, 100, 435]

offset = [0, 0, 25]
#offset = [0, 0, 0]


def func(xx, A, sigma):
    return 0.9 - A * np.exp(-(xx)**2/sigma)


def SE(xx, level):
    return level - SE_A[n_dose] * np.exp(-(xx)**2/SE_S[n_dose])


profile = np.loadtxt('EXP_' + str(n_dose+1) + '.txt')
profile = profile[profile[:, 0].argsort()]

height = 0.9

xx, yy = profile[:, 0], profile [:, 1]
xx = xx - shifts[n_dose]
yy = yy + height - yy.max()
plt.plot(xx, yy * 1000 + offset[n_dose], label='EXP')
#popt, pcov = curve_fit(func, xx, yy)

o7_level = (0.9 - (0.9-yy.min())*0.3)
plt.plot(xx, np.ones(len(xx))*o7_level * 1000 + offset[n_dose], 'k-.')
plt.plot(xx, SE(xx, o7_level) * 1000 + offset[n_dose], '--', label='SE')

plt.plot(xx, func(xx, *popt) * 1000 + offset[n_dose], '--', label='SIM')

plt.title('DEBER profile for dose ' + str(doses[n_dose]) + ' pC/cm')
plt.xlabel('x, μm')
plt.ylabel('y, nm')
plt.legend()

#plt.ylim(200, 1000)
#plt.ylim(0, 1000)

plt.grid()
plt.show()

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)


#plt.savefig('final_' + str(n_dose) + '_ATT.pdf', bbox_inches='tight')


#%%
#font_size = 10
#matplotlib.rcParams['font.family'] = 'Times New Roman'
#doses = ['25 pC/cm', '100 pC/cm', '435 pC/cm']
#
#n_dose = 0
#
#full_mat_3 = np.load('../mapping_EXP/2um_CT_160C_reducing_3400/full_mat_dose' +\
#                     str(n_dose) + '.npy')
#mono_mat_3 = np.load('../mapping_EXP/2um_CT_160C_reducing_3400/mono_mat_dose' +\
#                     str(n_dose) + '.npy')
#
#full_mat = np.average(full_mat_3, axis=1)
#mono_mat = np.average(mono_mat_3, axis=1)
#res_mat = np.zeros(np.shape(full_mat))
#full_arr_pre = np.average(full_mat, axis = 1)
#full_arr_add = np.ones(1500) * 57
#full_arr = np.concatenate((full_arr_add, full_arr_pre, full_arr_add))
#mono_arr_pre = np.average(mono_mat, axis = 1) / 1
#mono_arr_add = np.zeros(1500)
#mono_arr = np.concatenate((mono_arr_add, mono_arr_pre, mono_arr_add))
#res_arr = np.zeros(np.shape(full_arr))
#
#extra_mons = 0
#
#for xi in reversed(range(2000)):
#    diff = full_arr[xi] - mono_arr[xi] - extra_mons
#    if diff < 0:
#        res_arr[xi] = 0
#        extra_mons += mono_arr[xi] - full_arr[xi]
#    else:
#        res_arr[xi] = (full_arr[xi] - mono_arr[xi] - extra_mons) / full_arr[xi]
#        extra_mons = 0
#
#extra_mons = 0
#
#for xi in range(2000, 4000):
#    diff = mono_arr[xi] + extra_mons - full_arr[xi]
#    if diff > 0:
#        res_arr[xi] = 0
#        extra_mons += mono_arr[xi] - full_arr[xi]
#    else:
#        res_arr[xi] = - diff / full_arr[xi]
#        extra_mons = 0
#
#profile = res_arr*0.9
#x_centers = np.arange(-3999, 4000, 2) / 1000
#plt.plot(x_centers, profile, label=str(doses[n_dose]))
#height = 0.9
#volume_um2 = (x_centers[-1] - x_centers[0]) * height - np.trapz(profile, x=x_centers)
#vol_ratio = int(volume_um2 / vols_07[n_dose] * 100) / 100
#plt.title('Simulated DEBER structure profile after monomer diffusion', fontsize=font_size)
#plt.xlabel(r'x, μm', fontsize=font_size)
#plt.ylabel(r'y, μm', fontsize=font_size)
#
#plt.xlim(-1.5, 1.5)
#plt.ylim(0, 1)
#
#plt.grid()
#
#print(volume_um2)
#
#plt.legend(fontsize=font_size, loc='lower right')
#
#ax = plt.gca()
#for tick in ax.xaxis.get_major_ticks():
#    tick.label.set_fontsize(font_size)
#for tick in ax.yaxis.get_major_ticks():
#    tick.label.set_fontsize(font_size)

#
#plt.savefig('profile_AD_paper.jpg', bbox_inches='tight', dpi=1000)

