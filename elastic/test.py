#%% Import
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder, 'elastic'))


#%%
dapor_500eV = np.loadtxt('curves/Dapor_Si_500eV.txt')
dapor_1keV = np.loadtxt('curves/Dapor_Si_1keV.txt')
dapor_2keV = np.loadtxt('curves/Dapor_Si_2keV.txt')

plt.plot(dapor_500eV[:, 0], dapor_500eV[:, 1], '.', label='Dapor 500 eV')
plt.plot(dapor_1keV[:, 0], dapor_1keV[:, 1], '.', label='Dapor 1 keV')
plt.plot(dapor_2keV[:, 0], dapor_2keV[:, 1], '.', label='Dapor 2 keV')


EE = ['500 eV', '1 keV', '2 keV']

ind_500eV = 613
ind_1keV = 681
ind_2keV = 750

kind = 'muffin'

diff_cs_pnc = np.load('final_arrays/Si/Si_' + kind + '_diff_cs_plane_norm_cumulated.npy')


for i, ind in enumerate([ind_500eV, ind_1keV, ind_2keV]):
    
    now_diff_cs_pnc = diff_cs_pnc[ind, :]
    plt.plot(mc.THETA_deg, now_diff_cs_pnc, label='my ' + EE[i])
    

plt.grid()
plt.legend()

plt.xlabel('theta, degree')
plt.ylabel('probability')

plt.xlim(0, 180)
plt.ylim(0, 1)

plt.savefig('Si_compare.pdf')

