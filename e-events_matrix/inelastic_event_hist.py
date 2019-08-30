#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt

import my_constants as mc

mc = importlib.reload(mc)

os.chdir(mc.sim_folder + 'e-events_matrix')


#%%
path = '../e_DATA/'

folders = ['e_DATA_Harris_PMMA_cut_1.5e-4', 'e_DATA_Harris_PMMA_cut_1e-4',
           'Harris_cut_PMMA', 'Harris_PMMA', 'DATA_PMMA_Lenovo']

sizes = [10, 10, 1, 1, 1]


#%%
n_bins = 501
x_bins = np.linspace(-500e-7, 500e-7, n_bins)
x_arr = (x_bins[:-1] + x_bins[1:]) / 2
x_hist = np.zeros(len(x_arr))

for folder in folders:
    
    filenames = os.listdir(path + folder)
    
    for fname in filenames:
        
        if fname == '.DS_Store':
            continue
        
        now_DATA_PMMA = np.load(path + folder + '/' + fname)
        
        now_DATA_PMMA_1 = now_DATA_PMMA[np.where(now_DATA_PMMA[:, 3] == 1)]
        
        x_hist += np.histogram(now_DATA_PMMA_1[:, 5], bins=x_bins)[0]


#%%
plt.semilogy(x_arr * 1e+7, x_hist / np.max(x_hist))

plt.title('Harris experiment inelastic events')
plt.xlabel('x, nm')
plt.ylabel('inelastic events')

plt.grid()
plt.show()

plt.savefig('Harris_inel_distr.png', dpi=300)

