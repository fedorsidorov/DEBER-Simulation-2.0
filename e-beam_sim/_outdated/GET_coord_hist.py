#%% Import
import numpy as np
import importlib
import matplotlib.pyplot as plt
import os
import my_functions as mf
mf = importlib.reload(mf)
#import matplotlib.pyplot as plt

#%%
source_folder = 'DATA_25keV_160nm'
e_data_filenames = os.listdir(source_folder)

coord_bins = np.arange(-1000, 1001, 1)
hist = np.zeros(len(coord_bins) - 1)

n_files = 0
n_files_max = len(e_data_filenames)

for e_data_fname in e_data_filenames:
    
    DATA_Pn = np.load(source_folder + os.sep + e_data_fname)
    
    hist += np.histogram(DATA_Pn[:, 5], bins=coord_bins)[0]

#%%
coords = (coord_bins[:-1] + coord_bins[1:]) / 2
plt.semilogy(coords, hist / hist.max(), label='inelastic events')
plt.semilogy(coords, np.ones(len(coords))*1e-2, label='1%')
plt.semilogy(coords, np.ones(len(coords))*1e-3, label='0.1%')
plt.title('Inelastic events distribution - NORMED')
plt.xlabel('x, nm')
plt.ylabel('arbitrry unit')
plt.grid()
plt.legend()
plt.show()

