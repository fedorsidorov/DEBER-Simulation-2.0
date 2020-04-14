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

os.chdir(os.path.join(mc.sim_folder, '2ndary_yield'))


#%%
folder = '250_0p25'

source = os.path.join(mc.sim_folder, 'e_DATA', '2ndaries', folder)

filenames = os.listdir(source)

n_total = 0
n_2nd = 0


for fname in filenames:
    
    if fname == '.DS_Store':
        continue
    
    DATA = np.load(os.path.join(source, fname))
    
    n_total += 100
    n_2nd += DATA


my_d = n_2nd/n_total

print(my_d)


D_sim = np.loadtxt('Dapor_sim.txt')
D_exp = np.loadtxt('Dapor_exp.txt')

plt.plot(D_sim[:, 0], D_sim[:, 1], 'o', label='Dapor simulation')
plt.plot(D_exp[:, 0], D_exp[:, 1], 'o', label='experiment')

plt.plot(250, my_d, 'r*', label='my')

plt.legend()
plt.grid()

plt.xlim(0, 1500)
plt.ylim(0, 3)


