#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
#import copy

#from itertools import product

import my_constants as mc
import my_utilities as mu
import my_arrays_Dapor as ma

mc = importlib.reload(mc)
mu = importlib.reload(mu)
ma = importlib.reload(ma)

#from scipy.optimize import curve_fit

os.chdir(mc.sim_folder + 'DEBER_exp')


#%%
def func(xx, A1, S1, A2, S2):
    return A1 * np.exp(-xx**2/S1) - A2 * np.exp(-xx**2/S2)


A1 = np.array([35, 43, 75]) * 1e-4
S1 = np.array([1275, 1285, 1327]) * 1e-4
A2 = np.array([80, 101, 198]) * 1e-4
S2 = np.array([146, 144, 143]) * 1e-4


#%%
xx = np.linspace(-1, 1, 1001)

for dose in [0, 1, 2]:

    yy = func(xx, A1[dose], S1[dose], A2[dose], S2[dose])    
    plt.plot(xx, yy*1e+3, label='dose ' + str(dose))


plt.title('simulated profiles for 160$^\circ$C')
plt.xlabel('x, $\mu$m')
plt.ylabel('z, $n$m')
plt.legend()

plt.grid()
plt.show()


plt.savefig('sim_160.png', dpi=300)

#print(-np.min(yy) / (np.max(yy) - np.min(yy)))


