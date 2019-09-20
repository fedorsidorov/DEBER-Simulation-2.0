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

from scipy.optimize import curve_fit


#%%
def func(x, A, mu, sigma):
    return 0.9 - A * 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2 / (2*sigma**2))


podgon = np.loadtxt('podgon.txt')

px = (podgon[:, 0] - 0.52)*15
py = (podgon[:, 1] - 0.08)*0.9 + 0.09

popt, pcov = curve_fit(func, px, py)

#plt.plot(px, py)
plt.plot(px, func(px, *popt), label='simulation')

exp = np.loadtxt('EXP.txt')
plt.plot((exp[:, 0] - 2.85), (exp[:, 1] + 0.7), label='experiment')

plt.title('Final structure profile, D = 0.05 $\mu C/cm^3$')
plt.xlabel('x, $\mu$m')
plt.ylabel('y, $\mu$m')

plt.legend(loc='lower right')

plt.xlim(-3, 3)
plt.ylim(0.4, 1)

plt.grid()

plt.savefig('PODGON.png', dpi=300)

