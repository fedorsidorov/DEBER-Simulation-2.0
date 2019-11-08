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

from scipy.optimize import curve_fit

os.chdir(mc.sim_folder + 'DEBER_exp')


#%%
table = np.loadtxt('prof5min_final/150/2500+.txt') * 1e+6

xx, yy = table[:, 0], table[:, 1]

plt.plot(xx, yy)

#%%
def func(xx, A1, S1, A2, S2):
    return A1 * np.exp(-xx**2/S1) - A2 * np.exp(-xx**2/S2)


p0 = [0.001, 0.1, 0.01, 0.01]

popt, pcov = curve_fit(func, xx, yy, p0=p0)

A1, S1, A2, S2 = popt

plt.plot(xx, yy, label='experiment')
#plt.plot(xx, A0*np.ones(len(xx)), '--', label='const')
plt.plot(xx, A1 * np.exp(-xx**2/S1), '--', label='Gauss 1')
plt.plot(xx, -A2 * np.exp(-xx**2/S2), '--', label='Gauss 2')
plt.plot(xx, func(xx, *popt), '--', label='Gauss sum')

plt.title('fit: A1=%5.3f, S1=%5.3f, A2=%5.3f, S2=%5.3f' % tuple(popt))
plt.xlabel('x, $\mu$m')
plt.xlabel('x, $\mu$m')
plt.legend()

plt.grid()
plt.show()

#plt.savefig('exp.png', dpi=300)

