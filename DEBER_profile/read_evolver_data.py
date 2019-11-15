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

from scipy.optimize import curve_fit

#from mpl_toolkits.mplot3d import Axes3D


#%%
def func(xx, A, sigma):
    return A * np.exp(-(xx)**2/sigma)


n_dose = 1

profile = np.loadtxt('/Users/fedor/Documents/DEBER-Simulation-2.0/Surface Evolver' +\
                  '/fe/trench/dose1/dose' + str(n_dose) + '_65.txt')

#profile = np.loadtxt('/Users/fedor/Documents/DEBER-Simulation-2.0/Surface Evolver' +\
#                  '/fe/trench/dose1/dose' + str(n_dose) + '_70.txt')

profile = profile[profile[:, 2].argsort(kind='mergesort')]

inds = np.where(np.abs(profile[:, 1]) < 0.5)[0]

xx = profile[inds, 2]
zz = profile[inds, 3]

#plt.plot(xx, zz, 'r.')


inds = np.where(np.logical_and(zz > 0.3, np.abs(xx) < 2))

xx_peak = xx[inds]
zz_peak = zz[inds].max() - zz[inds]

plt.plot(xx_peak, zz_peak, 'r.', label='evolver')

popt, pcov = curve_fit(func, xx_peak, zz_peak)

plt.plot(xx_peak, func(xx_peak, *popt), label='Gauss')

plt.title('DEBER profile for dose' + str(n_dose)\
          + ', fit: A=%5.2f, sigma=%5.2f' % tuple(popt))

plt.xlabel('x, $\mu$m')
plt.xlabel('x, $\mu$m')
plt.legend()

plt.grid()
plt.show()




