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

os.chdir(mc.sim_folder + 'e-matrix_Harris')


#%%
def lin_f(xx, k, b):
    
    return(xx*k + b)


mat = np.loadtxt('charlesby1964/electrons.txt')
#mat = np.loadtxt('charlesby1964/gamma.txt')

x_data, y_data = mat[:, 0], mat[:, 1]

plt.plot(x_data, y_data, 'ro')


#%%
#mat = np.loadtxt('charlesby1964/electrons.txt')
#x_data, y_data = mat[:, 0], mat[:, 1]

x_data = 1000 / (np.array([-78, 0, 20, 100]) + 273)
#y_data = np.array([0.75, 1.4, 1.5, 2.3])
y_data = np.log(np.array([0.75, 1.4, 1.5, 2.3]))

popt, pcov = curve_fit(lin_f, x_data, y_data)

k, b = popt

plt.plot(x_data, y_data, 'ro')

xx = np.linspace(0, 5)

plt.plot(xx, lin_f(xx, *popt))

plt.xlabel('10$^3$/T')
plt.ylabel('log(G)')

plt.grid()


#%%
xx = np.linspace(30, 160, 100)
yy = np.exp(lin_f(1000/(xx+273), k, b))

plt.plot(xx, yy)


#%%
popt_new = np.array((popt[0], 2.14))


