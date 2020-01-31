#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

import numpy.random as rnd

import my_constants as mc
import my_utilities as mu
import my_arrays_Dapor as ma

mc = importlib.reload(mc)
mu = importlib.reload(mu)
ma = importlib.reload(ma)

os.chdir(mc.sim_folder + 'e-matrix_Harris')

import e_matrix_functions as emf
emf = importlib.reload(emf)

import scission_functions as sf
sf = importlib.reload(sf)

import MC_functions_Dapor as mcd
mcd = importlib.reload(mcd)

from scipy.optimize import curve_fit


#%%
def lin_f(xx, k, b):    
    return(xx*k + b)


#%%
weights = np.load('weights.npy')
Gs_array = (np.load('Gs_array_0.npy') + np.load('Gs_array_1.npy') + np.load('Gs_array_2.npy')) / 3

plt.plot(weights, Gs_array, 'ro', label='simulation')

popt, pcov = curve_fit(lin_f, weights, Gs_array)

k, b = popt

xx = np.linspace(0, 2, 100)

plt.plot(xx, lin_f(xx, *popt), label='linear fit')

plt.xlabel('ester group bond weight')
plt.ylabel('G(S)')
plt.title('G(S) dependence on ester group bond weight')

plt.xlim(0, 2)
plt.ylim(1.8, 3.2)

plt.legend()

plt.grid()

plt.savefig('G(S)_ester.png', dpi=300)


#%%
tt = np.linspace(30, 160)
ww = np.zeros(len(tt))

Gs_fit = lin_f(weights, *popt)

for i, t in enumerate(tt):

    now_G = sf.get_Gs_charlesby(t)
    ww[i] = weights[mcd.get_closest_el_ind(Gs_fit, now_G)]


plt.plot(tt, ww)

plt.xlabel('T, C$^\circ$')
plt.ylabel('ester group bond weight')
plt.title('ester group bond weight dependence on T')

plt.xlim(40, 160)
plt.ylim(0, 2)

plt.grid()

plt.savefig('W_ester_T.png', dpi=300)

