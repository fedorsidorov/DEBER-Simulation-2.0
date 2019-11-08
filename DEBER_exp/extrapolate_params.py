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
def func(xx, k, b):
    return k*xx + b


#%% 150 C
#D = np.array([150, 200, 300, 500, 700, 1000, 1200, 1500, 2000, 2500])
#A1 = np.array([3, 3, 4, 6, 8, 11, 11, 12, 14, 23]) * 1e-3
#S1 = np.array([68, 62, 100, 114, 122, 102, 122, 129, 134, 122]) * 1e-3
#A2 = np.array([7, 9, 14, 21, 24, 35, 35, 39, 52, 64]) * 1e-3
#S2 = np.array([12, 11, 12, 11, 11, 12, 10, 11, 16, 18]) * 1e-3


#%% 130 C
D = np.array([500, 700, 1000, 1200, 1500, 2000, 2500])
A1 = np.array([1, 2, 16, 17, 18, 19, 18]) * 1e-3
S1 = np.array([106, 126, 71, 82, 69, 82, 79]) * 1e-3
A2 = np.array([8, 12, 32, 33, 36, 35, 36]) * 1e-3
S2 = np.array([6, 8, 8, 9, 8, 9, 9]) * 1e-3
#
##%% 120 C
#D = np.array([1000, 1200, 1500, 2000, 2500])
#A1 = np.array([3, 4, 4, 3, 4]) * 1e-3
#S1 = np.array([55, 35, 45, 55, 48]) * 1e-3
#A2 = np.array([16, 17, 17, 18, 20]) * 1e-3
#S2 = np.array([3, 4, 3, 3, 3]) * 1e-3

arr = A2

plt.plot(D, arr, 'ro')

popt, pcov = curve_fit(func, D, arr)
k, b = popt

plt.plot(D, func(D, *popt), '--')


#%%
print(func(435, k, b))


#%% Make this tough shit!!
T = np.array([120, 130, 150])
A1 = np.array([[33, 33, 34], [153, 155, 159], [21, 27, 52]]) * 1e-4
S1 = np.array([[431, 433, 442], [698, 701, 715], [1058, 1066, 1100]]) * 1e-4
A2 = np.array([[137, 139, 147], [306, 308, 316], [74, 91, 168]]) * 1e-4
S2 = np.array([[37, 37, 36], [77, 78, 80], [117, 116, 114]]) * 1e-4

arr = S2
dose = 2

popt, pcov = curve_fit(func, T, arr[:, dose])
k, b = popt

print(func(160, k, b))


#%%








