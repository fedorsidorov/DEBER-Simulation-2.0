#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import matplotlib.pyplot as plt

mc = importlib.reload(mc)

from scipy.optimize import curve_fit

os.chdir(mc.sim_folder + 'Harris')


#%%
## peak A
#Mn_harris = 5.63e+5
#Mw_harris = 2.26e+6

## peak B
#Mn_harris = 2370
#Mw_harris = 8160


#%%
mat = np.loadtxt('curves/before.txt')

x = mat[:, 0]
y = mat[:, 1]

plt.semilogx(x, y, 'ro')


#%%
def func(x, a, b, c):
    return 1 - a / (np.exp((np.log(x) - b)/c) + 1)


popt, pcov = curve_fit(func, x, y)

## peak A
xdata = np.logspace(4, 8, 500)

## peak B
#xdata = np.logspace(2, 6, 500)
ydata = func(xdata, *popt)


plt.semilogx(xdata, ydata, 'b-')


#%%
x_diff_rough = x[:-1]
y_diff_rough = np.diff(y)
y_diff_rough_n = y_diff_rough / np.max(y_diff_rough)

plt.semilogx(x_diff_rough, y_diff_rough_n, label='diff')

#%%
Mn_rough = np.dot(x_diff_rough, y_diff_rough_n) / np.sum(y_diff_rough_n)
Mw_rough = np.dot(np.power(x_diff_rough, 2), y_diff_rough_n) / np.dot(x_diff_rough, y_diff_rough_n)

#%%
x_diff = xdata[:-1]

y_diff = np.diff(ydata)
y_diff_n = y_diff / y_diff.max()

plt.semilogx(x_diff, y_diff_n, label='fit')

#%%
y_SZ = mf.get_schulz_zimm(Mn_harris, Mw_harris, x_diff)
y_SZ_n = y_SZ / np.max(y_SZ)

plt.semilogx(x_diff, y_SZ_n, label='Schulz-Zimm')

#%%
Mn = np.dot(x_diff, y_diff_n) / np.sum(y_diff_n)
Mw = np.dot(np.power(x_diff, 2), y_diff_n) / np.dot(x_diff, y_diff_n)

#%%
Mn_SZ = np.dot(x_diff, y_SZ_n) / np.sum(y_SZ_n)
Mw_SZ = np.dot(np.power(x_diff, 2), y_SZ_n) / np.dot(x_diff, y_SZ_n)

#%%
plt.legend()
plt.grid()
