#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)

from scipy.optimize import curve_fit

os.chdir(mv.sim_path_MAC + 'make_chains')

#%%
mat = np.loadtxt('harris1973_B.dat')

x_log = np.log10(mat[:, 0])
y = mat[:, 1]

plt.plot(x_log, y, 'ro')

def func(x, a, b, c):
    
    return 1 - a / (np.exp((x - b)/c) + 1)

popt, pcov = curve_fit(func, x_log, y)

xdata = np.linspace(x_log.min(), x_log.max(), 501)
ydata = func(xdata, *popt)

plt.plot(xdata, ydata, 'b-')

#%%
x_diff = xdata[:-1]

y_diff = np.diff(ydata)
y_diff_n = y_diff / y_diff.max()

plt.plot(x_diff, y_diff_n)

#%%
mat_A_diff = np.zeros((np.shape(x_diff)[0], 2))

mat_A_diff[:, 0] = x_diff
mat_A_diff[:, 1] = y_diff_n
