#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import my_constants as mc
import my_utilities as mu
import chain_functions as cf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
cf = importlib.reload(cf)

from scipy.optimize import curve_fit

os.chdir(mc.sim_folder + 'PMMA_sim')


#%%
def func(x, a, b, c):
    return 1 - a / (np.exp((np.log(x) - b)/c) + 1)


#%%
Mn_before = 5.63e+5
Mw_before = 2.26e+6

mat = np.loadtxt('curves/harris_before.txt')

x = mat[:, 0]
y = mat[:, 1]

plt.semilogx(x, y, 'ro', label='paper data')

popt, pcov = curve_fit(func, x, y)

x_fit = np.logspace(4, 8, 10001) ## before
#x_fit = np.logspace(2, 6, 500) ## after

y_fit = func(x_fit, *popt)

plt.semilogx(x_fit, y_fit, 'b-', label='fit')

plt.title('Harris initial integral molecular weight distribution')
plt.xlabel('molecular weight')
plt.ylabel('Distribution function')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Harris_integral_before.png', dpi=300)


#%%
x_diff = x_fit[:-1]

y_diff = np.diff(y_fit)
y_diff_n = y_diff / y_diff.max()

plt.semilogx(x_diff, y_diff_n, label='fit')

y_SZ = cf.schulz_zimm(x_diff, Mn_before, Mw_before)
y_SZ_n = y_SZ / np.max(y_SZ)

plt.semilogx(x_diff, y_SZ_n, label='Schulz-Zimm')

plt.title('Harris initial molecular weight distribution')
plt.xlabel('molecular weight')
plt.ylabel('density')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Harris_before.png', dpi=300)


#%%
Mn_fit = np.dot(x_diff, y_diff_n) / np.sum(y_diff_n)
Mw_fit = np.dot(np.power(x_diff, 2), y_diff_n) / np.dot(x_diff, y_diff_n)

Mn_SZ = np.dot(x_diff, y_SZ_n) / np.sum(y_SZ_n)
Mw_SZ = np.dot(np.power(x_diff, 2), y_SZ_n) / np.dot(x_diff, y_SZ_n)

names = 'Mn', 'Mw'

plt.plot(names, [Mn_fit, Mw_fit], '^-', label='fit')
plt.plot(names, [Mn_before, Mw_before], 'o-', label='paper')
plt.plot(names, [Mn_SZ, Mw_SZ], '+-', label='Schulz-Zimm')

plt.title('Harris Mn and Mw')
plt.ylabel('average M')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Mn_Mw_before.png', dpi=300)


#%% Test integral Schulz-Zimm distribution
y_SZ_mat = y_SZ.reshape((1, len(y_SZ)))

y_SZ_int = mu.diff2int(y_SZ_mat, [0], x_diff)[0]

plt.semilogx(x_diff, y_SZ_int, label='Schulz-Zimm')
plt.semilogx(x, y, 'o', label='paper data')
plt.semilogx(x_fit, y_fit, '-', label='fit')

plt.title('Harris initial integral molecular weight distribution')
plt.xlabel('molecular weight')
plt.ylabel('Distribution function')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Harris_integral_before.png', dpi=300)
