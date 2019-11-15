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

os.chdir(mc.sim_folder + 'DEBER_profile')


#%%
def func(xx, A, mu, sigma):
    return A * np.exp(-(xx-mu)**2/sigma)


#%%
n_dose = 2

profile = np.loadtxt('EXP_' + str(n_dose+1) + '.txt')

profile = profile[profile[:, 0].argsort()]

height = 0.9

xx, yy = profile[:, 0], profile [:, 1]
yy_900 = yy + height - yy[-1]

#plt.plot(xx, yy)

inv_yy = -(yy - yy.max())

plt.plot(xx, inv_yy, label='profile')

popt, pcov = curve_fit(func, xx, inv_yy)

plt.plot(xx, func(xx, *popt), '--', label='Gauss')

plt.plot(xx, np.ones(len(xx))*inv_yy.max()*0.3)

plt.title('DEBER profile for dose' + str(n_dose)\
          + ', fit: A=%5.2f, mu=%5.2f, sigma=%5.2f' % tuple(popt))
plt.xlabel('x, $\mu$m')
plt.xlabel('x, $\mu$m')
plt.legend()

plt.grid()
plt.show()


#%%
#plt.plot(xx, func(xx, 0.13, popt[1], 1.14) + 0.057) ## surface evolver 1
#plt.plot(xx, func(xx, 0.46, popt[1], 0.48) + 0.198) ## surface evolver 2
#plt.plot(xx, func(xx, 0.43, popt[1], 0.46) + 0.198)
#plt.plot(xx, func(xx, 0.48, popt[1], 0.36) + 0.198)
plt.plot(xx, func(xx, 0.69, popt[1], 0.63) + 0.3)

#plt.savefig('profile_fit_dose' + str(n_dose) + '.png', dpi=300)


#%%
#def func(xx, a, b, c):
#    return a*xx + b


#def func(xx, a, b, c):
#    return a*np.exp(-xx*b)


def func(xx, a, b, c):
    return a/((xx+b))


## 
#doses = np.array([0, 0.05, 0.2, 0.87])
#true_depths = 0.9 - np.array([0.0, 0.2, 0.66, 0.9])

## 0.7
#doses = np.array([0, 0.05, 0.2, 0.87])
#true_depths = 0.9 - np.array([0.0, 0.2, 0.66, 0.92])*0.7

## 0.7 + FIT
doses = np.array([0, 0.05, 0.2, 0.87])
#true_depths = 0.9 - np.array([0.0, 0.14, 0.66, 1])
true_depths = 0.9 - np.array([0.0, 0.19, 0.66, 1])*0.7


#p0 = 0.2, 0.18, 0.9

popt, pcov = curve_fit(func, doses, true_depths)

plt.plot(doses, true_depths, 'ro')

xx = np.linspace(0, 1, 100)

legend = 'a/(xx+b), a=%5.2f, b=%5.2f, c=%5.2f' % tuple(popt)

plt.plot(xx, func(xx, *popt), label=legend)

plt.title('True resist thickness')

plt.xlabel('dose, $\mu C/cm^2$')
plt.ylabel('resist thickness, $\mu m$')
plt.legend()

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.grid()
plt.show()

#plt.savefig('kinetic_curve_FIT_07_FINAL.png', dpi=300)

