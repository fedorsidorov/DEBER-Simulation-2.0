#%% Import
import numpy as np
import os
import importlib

import my_constants as mc
mc = importlib.reload(mc)

import matplotlib.pyplot as plt

os.chdir(mc.sim_folder + 'EXP_weight_distribution')


#%%
def Flory_Schulz(k, N, p):
    return N * np.power(1-p, 2) * k * np.power(p, k-1)

def get_Mn(x, y):
    return np.sum(y * x) / np.sum(y)

def get_Mw(x, y):
    return np.sum(y * np.power(x, 2)) / np.sum(y * x)


#%%
mma_mass = 100
Mn_0 = 27.1e+4
Mw_0 = 66.9e+4

x_Mn_Mw = np.ones(100)
y_Mn_Mw = np.linspace(0, 10, len(x_Mn_Mw))


def Gauss(k, N_G, mu, sigma):
    return N_G * np.exp(-(k - mu)**2 / (2 * sigma**2))


def Flory_Schulz_mod(k, N, p, N_G, mu, sigma):
    return N * np.power(1-p, 2) * k * np.power(p, k-1) + Gauss(k, N_G, mu, sigma)  


#%%
N = 1e+6
p = 99.998963e-2
N_G = 0.1
mu = 1e+5
sigma = 1.008e+6

params = N, p, N_G, mu, sigma

#x_FS = np.linspace(0, 5e+4 * mma_mass, 10000)
x_FS = np.logspace(2, 7, 500)
y_FS = Flory_Schulz_mod(x_FS, *params)

x_Mn_Mw = np.ones(50)
y_Mn_Mw = np.linspace(0, 3, 50)

Mn_FS = get_Mn(x_FS, y_FS)
Mw_FS = get_Mw(x_FS, y_FS)

plt.semilogx(x_FS, y_FS, 'ro', label='Flory-Schulz + Gauss')
#plt.plot(x_Mn_Mw * Mn_FS, y_Mn_Mw, label='Mn_model')
#plt.plot(x_Mn_Mw * Mw_FS, y_Mn_Mw, label='Mw_model')
#plt.plot(x_Mn_Mw * Mn_0, y_Mn_Mw, '--', label='Mn_exp')
#plt.plot(x_Mn_Mw * Mw_0, y_Mn_Mw, '--', label='Mw_exp')

#plt.plot(x_FS, Flory_Schulz(x_FS, N, p), label='Flory-Schulz')
#plt.plot(x_FS, Gauss(x_FS, N_G, mu, sigma), label='Gauss')

#plt.xlim(0, 1e+6)
plt.legend()
#plt.gca().get_xaxis().get_major_formatter().set_powerlimits((0, 0))
plt.grid()
plt.show()
