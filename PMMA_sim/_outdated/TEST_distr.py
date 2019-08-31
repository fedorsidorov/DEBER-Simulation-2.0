#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt

from math import gamma


#os.chdir(mv.sim_path_MAC + 'schulz-zimm_distribution')

#%%
def schulz_zimm(Mn, Mw, x):
    
    z = Mn / (Mw - Mn)
    l = 1 / (Mw - Mn)
    
#    f = l**z / gamma(z) * np.power(x, z-1) * np.exp(-l*x)
    f = l**z / (gamma(z) * Mn) * np.power(x, z) * np.exp(-l*x)
    
    return f

#%%
x = np.logspace(1, 7, 100000)

fx = schulz_zimm(5.63e+5, 2.26e+6, x)

plt.semilogx(x, fx, 'ro')

#%%
Mn = np.dot(x, fx) / np.sum(fx)
Mw = np.dot(np.power(x, 2), fx) / np.dot(x, fx)
