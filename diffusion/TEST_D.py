#%% Import
import numpy as np
from matplotlib import pyplot as plt, cm
import os
import importlib
import copy
from itertools import product

import my_constants as mc

mc = importlib.reload(mc)

os.chdir(mc.sim_folder + 'diffusion')


#%% Free-Volume parameters: faldi1994.pdf
V1 = 0.87
V2 = 0.757
K11g = 0.815e-3
K21 = 143
K12g = 0.477e-3
K22 = 52.38
Tg1 = 143
Tg2 = 392
a = 0.44

R = 1.98
#R = 8.31

D0 = 1.61e-3
E = 778
ksi = 0.60


def get_Dmma(w2, T_C):
    
    T = T_C + 273
    
    Vfg = (1-w2)*(K11g)*(K21 + T - Tg1) + w2*(K12g)*(K22 + a*(T-Tg2))
    
    logDmma = np.log10(D0) - E/(2.303 * R * T) - 1/2.303 *\
        ( ( (1-w2)*V1 + ksi*w2*V2 ) / Vfg )
    
    return np.power(10, logDmma)


#%% Chen 50 C
chen_wp = np.array((0.5, 0.4, 0.3, 0.2, 0.1))
chen_D = np.array((2.4, 16.5, 34.5, 37.9, 39)) * 1e-7


#%%
diff_points = np.loadtxt('diff_points.csv')


#%%
#plt.plot(diff_points[:, 0], diff_points[:, 1], '--', label='Faldi points')

#plt.semilogy(chen_wp, chen_D, 'ro', label='Chen T = 50$\degree$C')

wp = np.linspace(0, 0.99, 100)

#plt.semilogy(wp, get_Dmma(wp, 50), label='Faldi T = 50$\degree$C')
#plt.semilogy(wp, get_Dmma(wp, 120), label='Faldi T = 120$\degree$C')
#plt.semilogy(wp, get_Dmma(wp, 160), label='Faldi T = 160$\degree$C')

plt.semilogy(wp, get_Dmma(wp, 20), label='T = 20$\degree$C')
plt.semilogy(wp, get_Dmma(wp, 50), label='T = 50$\degree$C')
plt.semilogy(wp, get_Dmma(wp, 120), label='T = 120$\degree$C')
plt.semilogy(wp, get_Dmma(wp, 160), label='T = 160$\degree$C')

plt.title('Diffusion coefficient for MMA in PMMA (free volume theory)')
plt.xlabel('$\omega_p$')
plt.ylabel('$D_{MMA}$, cm$^2$/s')

#plt.semilogy(wp, get_Dmma(wp, 20), label='Faldi T = 20$\degree$C')
#plt.semilogy(wp, get_Dmma(wp, -10), label='Faldi T = -10$\degree$C')

plt.xlim(0, 1)
plt.ylim(1e-8, 1e-4)

plt.legend()
plt.grid()
plt.show()

plt.savefig('diffusitivi.png', dpi=300)

