import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

import my_constants as mc
mc = importlib.reload(mc)

os.chdir(os.path.join(mc.sim_folder, 'E_loss'))


#%%
def get_e_density(E0, Q, z):
    
    q = 1.6e-19
    rho = 1.19
    
    RG = 4.6e-6 / rho * (E0 / 1e+3)**1.75
    f = z / RG
    lambda_f = 0.74 + 4.7*f - 8.9*f**2 + 3.5*f**3
    
    eps = Q/q * E0/RG * lambda_f
    
    return eps

#%%
z = np.linspace(0, 7e-4, 1000)

eps_5  = get_e_density( 5e+3, 1e-4, z)
eps_10 = get_e_density(10e+3, 1e-4, z)
eps_20 = get_e_density(20e+3, 1e-4, z)
eps_30 = get_e_density(30e+3, 1e-4, z)

plt.semilogy(z[:95] * 1e+4, eps_5[:95])
plt.semilogy(z[:317] * 1e+4, eps_10[:317])
plt.semilogy(z * 1e+4, eps_20)
plt.semilogy(z * 1e+4, eps_30)

plt.xlim(0, 7)
plt.ylim(1e+21, 1e+23)

plt.xlabel('глубина (z), мкм')
plt.ylabel(r'$\varepsilon$, эВ/см$^3$')

plt.grid()
plt.show()

#plt.savefig('depth-dose.png', dpi=300)

