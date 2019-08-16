#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import my_constants as mc
import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mc = importlib.reload(mc)

os.chdir(mv.sim_path_MAC + 'phonons&polarons')

#%%
def get_l_ph_inv(E):
    
    a0 = 5.292e-11
    T = 300
    kT = 8.612e-5 * T ## eV
    hw = 0.1
    e_0 = 3.9
    e_inf = 2.2
    
    nT = 1 / (np.exp(hw/(kT)) - 1)
    KT = 1/a0 * ((nT + 1)/2) * ((e_0 - e_inf)/(e_0*e_inf))
    
    l_ph_inv = KT * hw/E * np.log( (1 + np.sqrt(1 - hw/E)) / (1 - np.sqrt(1 - hw/E)) )
    
    return l_ph_inv


def get_l_pol_inv(E):
    
    C = 1 ## nm^-1
    gamma = 0.15 ## ev^-1
    
    l_pol_inv = C * np.exp(-gamma * E)
    
    return l_pol_inv

#%%
E_arr = np.logspace(-1, 3, 1000)

L_ph_inv_arr = get_l_ph_inv(E_arr)
L_pol_inv_arr = get_l_pol_inv(E_arr)

L_ph_arr = 1 / L_ph_inv_arr * 1e+10
L_pol_arr = 1 / L_pol_inv_arr * 1e+1

plt.loglog(E_arr, L_ph_arr, label='phonon')
plt.loglog(E_arr, L_pol_arr, label='polaron')

plt.xlim(1e-1, 100)
plt.ylim(1, 1e+5)

plt.title('PMMA phonon mean free path')
plt.xlabel('E, eV')
plt.ylabel('$\lambda$, $\AA$')
plt.legend()
plt.grid()
plt.show()
