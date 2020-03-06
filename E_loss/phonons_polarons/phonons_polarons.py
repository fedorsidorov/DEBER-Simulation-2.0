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

os.chdir(mv.sim_path_MAC + 'E_loss/phonons_polarons')


#%%
def get_PMMA_U_phonon(EE, T=300):
    
    a0 = 5.292e-11
#    T = 300
    kT = 8.612e-5 * T ## eV
    hw = 0.1
    e_0 = 3.9
    e_inf = 2.2
    
    nT = 1 / (np.exp(hw/(kT)) - 1)
    KT = 1/a0 * ((nT + 1)/2) * ((e_0 - e_inf)/(e_0*e_inf))
    
    U_PH = KT * hw/EE * np.log( (1 + np.sqrt(1 - hw/EE)) / (1 - np.sqrt(1 - hw/EE)) ) ## m^-1
    
    return U_PH * 1e-2 ## cm^-1


def get_PMMA_U_polaron(EE):
    
    C = 1 ## nm^-1
    gamma = 0.15 ## ev^-1
    
    U_POL = C * np.exp(-gamma * EE) * 1e+7 ## cm^-1
    
    return U_POL


#%%
EE = mv.EE    

U_PH = get_PMMA_U_phonon(mv.EE)
U_POL = get_PMMA_U_polaron(mv.EE)

plt.loglog(mv.EE, U_PH, label='phonon')
plt.loglog(mv.EE, U_POL, label='polaron')

plt.xlim(1, 100)
plt.ylim(1e+5, 1e+7)

plt.title('PMMA phonon and polaron U')
plt.xlabel('E, eV')
plt.ylabel('U, cm$^{-1}$')
plt.legend()
plt.grid()
plt.show()

#plt.savefig('PMMA_ph_pol_U.png', dpi=300)


#%%
L_PH = 1 / U_PH[:500]
L_POL = 1 / U_POL[:500]

plt.loglog(mv.EE[:500], L_PH, label='phonon')
plt.loglog(mv.EE[:500], L_POL, label='polaron')

plt.xlim(1, 100)
plt.ylim(1e-8, 1)

plt.title('PMMA phonon and polaron $\lambda$')
plt.xlabel('E, eV')
plt.ylabel('$\lambda$, cm')
plt.legend()
plt.grid()
plt.show()

#plt.savefig('PMMA_ph_pol_L.png', dpi=300)


#%%
U_PH_room = get_PMMA_U_phonon(mv.EE)
U_PH_120 = get_PMMA_U_phonon(mv.EE, 273 + 120)

plt.loglog(mv.EE, U_PH_room, label='27 C$^\circ$')
plt.loglog(mv.EE, U_PH_120, label='120 C$^\circ$')

plt.xlim(1, 100)
plt.ylim(1e+5, 1e+7)

plt.title('PMMA phonon U for 27 C$^\circ$ and 120 C$^\circ$')
plt.xlabel('E, eV')
plt.ylabel('U, cm$^{-1}$')
plt.legend()
plt.grid()
plt.show()

plt.savefig('PMMA_pol_L_27_120.png', dpi=300)

