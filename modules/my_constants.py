#%%
import numpy as np

#%% Variables
#e = 4.8e-10
e = 1.6e-19
#m = 9.11e-28
m = 9.1e-31
m_eV = 511e+3
h = 6.626e-34
hbar = 1.054e-34
#eV = 1.6e-12
eV = 1.6e-19
E0 = 20e+3
Na = 6.02e+23
eps0 = 8.854e-12
k_el = 1 / (4*np.pi*eps0)
c = 3e+8

Z_H = 1
u_H = 1.01
rho_H = 8.988e-5
n_H =  rho_H*Na/u_H

Z_C = 6
u_C = 12
rho_C = 2.265
n_C =  rho_C*Na/u_C

Z_O = 8
u_O = 16
rho_O = 1.429e-3
n_O =  rho_O*Na/u_O

Z_Si = 14
u_Si = 28.09
rho_Si = 2.33
n_Si =  rho_Si*Na/u_Si

Z_PMMA = 3.6
u_PMMA = 100.12
rho_PMMA = 1.18
n_PMMA =  rho_PMMA*Na/u_PMMA
n_PMMA_at = n_PMMA*(5 + 2 + 8)

CONC_at = {'H': n_H, 'C': n_C, 'O': n_O, 'Si': n_Si}
CONC = [n_PMMA_at, n_PMMA_at, n_PMMA_at, n_Si]

#%%
sim_path_MAC = '/Users/fedor/Documents/DEBER-Simulation/'


#%%
uint16_max = 65535
