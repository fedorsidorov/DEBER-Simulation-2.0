import numpy as np


#%%
sim_path_MAC = '/Users/fedor/Documents/DEBER-Simulation-2.0/'
#sim_path_FTIAN = '/home/fedor/Yandex.Disk/Study/Simulation/'


#%% SI!!!
e = 1.6e-19
m = 9.1e-31
m_eV = 511e+3
h = 6.626e-34
hbar = 1.054e-34
eV = 1.6e-19
E0 = 20e+3
Na = 6.02e+23
eps0 = 8.854e-12
k_el = 1 / (4*np.pi*eps0)
c = 3e+8

hw_phonon = 0.1


#%% PMMA ans Si
Z_H = 1
u_H = 1.01
rho_H = 8.988e-5
#n_H =  rho_H * Na/u_H

Z_C = 6
u_C = 12
rho_C = 2.265
#n_C =  rho_C * Na/u_C

Z_O = 8
u_O = 16
rho_O = 1.429e-3
#n_O =  rho_O * Na/u_O

Z_Si = 14
u_Si = 28.09
rho_Si = 2.33
n_Si =  rho_Si * Na/u_Si

Z_PMMA = 3.6
u_PMMA = 100.12
rho_PMMA = 1.18

n_H_PMMA = 8
n_C_PMMA = 5
n_O_PMMA = 2

n_PMMA_mon =  rho_PMMA * Na/u_PMMA
n_PMMA_at = n_PMMA_mon * (n_H_PMMA + n_C_PMMA + n_O_PMMA)

m_PMMA_mon = u_PMMA / Na

#CONC_at = {'H': n_H, 'C': n_C, 'O': n_O, 'Si': n_Si}
#CONC = [n_PMMA_at, n_PMMA_at, n_PMMA_at, n_Si]


#%%
uint16_max = 65535


#%%
EE = np.logspace(0, 4.4, 1000)

THETA_deg = np.linspace(0.1, 180, 1000)
THETA = np.deg2rad(THETA_deg)

WW = np.logspace(0, 4.4, 3000)
WW_ext = np.logspace(-4, 4.4, 5000)


#%% PMMA and Si
binding_C_1S = 296
binding_O_1S = 538

occupancy_1S = 2

##              1s    2s   2p   3s     3p
binding_Si   = [1844, 154, 104, 13.46, 8.15]
occupancy_Si = [2,    2,   6,   2,     2]

n_val_PMMA = 40
n_val_Si = 4


#%%
E_cut = 5

#%%
TT_len = np.int(1e+6)
#DATA_len = np.int(1e+5)

