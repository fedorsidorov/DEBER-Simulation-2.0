import numpy as np
import os


#%%
sim_folder = os.path.join('/Users', 'fedor', 'Documents', 'DEBER-Simulation-2.0') ## MAC
#sim_folder = os.path.join('home', 'fedor', 'DEBER-Simulation-2.0') ## FTIAN
#sim_folder = os.path.join('C:\\', 'Users', 'User', 'Documents', 'GitHub',
#            'DEBER-Simulation-2.0')

#%% SI!!!
e = 1.602e-19 ## SI !!!
m = 9.1e-31 ## SI !!!
m_eV = 511e+3
h = 6.626e-34 ## SI !!!
h_bar = 1.054e-34 ## SI !!!
eV = 1.602e-19 ## SI !!!
Na = 6.02e+23 
eps_0 = 8.854e-12 ## SI!!!
k_el = 1 / ( 4 * np.pi * eps_0 ) ## SI !!!
c = 3e+8 ## SI!!!

a0 = 5.292e-9 ## cm

k_B = 8.617e-5 ## eV / K

r0 = 2.828e-13 ## cm - classical electron radius

hw_phonon = 0.1 ## eV

h2si = k_el * m * e**2 / h_bar**2 ## Hartree units to SI

a0_2 = 2.8002852e-17 ## cm


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
n_Si = rho_Si * Na/u_Si

N_H_MMA = 8
N_C_MMA = 5
N_O_MMA = 2

u_MMA = (N_H_MMA*u_H + N_C_MMA*u_C + N_O_MMA*u_O)

M0 = u_MMA / Na

rho_PMMA = 1.19


n_MMA = rho_PMMA * Na/u_MMA
#n_PMMA_at = n_PMMA_mon * (n_H_PMMA + n_C_PMMA + n_O_PMMA)

m_MMA = u_MMA / Na

PMMA_val_kF = ( 3 * np.pi**2 * n_MMA * 40 )**(1/3) * 1e+2 ## cm^-1 -> m^-1
PMMA_val_pF = h_bar * PMMA_val_kF ## kg * m/s
PMMA_val_vF = PMMA_val_pF / m ## m/s



#%%
uint16_max = 65535


#%%
EE = np.logspace(0, 4.4, 1000)
EE_10 = np.logspace(1, 4.4, 1000)

THETA_deg = np.linspace(0, 180, 1000)
THETA_rad = np.deg2rad(THETA_deg)

EE_prec = np.logspace(-1, 4.4, 1000)

#WW = np.logspace(0, 4.4, 3000)
#WW_ext = np.logspace(-4, 4.4, 5000)


#%% PMMA and Si
binding_C_1S = 296
binding_O_1S = 538

occupancy_1S = 2

##               plasm    3p     3s      2p      2s      1s
Si_MuElec_Eb  = [16.65, 6.52, 13.63, 107.98, 151.55, 1828.5]
Si_MuElec_occ = [    4,    2,     2,      6,      2,      2]

n_val_PMMA = 40
n_val_Si = 4


#%%
PMMA_Ebind = [25.84, 3.6, 288, 543]


#%%
TT_len = int(7e+3)

# DATA_tr_len = int(3e+4)
DATA_tr_len = int(3e+5)

Wf_PMMA = 4.68

E_cut_PMMA = 3.7
E_cut_Si = 16.7
# E_cut_Si = 20

