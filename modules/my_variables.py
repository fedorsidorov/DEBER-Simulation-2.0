import numpy as np

#%% Variables
e = 4.8e-10
m = 9.11e-28
m_eV = 511e+3
h = 6.63e-27
eV = 1.6e-12
E0 = 20e+3
Na = 6.02e+23

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
u_Si = 28.08
rho_Si = 2.33
n_Si =  rho_Si*Na/u_Si

u_PMMA = 100
rho_PMMA = 1.18
n_PMMA =  rho_PMMA*Na/u_PMMA
n_PMMA_at = n_PMMA*(5 + 2 + 8)

CONC_at = {'H': n_H, 'C': n_C, 'O': n_O, 'Si': n_Si}
CONC = [n_PMMA_at, n_PMMA_at, n_PMMA_at, n_Si]

m_PMMA_mon = u_PMMA / Na

#%%
tr_num, par_num = 0, 1
atom_id, coll_id = 2, 3
e_E, e_dE = 4, 8
e_x, e_y, e_z = 5, 6, 7

elastic, exc = 0, 1
ion_K = 2
ion_L1, ion_L2, ion_L3 = 3, 4, 5
ion_M1, ion_M2, ion_M3 = 6, 7, 8

H = 0
C = 1
O = 2
Si = 3

## chain matrix
n_chain = 0
n_mon = 1
mon_type = 2

## chain inv matrix
inv_pos = 1
c_Z, c_XY, c_x, c_y, c_z, n_mon, pos, mon_type = 0, 1, 2, 3, 4, 5, 6, 7

#%%
sim_path_MAC = '/Users/fedor/Documents/DEBER-Simulation/'
#sim_path_FTIAN = '/home/fedor/Yandex.Disk/Study/Simulation/'

#%% Sharma cube parameters
cube_size = 10.
cell_size = 2.

eps = 1e-3

## min and max coordinates
xyz_min = np.array((0., 0., 0.))
xyz_max = np.array((100., 100., 160.))
xyz_max_new = np.array((100., 100., 480.))

## cubes parameters
n_X, n_Y, n_Z = (np.round((xyz_max - xyz_min) / cube_size)).astype(int)
_, _, n_Z_new = (np.round((xyz_max_new - xyz_min) / cube_size)).astype(int)
n_XY = n_X * n_Y
n_x = n_y = n_z = int(cube_size / cell_size)

## chains
n_chains = 12709
n_chains_short = 12643
n_mon_max = 325
n_part_max = 600
chain_len_max = 8294
chain_len_max_short = 6000

## process indexes
proc_indexes = 0, 1, 2
sci_ester, sci_direct, ester = proc_indexes
ester_CO, ester_CO2 = 0, 1
