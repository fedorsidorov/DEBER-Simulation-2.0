#%% Import
import numpy as np

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

#%%
#sim_path_MAC = '/Users/fedor/Documents/DEBER-Simulation/'
##sim_path_FTIAN = '/home/fedor/Yandex.Disk/Study/Simulation/'
#
##%% Sharma cube parameters
#cube_size = 10.
#cell_size = 2.
#
#eps = 1e-3
#
### min and max coordinates
#xyz_min = np.array((0., 0., 0.))
#xyz_max = np.array((100., 100., 160.))
#xyz_max_new = np.array((100., 100., 480.))
#
### cubes parameters
#n_X, n_Y, n_Z = (np.round((xyz_max - xyz_min) / cube_size)).astype(int)
#_, _, n_Z_new = (np.round((xyz_max_new - xyz_min) / cube_size)).astype(int)
#n_XY = n_X * n_Y
#n_x = n_y = n_z = int(cube_size / cell_size)
#
### chains
#n_chains = 12709
#n_chains_short = 12643
#n_mon_max = 325
#n_part_max = 600
#chain_len_max = 8294
#chain_len_max_short = 6000
#
### process indexes
#proc_indexes = 0, 1, 2
#sci_ester, sci_direct, ester = proc_indexes
#ester_CO, ester_CO2 = 0, 1
