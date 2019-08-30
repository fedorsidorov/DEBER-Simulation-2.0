#%% Import
import numpy as np

#%%
tr_num, par_num = 0, 1
layer_ind, coll_id = 2, 3
E_ind, dE_ind = 4, 8
x_ind, y_ind, z_ind = 5, 6, 7

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
