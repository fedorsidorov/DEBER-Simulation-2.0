#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib

from itertools import product

import my_constants as mc
import my_utilities as mu
import MC_functions as mcf
import chain_functions as cf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
mcf = importlib.reload(mcf)
cf = importlib.reload(cf)

#import time

os.chdir(mc.sim_folder + 'PMMA_sim')


#%%
source_dir = '/Volumes/ELEMENTS/Chains_Harris_fit/'

#print(os.listdir(source_dir))


#%% constants
#N_chains_total = 6111
N_chains_total = 1606
N_mon_cell_max = 700

l_xyz = np.array((100, 100, 500))

x_min, y_min, z_min = (-l_xyz[0]/2, -l_xyz[0]/2, 0)
xyz_min = np.array((x_min, y_min, z_min))
xyz_max = xyz_min + l_xyz
x_max, y_max, z_max = xyz_max

step_2nm = 2

x_bins_2nm = np.arange(x_min, x_max + 1, step_2nm)
y_bins_2nm = np.arange(y_min, y_max + 1, step_2nm)
z_bins_2nm = np.arange(z_min, z_max + 1, step_2nm)

bins_2nm = x_bins_2nm, y_bins_2nm, z_bins_2nm

x_grid_2nm = (x_bins_2nm[:-1] + x_bins_2nm[1:]) / 2
y_grid_2nm = (y_bins_2nm[:-1] + y_bins_2nm[1:]) / 2
z_grid_2nm = (z_bins_2nm[:-1] + z_bins_2nm[1:]) / 2

resist_shape = len(x_grid_2nm), len(y_grid_2nm), len(z_grid_2nm)

xs = len(x_grid_2nm)
ys = len(y_grid_2nm)
zs = len(z_grid_2nm)


#%%
pos_matrix = np.zeros(resist_shape, dtype=np.uint32)

resist_matrix = -np.ones((*resist_shape, N_mon_cell_max, 3), dtype=np.uint32)

chain_tables = []

uint32_max = 4294967295


#%%
for chain_num in range(N_chains_total):
    
    mu.pbar(chain_num, N_chains_total)
    
    now_chain = np.load(source_dir + 'chain_shift_' + str(chain_num) + '.npy')
    
    chain_table = np.zeros((len(now_chain), 5), dtype=np.uint32)
    
    for n_mon, mon_line in enumerate(now_chain):
        
        if n_mon == 0:
            mon_type = 0
        elif n_mon == len(now_chain) - 1:
            mon_type = 2
        else:
            mon_type = 1
        
        now_x, now_y, now_z = mon_line
        
        xi = mcf.get_closest_el_ind(x_grid_2nm, now_x)
        yi = mcf.get_closest_el_ind(y_grid_2nm, now_y)
        zi = mcf.get_closest_el_ind(z_grid_2nm, now_z)
        
        mon_line_pos = pos_matrix[xi, yi, zi]
        
        resist_matrix[xi, yi, zi, mon_line_pos] = chain_num, n_mon, mon_type
        chain_table[n_mon] = xi, yi, zi, mon_line_pos, mon_type
        
        pos_matrix[xi, yi, zi] += 1
    
    
    chain_tables.append(chain_table)


#%%
print('resist_matrix size, Gb:', resist_matrix.nbytes / 1024**3)
np.save('MATRIX_resist_Harris_fit.npy', resist_matrix)

        
#%%        
#print('resist_matrix size, Gb:', resist_matrix.nbytes / 1024**3)
#np.save('/Volumes/ELEMENTS/Chains_Harris/MATRIX_resist_Harris.npy', resist_matrix)

#dest_folder = '/Volumes/ELEMENTS/Harris_resist_matrix/'
#
#max_end_ind = 0
#
#for xi, yi, zi in product(range(xs), range(ys), range(zs)):
#    
#    mu.pbar(xi, xs)
#    
#    now_arr = resist_matrix[xi, yi, zi]
#    
#    end_ind = np.where(now_arr[:-1]==uint32_max)[0][0]
#    
#    if end_ind > max_end_ind:
#        max_end_ind = end_ind
    
#    now_arr = now_arr[:end_ind]
    
#    name = 'resist_matrix_' + str(xi) + '_' + str(yi) + '_' + str(zi) + '.npy'
    
#    np.save(dest_folder + name, now_arr)


#%%
dest_folder = '/Volumes/ELEMENTS/Harris_chain_tables_fit/'

for i, ct in enumerate(chain_tables):
    
    mu.pbar(i, len(chain_tables))
    
    np.save(dest_folder + 'chain_table_' + str(i) + '.npy', ct)


#%%
for i, chain in enumerate(chain_tables):
    
    mu.pbar(i, len(chain_tables))
    
    for j, line in enumerate(chain):
        
        x, y, z, pos, mon_t = line.astype(int)
        
        mat_cn, n_mon, mat_type = resist_matrix[x, y, z, pos]
        
        if mat_cn != i or n_mon != j or mat_type != mon_t:
            print('ERROR!', i, j)
            print('chain_num:', mat_cn, i)
            print('n_mon', n_mon, j)
            print('mon_type', mon_t, mat_type)

