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

os.chdir(mc.sim_folder + 'PMMA_sim_EXP')


#%%
source_dir = '/Volumes/ELEMENTS/Chains_EXP_2um/'

#print(os.listdir(source_dir))


#%% constants
N_chains_total = 128367

N_mon_cell_max = 600 ## 571

#%% prepare histograms
l_xyz = np.array((2000, 10, 900))

x_min, y_min, z_min = -l_xyz[0]/2, -l_xyz[0]/2, 0
xyz_min = np.array((x_min, y_min, z_min))
xyz_max = xyz_min + l_xyz
x_max, y_max, z_max = xyz_max

#step_10nm = 10
step_2nm = 2

bins_total = np.array(np.hstack((xyz_min.reshape(3, 1), xyz_max.reshape(3, 1))))

x_bins_2nm = np.arange(x_min, x_max+1, step_2nm)
y_bins_2nm = np.arange(y_min, y_max+1, step_2nm)
z_bins_2nm = np.arange(z_min, z_max+1, step_2nm)

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

#chain_tables = []

#uint16_max = 65535
uint32_max = 4294967295


#%% create resist matrix
#dest = '/Volumes/ELEMENTS/Resist_EXP_1200nm/'
#dest = 'Resist_EXP_1200nm_1/'
#
#now_resist_cell = np.ones((N_mon_cell_max, 3), dtype=np.uint32)
#
##for xi, yi, zi in product(range(xs), range(ys), range(zs)):
#
#for yi in range(ys):
#
#    for xi, yi, zi in product(range(xs), range(yi, ), range(zs)):
#        
#        if zi == 0:
#            mu.pbar(xi, xs)
#        
#        np.save(dest + 'resist_cell_' + str(xi) + '_' + str(yi) + '_' + str(zi) + '.npy',\
#                now_resist_cell)


#%%
dest_folder = '/Volumes/ELEMENTS/Chain_tables_EXP_2um/'


for chain_num in range(N_chains_total):
    
    mu.pbar(chain_num, N_chains_total)
    
    now_chain = np.load(source_dir + 'chain_shift_' + str(chain_num) + '.npy')    
    
    chain_table = np.zeros((len(now_chain), 5), dtype=np.uint16)
    
    
    for n_mon, mon_line in enumerate(now_chain):
        
        if n_mon > uint32_max - 10:
            print('n_mon over uint32_max !!!')
        
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
        
        if mon_line_pos > uint32_max - 10:
            print('mon_line_pos over uint32_max !!!')
        
        resist_matrix[xi, yi, zi, mon_line_pos] = chain_num, n_mon, mon_type
        chain_table[n_mon] = xi, yi, zi, mon_line_pos, mon_type
        
        pos_matrix[xi, yi, zi] += 1
    
    
    np.save(dest_folder + 'chain_table_' + str(chain_num) + '.npy', chain_table)



print('resist_matrix size, Gb:', resist_matrix.nbytes / 1024**3)
#np.save('/Volumes/ELEMENTS/MATRIX_resist_EXP_10nm.npy', resist_matrix)

##%%
#ans = resist_matrix[1, 1, 1]


##%%
np.save('MATRIX_resist_EXP_2um.npy', resist_matrix)

        
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
#for i, chain in enumerate(chain_tables):
#    
#    mu.pbar(i, len(chain_tables))
#    
#    for j, line in enumerate(chain):
#        
#        x, y, z, pos, mon_t = line.astype(int)
#        
#        mat_cn, n_mon, mat_type = resist_matrix[x, y, z, pos]
#        
#        if mat_cn != i or n_mon != j or mat_type != mon_t:
#            print('ERROR!', i, j)
#            print('chain_num:', mat_cn, i)
#            print('n_mon', n_mon, j)
#            print('mon_type', mon_t, mat_type)


#%%
rm = np.load('MATRIX_resist_EXP_2um.npy')


#%%
dest_folder = '/Volumes/ELEMENTS/MATRIX_resist_2um/'


for i in range(rm.shape[1]):
    
    mu.pbar(i, rm.shape[1])
    
    np.save(dest_folder + 'MATRIX_resist_' + str(i) + '_2um.npy', rm[:, i, :, :, :])


#%%
#rm_test = np.zeros(np.shape(rm))
#
#
#for i in range(rm.shape[1]):
#    
#    mu.pbar(i, rm.shape[1])
#    
#    rm_test[:, i, :, :, :] = np.load(dest_folder + 'MATRIX_resist_' + str(i) + '_1400nm.npy')


#%%







