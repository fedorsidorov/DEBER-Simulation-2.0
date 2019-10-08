
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
source_dir = '/Volumes/ELEMENTS/Chains_EXP_2um_NEW2/'

#print(os.listdir(source_dir))


#%% constants
N_chains_total = 128330

N_mon_cell_max = 700 ## 689


#%% prepare histograms
l_xyz = np.array((2000, 10, 900))

x_min, y_min, z_min = -l_xyz[0]/2, -l_xyz[1]/2, 0
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

uint16_max = 65535
uint32_max = 4294967295


#%%
dest_folder = '/Volumes/ELEMENTS/Chain_tables_EXP_2um_NEW2/'


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
        
        if mon_line_pos > uint16_max - 10:
            print('mon_line_pos over uint16_max !!!')
        
        resist_matrix[xi, yi, zi, mon_line_pos] = chain_num, n_mon, mon_type
        chain_table[n_mon] = xi, yi, zi, mon_line_pos, mon_type
        
        pos_matrix[xi, yi, zi] += 1
    
    
    np.save(dest_folder + 'chain_table_' + str(chain_num) + '.npy', chain_table)


print('resist_matrix size, Gb:', resist_matrix.nbytes / 1024**3)
#np.save('/Volumes/ELEMENTS/MATRIX_resist_EXP_10nm.npy', resist_matrix)


##%%
#ans = resist_matrix[1, 1, 1]


##%%
np.save('MATRIX_resist_EXP_2um_NEW.npy', resist_matrix)




#%%
rm = np.load('MATRIX_resist_EXP_2um.npy')


#%%
rm = resist_matrix

dest_folder = '/Volumes/ELEMENTS/MATRIX_resist_2um_NEW2/'


for i in range(rm.shape[1]):
    
    mu.pbar(i, rm.shape[1])
    
    np.save(dest_folder + 'MATRIX_resist_' + str(i) + '_2um.npy', rm[:, i, :, :, :])


#%%
full_mat = np.zeros(resist_shape)
mono_mat = np.zeros(resist_shape)


for xi, yi, zi in product(range(resist_shape[0]), range(resist_shape[1]), range(resist_shape[2])):
    
    if yi == zi == 0:
        mu.pbar(xi, resist_shape[0])
    
    full_mat[xi, yi, zi] = len(np.where(resist_matrix[xi, yi, zi, :, 0] != uint32_max)[0])


#%%
ans = full_mat[:, 0, :]

plt.imshow(ans.transpose())


#%%
bns = np.load('/Volumes/ELEMENTS/MATRIX_resist_Harris.npy')


#%%
h_mat = np.zeros((50, 50, 250))

for xi, yi, zi in product(range(50), range(50), range(250)):
    
    if yi == zi == 0:
        mu.pbar(xi, 50)
    
    h_mat[xi, yi, zi] = len(np.where(bns[xi, yi, zi, :, 0] != uint32_max)[0])


#%%
cns = h_mat[:, 0, :]

plt.imshow(cns.transpose())






