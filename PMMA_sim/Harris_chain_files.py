#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib

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
source_dir = '/Volumes/ELEMENTS/Chains_Harris_period_500nm_offset/'

#print(os.listdir(source_dir))


#%%
lens = []

files = os.listdir(source_dir)

for file in files:
    
    if 'chain' not in file:
        continue
    
    chain = np.load(source_dir + file)
    
    lens.append(len(chain))


chain_lens = np.array(lens)


#%%
print(np.max(chain_lens))


#%%
hist_2nm = np.load(source_dir + 'hist_2nm.npy')

print('n_mon_max =', np.max(hist_2nm))


#%% constants
N_chains_total = 6236
N_mon_cell_max = 550

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


#%%
pos_matrix = np.zeros(resist_shape, dtype=np.uint16)
resist_matrix = - np.ones((*resist_shape, N_mon_cell_max, 3), dtype=np.uint16)

chain_table = []


#%%
for chain_num in range(N_chains_total):
    
    mu.upd_progress_bar(chain_num, N_chains_total)
    
    now_chain = np.load(source_dir + 'chain_shift_' + str(chain_num) + '.npy')
    
    now_chain_table = np.zeros((len(now_chain), 5))
    
    for n_mon, mon_line in enumerate(now_chain):
        
        if n_mon == 0:
            mon_type = 0
        elif n_mon == len(now_chain) - 1:
            mon_type = 2
        else:
            mon_type = 1
        
#        if not (np.all(mon_line >= xyz_beg) and np.all(mon_line <= xyz_end)):
#            chain_table[chain_num, n_mon, -1] = mon_type
#            continue
        
        now_x, now_y, now_z = mon_line
        
        x_ind = mcf.get_closest_el_ind(x_grid_2nm, now_x)
        y_ind = mcf.get_closest_el_ind(y_grid_2nm, now_y)
        z_ind = mcf.get_closest_el_ind(z_grid_2nm, now_z)
        
        mon_line_pos = pos_matrix[x_ind, y_ind, z_ind]
        
        resist_matrix[x_ind, y_ind, z_ind, mon_line_pos] = chain_num, n_mon, mon_type
        
        now_chain_table[n_mon] = x_ind, y_ind, z_ind, mon_line_pos, mon_type
        
        pos_matrix[x_ind, y_ind, z_ind] += 1
    
    chain_table.append(now_chain_table)
        
        
#%%        
print('resist_matrix size, Gb:', resist_matrix.nbytes / 1024**3)
np.save('MATRIX_resist_Harris.npy', resist_matrix)


#%%
dest_folder = '/Volumes/ELEMENTS/Harris_chain_tables/'

for i, ct in enumerate(chain_table):
    
    mu.upd_progress_bar(i, len(chain_table))
    
    np.save(dest_folder + 'chain_table_' + str(i) + '.npy', ct)

