#%% Import
import numpy as np
import os
import importlib

import my_functions as mf
import my_variables as mv
import my_indexes as mi
import my_mapping as mm

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mi = importlib.reload(mi)
mm = importlib.reload(mm)

os.chdir(mv.sim_path_MAC + 'make_chain_arrays')

#%%
source_dir = mv.sim_path_MAC + 'CHAINS/950K_122nm/comb_600x100x122_center/'

#%%
pos_matrix = np.zeros(mm.resist_shape, dtype=np.uint16)
resist_matrix = - np.ones((*mm.resist_shape, mm.N_mon_cell_max, 3), dtype=np.uint16)

chain_table = - np.ones((mm.N_chains_total, mm.N_mon_chain_max, 5), dtype=np.uint16)

#%%
for chain_num in range(mm.N_chains_total):
    
    mf.upd_progress_bar(chain_num, mm.N_chains_total)
    
    now_chain = np.load(source_dir + 'chain_shift_' + str(chain_num) + '.npy')
    
    for n_mon, mon_line in enumerate(now_chain):
        
        if n_mon == 0:
            mon_type = 0
        elif n_mon == len(now_chain) - 1:
            mon_type = 2
        else:
            mon_type = 1
        
        if not (np.all(mon_line >= mm.xyz_beg) and np.all(mon_line <= mm.xyz_end)):
            chain_table[chain_num, n_mon, mi.mon_type] = mon_type
            continue
        
        now_x, now_y, now_z = mon_line
        
        x_ind = mf.get_closest_el_ind(mm.x_grid_2nm, now_x)
        y_ind = mf.get_closest_el_ind(mm.y_grid_2nm, now_y)
        z_ind = mf.get_closest_el_ind(mm.z_grid_2nm, now_z)
        
        mon_line_pos = pos_matrix[x_ind, y_ind, z_ind]
        
        resist_matrix[x_ind, y_ind, z_ind, mon_line_pos] = chain_num, n_mon, mon_type
        
        chain_table[chain_num, n_mon] = x_ind, y_ind, z_ind, mon_line_pos, mon_type
        
        pos_matrix[x_ind, y_ind, z_ind] += 1
        
#%%        
print('resist_matrix size, Gb:', resist_matrix.nbytes / 1024**3)
np.save('MATRIX_resist.npy', resist_matrix)

#%%
print('chain_table size, Gb:', chain_table.nbytes / 1024**3)
np.save('TABLE_chains.npy', chain_table)

#%%
t = chain_table[2]
