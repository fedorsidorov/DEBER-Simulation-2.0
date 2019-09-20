#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
#import copy
from itertools import product

import my_functions as mf
import my_variables as mv
import my_indexes as mi
import my_constants as mc

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mi = importlib.reload(mi)
mc = importlib.reload(mc)

os.chdir(mv.sim_path_MAC + 'mapping')

import my_mapping as mm
mm = importlib.reload(mm)

#%%
e_matrix  = np.load(mv.sim_path_MAC + 'MATRIXES/Aktary/MATRIX_Aktary_31files_C_ion.npy')
dE_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/Aktary/MATRIX_dE_Aktary_31files.npy')

resist_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/Aktary/MATRIX_resist_Aktary.npy')
chain_table   = np.load(mv.sim_path_MAC + 'MATRIXES/Aktary/TABLE_chains_Aktary.npy')

scission_matrix = np.zeros(np.shape(e_matrix))

N_chains_total = 9331
N_mon_chain_max = 6590

sci_per_mol_matrix = np.zeros(N_chains_total)

resist_shape = np.shape(resist_matrix)[:3]

#%% check N max
lens_list = []

for now_chain in chain_table:
    
    now_len = len(np.where(now_chain[:, -1] != mm.uint16_max)[0])
    
    lens_list.append(now_len)

lens_array = np.array(lens_list)

#%%
monomer_matrix = np.zeros(np.shape(e_matrix))

for x_ind, y_ind, z_ind in product(range(resist_shape[0]),\
           range(resist_shape[1]), range(resist_shape[2])):
    
    monomer_matrix[x_ind, y_ind, z_ind] =\
        len(np.where(resist_matrix[x_ind, y_ind, z_ind, :, 0]!=mm.uint16_max)[0])

#%%
#chain_sum_len_matrix_before, n_chains_matrix_before =\
#    mm.get_local_chain_len(resist_shape, N_mon_chain_max, chain_table, N_chains_total)
#
#np.save('Aktary/chain_sum_len_matrix_before.npy', chain_sum_len_matrix_before)
#np.save('Aktary/n_chains_matrix_before.npy', n_chains_matrix_before)

#%%
p_scission = 0.4
n_scissions = 0

n_events_total = 0

#%%
for x_ind, y_ind, z_ind in product(range(resist_shape[0]),\
           range(resist_shape[1]), range(resist_shape[2])):
    
    if y_ind == z_ind == 0:
        mf.upd_progress_bar(x_ind, resist_shape[0])
    
    n_events = mm.get_n_events(e_matrix, x_ind, y_ind, z_ind)
    
    n_events_total += n_events
    
    for i in range(n_events):
        
        if mf.random() >= p_scission:
            continue
        
        resist_part_ind = mm.get_resist_part_ind(resist_matrix, x_ind, y_ind, z_ind)
        
        if resist_part_ind == -1:
            continue
        
        n_chain, n_mon, mon_type = mm.get_resist_part_line(resist_matrix,\
                                            x_ind, y_ind, z_ind, resist_part_ind)
        
        sci_per_mol_matrix[n_chain] += 1
        
############################################################################### 
        if mon_type == mm.mid_mon: ## bonded monomer ##########################
###############################################################################
            
            new_mon_type = mm.get_mon_type()
            
            mm.rewrite_mon_type(resist_matrix, chain_table,\
                                n_chain, n_mon, new_mon_type)
            
            n_next_mon = n_mon + new_mon_type - 1
            
            next_x_ind, next_y_ind, next_z_ind, _, next_mon_type =\
                mm.get_chain_table_line(chain_table, n_chain, n_next_mon)
            
            ## if next monomer was at the end
            if next_mon_type in [mm.beg_mon, mm.end_mon]:
                mm.rewrite_mon_type(resist_matrix, chain_table,\
                                 n_chain, n_next_mon, mm.free_mon)
            
            ## if next monomer is full bonded
            elif next_mon_type == mm.mid_mon:
                next_mon_new_type = next_mon_type - (new_mon_type - 1)
                mm.rewrite_mon_type(resist_matrix, chain_table,\
                                 n_chain, n_next_mon, next_mon_new_type)
            
            else:
                print('error 1, next_mon_type =', next_mon_type)
                print(x_ind, y_ind, z_ind)
            
            n_scissions += 1
            scission_matrix[x_ind, y_ind, z_ind] += 1

###############################################################################
        elif mon_type in [mm.beg_mon, mm.end_mon]: ## half-bonded monomer #####
###############################################################################
            
            new_mon_type = mm.free_mon
            
            mm.rewrite_mon_type(resist_matrix, chain_table,\
                                n_chain, n_mon, new_mon_type)
            
            n_next_mon = n_mon - (mon_type - 1) ## minus, Karl!
            
            next_x_ind, next_y_ind, next_z_ind, _, next_mon_type =\
                mm.get_chain_table_line(chain_table, n_chain, n_next_mon)
            
            ## if next monomer was at the end
            if next_mon_type in [mm.beg_mon, mm.end_mon]:
                mm.rewrite_mon_type(resist_matrix, chain_table,\
                                 n_chain, n_next_mon, mm.free_mon)
            
            ## if next monomer is full bonded
            elif next_mon_type == mm.mid_mon:
                next_mon_new_type = next_mon_type + (mon_type - 1)
                mm.rewrite_mon_type(resist_matrix, chain_table,\
                                 n_chain, n_next_mon, next_mon_new_type)
                
            else:
                print('error 2', next_mon_type)
            
            n_scissions += 1
            scission_matrix[x_ind, y_ind, z_ind] += 1
            
###############################################################################
        elif mon_type == mm.free_mon: ## free monomer with ester group ########
###############################################################################
            
            ## only ester group deatachment is possible
            mm.rewrite_mon_type(resist_matrix, chain_table,\
                             n_chain, n_mon, mm.free_rad_mon)
            
            n_scissions += 1
            scission_matrix[x_ind, y_ind, z_ind] += 1
        
        elif mon_type == mm.free_rad_mon:
            continue
        
        else:
            print('WTF', mon_type)

#%% G-value
E_dep = np.sum(dE_matrix)

G_value = n_scissions / (E_dep / 100)
#G_value = n_events_total / (E_dep / 100)

print(G_value)

#%%
chain_sum_len_matrix_2C_ion, n_chains_matrix_2C_ion =\
    mm.get_local_chain_len(resist_shape, N_mon_chain_max, chain_table, N_chains_total)

#%%
np.save('Aktary/chain_sum_len_matrix_2C_ion.npy', chain_sum_len_matrix_2C_ion)
np.save('Aktary/n_chains_matrix_2C_ion.npy', n_chains_matrix_2C_ion)
np.save('Aktary/scission_matrix.npy', scission_matrix)

#%%
#monomer_2D = np.sum(monomer_matrix, axis=1) / 50
monomer_2D = np.ones((50, 50)) * 57

scission_2D = np.sum(e_matrix, axis=1) / 50 * 0.4

sci_prob_2D = scission_2D / monomer_2D

mean_n_2D = 9500 / (1 + 9500 * sci_prob_2D)

Cn_2D = np.zeros((50, 50))

for x_ind, z_ind in product(range(resist_shape[0]), range(resist_shape[2])):
    
    n_avg = mean_n_2D[x_ind, z_ind]
    
    for i in range(1, 11):
        Cn_2D[x_ind, z_ind] += (n_avg - 1)**(i-1) / (np.math.factorial(i-1))\
            * np.exp(-n_avg + 1)

#%%
#scission_matrix = np.load('Aktary/scission_matrix.npy')

sci_xz = (np.sum(scission_matrix, axis=1)).transpose()

sci_xz_int = np.array(sci_xz, dtype=np.uint8)

np.savetxt('2500_pC_cm.txt', sci_xz_int, fmt='%d', delimiter=',')

#%%
plt.imshow(sci_xz)
#plt.xticks((1, 2, 3))

plt.colorbar()

ax = plt.gca()
ax.set_xticklabels((0, 0, 20, 40, 60, 80))
ax.set_yticklabels((0, 0, 20, 40, 60, 80))
plt.title('PMMA chain scission distribution')
plt.xlabel('x, nm')
plt.ylabel('x, nm')

plt.savefig('Sci_distr.png', dpi=300)
