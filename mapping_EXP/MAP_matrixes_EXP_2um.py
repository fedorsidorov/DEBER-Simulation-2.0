#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
#import copy
from itertools import product

import my_utilities as mu
mu = importlib.reload(mu)

import my_constants as mc
mc = importlib.reload(mc)

os.chdir(mc.sim_folder + 'mapping_EXP')


#%%
beg_mon, mid_mon, end_mon = 0, 1, 2
free_mon = 10

n_chain_ind = 0
mon_line_ind = 3
mon_type_ind = -1
uint32_max = 4294967295


def rewrite_mon_type(resist_matrix, chain_table, n_mon, new_type):

    chain_table[n_mon, mon_type_ind] = new_type

    xi, yi, zi, mon_line_pos = chain_table[n_mon, :mon_type_ind].astype(int)
    resist_matrix[xi, yi, zi, mon_line_pos, mon_type_ind] = new_type


#%%
e_matrix = np.load(mc.sim_folder + 'e-matrix_EXP/EXP_2um/EXP_e_matrix_val_MY_dose2.npy')

resist_matrix = np.zeros((1000, 5, 450, 700, 3))


for i in range(resist_matrix.shape[1]):
    
    mu.pbar(i, resist_matrix.shape[1])
    
    resist_matrix[:, i, :, :, :] = np.load('/Volumes/ELEMENTS/MATRIX_resist_2um_NEW2/MATRIX_resist_' +\
                 str(i) + '_2um.npy')


#%%
chain_tables_folder = '/Volumes/ELEMENTS/Chain_tables_EXP_2um_NEW2/'
files = os.listdir(chain_tables_folder)

chain_tables = []

N_chains_total = 128330


for i in range(N_chains_total):
    
    mu.pbar(i, len(files))
    
    chain_tables.append(np.load(chain_tables_folder + 'chain_table_' + str(i) + '.npy'))


N_chains_total = len(chain_tables)

resist_shape = np.shape(resist_matrix)[:3]


#%%
for xi, yi, zi in product(range(resist_shape[0]), range(resist_shape[1]), range(resist_shape[2])):
    
    if yi == zi == 0:
        mu.pbar(xi, resist_shape[0])
    
    n_events = e_matrix[xi, yi, zi]
    
    if n_events == 0:
        continue
    
    for i in range(int(n_events)):
        
        monomer_positions = np.where(resist_matrix[xi, yi, zi, :, n_chain_ind] != uint32_max)[0]
        
        if len(monomer_positions) == 0:
            
            if xi+1 < resist_shape[0]:
                e_matrix[xi+1, yi, zi] += n_events
                break
            
            elif yi+1 < resist_shape[1]:
                e_matrix[xi, yi+1, zi] += n_events
                break
            
            elif zi+1 < resist_shape[2]:
                e_matrix[xi, yi, zi+1] += n_events
                break
            
            else:
                print('no space for extra events')
                break
        
        monomer_pos = np.random.choice(monomer_positions)
        
        n_chain, n_mon, mon_type = resist_matrix[xi, yi, zi, monomer_pos, :].astype(int)
        
        chain_table = chain_tables[n_chain]
        
        
        if mon_type != chain_table[n_mon, -1]:
            print('FUKKK!!', n_chain, n_mon)
            print(mon_type, chain_table[n_mon, -1])
        
        
        if len(chain_table) == 1:
            continue
        
        
############################################################################### 
        if mon_type == mid_mon: ## bonded monomer #############################
###############################################################################
            
            ## choose between left and right bond
            new_mon_type = np.random.choice([0, 2])
                        
            rewrite_mon_type(resist_matrix, chain_table, n_mon, new_mon_type)
            
            n_next_mon = n_mon + new_mon_type - 1
            
            next_xi, next_yi, next_zi, _, next_mon_type = chain_table[n_next_mon]
            
            ## if next monomer was at the end
            if next_mon_type in [beg_mon, end_mon]:
                
                next_mon_new_type = free_mon
                
                rewrite_mon_type(resist_matrix, chain_table, n_next_mon, next_mon_new_type)
                
            
            ## if next monomer is full bonded
            elif next_mon_type == mid_mon:
                
                next_mon_new_type = next_mon_type - (new_mon_type - 1)
                
                rewrite_mon_type(resist_matrix, chain_table, n_next_mon, next_mon_new_type)
                
                
            else:
                print('\nerror!')
                print('n_chain', n_chain)
                print('n_mon', n_mon)
                print('next_mon_type', next_mon_type)
                
                
###############################################################################
        elif mon_type in [beg_mon, end_mon]: ## half-bonded monomer ###########
###############################################################################
            
            new_mon_type = free_mon
            
            rewrite_mon_type(resist_matrix, chain_table, n_mon, new_mon_type)
            
            n_next_mon = n_mon - (mon_type - 1) ## minus, Karl!
            
            next_xi, next_yi, next_zi, _, next_mon_type = chain_table[n_next_mon]
            
            ## if next monomer was at the end
            if next_mon_type in [beg_mon, end_mon]:
                
                next_mon_new_type = free_mon
                
                rewrite_mon_type(resist_matrix, chain_table, n_next_mon, next_mon_new_type)
                
            ## if next monomer is full bonded !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            elif next_mon_type == mid_mon:
                
                next_mon_new_type = next_mon_type + (mon_type - 1)
                
                rewrite_mon_type(resist_matrix, chain_table, n_next_mon, next_mon_new_type)
                
            else:
                print('error 2', next_mon_type)
        
        else:
            continue


#%%
for ind, chain_table in enumerate(chain_tables):
    
    mu.pbar(ind, len(chain_tables))
    
    now_len = len(chain_table)
    
    for i in range(now_len):
        
        if i in [0, now_len-1] or chain_table[i, mon_type_ind] == free_mon:
            continue
        
        if i == 1:
            continue
        
        elif chain_table[i, mon_type_ind] == 2:
            
            side = np.random.choice([-1, 1])
            
            if side == -1:
                
                for j in reversed(range(i-1000, i)):
                    
                    if chain_table[j, mon_type_ind] == 0 or j == 0:
                        
                        rewrite_mon_type(resist_matrix, chain_table, j, free_mon)
                        break
                    
                    rewrite_mon_type(resist_matrix, chain_table, j, free_mon)
            
            else:
                
                for j in range(i+1, i+1001):
                    
                    if chain_table[j, mon_type_ind] == 2 or j == now_len-1:
                        
                        rewrite_mon_type(resist_matrix, chain_table, j, free_mon)
                        break
                    
                    rewrite_mon_type(resist_matrix, chain_table, j, free_mon)
            

#%%
full_mat = np.zeros(np.shape(e_matrix))
mono_mat = np.zeros(np.shape(e_matrix))


for xi, yi, zi in product(range(resist_shape[0]), range(resist_shape[1]), range(resist_shape[2])):
    
    if yi == zi == 0:
        mu.pbar(xi, resist_shape[0])
    
    full_mat[xi, yi, zi] = len(np.where(resist_matrix[xi, yi, zi, :, 0] != uint32_max)[0])
    mono_mat[xi, yi, zi] = len(np.where(resist_matrix[xi, yi, zi, :, mon_type_ind] == free_mon)[0])


#%%
#np.sum(full_mat)


#%%
np.save('2um/full_mat_dose2.npy', full_mat)
np.save('2um/mono_mat_dose2.npy', mono_mat)


#%%
#ratio_mat = mon_mat / fullness_mat

#ans = np.average(ratio_mat, axis=1)
