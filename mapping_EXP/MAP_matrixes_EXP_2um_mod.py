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
e_matrix = np.load(mc.sim_folder + 'e-matrix_EXP/EXP_2um_ones/EXP_e_matrix_val_MY_dose1.npy')


#%%
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
        
        monomer_positions = np.where(resist_matrix[xi, yi, zi, :, n_chain_ind] - 3 < 0)[0]
        
        if len(monomer_positions) == 0:
            
            if zi+1 < resist_shape[2]:
                e_matrix[xi, yi, zi+1] += n_events
                break
            
            elif xi+1 < resist_shape[0]:
                e_matrix[xi+1, yi, zi] += n_events
                break
            
            elif yi+1 < resist_shape[1]:
                e_matrix[xi, yi+1, zi] += n_events
                break
            
            else:
                print('no space for extra events')
                break
        
        monomer_pos = np.random.choice(monomer_positions)
        
        n_chain, n_mon, mon_type = resist_matrix[xi, yi, zi, monomer_pos, :].astype(int)
        
        chain_table = chain_tables[n_chain]
        
        now_len = len(chain_table)
        
        
        if mon_type != chain_table[n_mon, -1]:
            print('FUKKK!!', n_chain, n_mon)
            print(mon_type, chain_table[n_mon, -1])
        
        
        if len(chain_table) == 1:
            continue
        
        
        side = np.random.choice([-1, 1])
        
        if side == -1:
                
            for j in reversed(range(i-1000, i)):
                
                if chain_table[j, mon_type_ind] == free_mon or j == 0:
                    break
                
                rewrite_mon_type(resist_matrix, chain_table, j, free_mon)
            
        else:
                
            for j in range(i+1, i+1001):
                
                if chain_table[j, mon_type_ind] == free_mon or j == now_len-1:
                    break
                
                rewrite_mon_type(resist_matrix, chain_table, j, free_mon)
        
        
        continue


#%%
full_mat = np.zeros(np.shape(e_matrix))
mono_mat = np.zeros(np.shape(e_matrix))


for xi, yi, zi in product(range(resist_shape[0]), range(resist_shape[1]), range(resist_shape[2])):
    
    if yi == zi == 0:
        mu.pbar(xi, resist_shape[0])
    
    full_mat[xi, yi, zi] = len(np.where(resist_matrix[xi, yi, zi, :, 0] != uint32_max)[0])
    mono_mat[xi, yi, zi] = len(np.where(resist_matrix[xi, yi, zi, :, mon_type_ind] == free_mon)[0])


#%%
for i, ct in enumerate(chain_tables):
    
    if len(np.where(ct[:, -1] == 10)[0]) != 0:
        print(i)


#%%
np.save('2um_mod/full_mat_dose1.npy', full_mat)
np.save('2um_mod/mono_mat_dose1.npy', mono_mat)


#%%
#ratio_mat = mon_mat / fullness_mat

#ans = np.average(ratio_mat, axis=1)
