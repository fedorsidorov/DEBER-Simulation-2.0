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


#%%
def rewrite_mon_type(resist_matrix, chain_table, n_mon, new_type):

    chain_table[n_mon, mon_type_ind] = new_type

    xi, yi, zi, mon_line_pos = chain_table[n_mon, :mon_type_ind].astype(int)
    resist_matrix[xi, yi, zi, mon_line_pos, mon_type_ind] = new_type


def inc_mon_type(resist_matrix, chain_table, n_mon):
    
    now_type = chain_table[n_mon, mon_type_ind]
    new_type = now_type + 1
    
    chain_table[n_mon, mon_type_ind] = new_type

    xi, yi, zi, mon_line_pos = chain_table[n_mon, :mon_type_ind].astype(int)
    resist_matrix[xi, yi, zi, mon_line_pos, mon_type_ind] = new_type


def move_events(e_matrix, xi, yi, zi, n_events):
    
    e_matrix[xi, yi, zi] = 0
    
    if zi+1 < resist_shape[2]:
        e_matrix[xi, yi, zi+1] += n_events
        
    elif xi+1 < resist_shape[0]:
        e_matrix[xi+1, yi, zi] += n_events

    elif yi+1 < resist_shape[1]:
        e_matrix[xi, yi+1, zi] += n_events
    
    elif xi+1 < resist_shape[0] and zi+1 < resist_shape[2]:
        e_matrix[xi+1, yi, zi+1] += n_events
    
    elif yi+1 < resist_shape[1] and zi+1 < resist_shape[2]:
        e_matrix[xi, yi+1, zi+1] += n_events
    
    elif xi+1 < resist_shape[0] and yi+1 < resist_shape[1]:
        e_matrix[xi+1, yi+1, zi] += n_events


#%%
e_matrix = np.load(mc.sim_folder + 'e-matrix_EXP/EXP_2um_ones/EXP_e_matrix_val_MY_dose1.npy')


#%%
resist_matrix = np.load('MATRIX_resist_EXP_2um_0.npy')

chain_tables_folder = 'Chain_tables_EXP_2um_0/'
files = os.listdir(chain_tables_folder)

N_chains_total = 128330
chain_tables = [None] * N_chains_total


for i in range(N_chains_total):
    
    mu.pbar(i, len(files))
    now_chain_table = np.load(chain_tables_folder + 'chain_table_' + str(i) + '.npy')
    chain_tables[i] = now_chain_table


#%%
zip_lens = np.zeros(int(np.sum(e_matrix)))
unzip_lens = np.zeros(int(np.sum(e_matrix)))
unzip_cells = np.zeros((int(np.sum(e_matrix)), 3))


resist_shape = np.shape(resist_matrix)[:3]

zip_len = 1000
go = True

pos = -1


for xi, yi, zi in product(range(resist_shape[0]),\
                          range(resist_shape[1]),\
                          range(resist_shape[2])):
    
    if not go:
        break
    
    if yi == zi == 0:
        mu.pbar(xi, resist_shape[0])
    
    n_events = e_matrix[xi, yi, zi]
    
    if n_events == 0:
        continue
    
    
    while n_events > 0:
        
        monomer_positions =\
            np.where(resist_matrix[xi, yi, zi, :, mon_type_ind] - 1e+6 < 0)[0]
        
        ## if there are no bonded monomers, move events further
        if len(monomer_positions) == 0:
            move_events(e_matrix, xi, yi, zi, n_events)
            break
        
        pos += 1
        
        monomer_pos = np.random.choice(monomer_positions)
        
        n_chain, n_mon, _ = resist_matrix[xi, yi, zi, monomer_pos, :].astype(int)
        
        chain_table = chain_tables[n_chain]
        now_len = len(chain_table)
        
        if len(chain_table) == 1:
            print('chain len = 1!')
            print(n_chain)
            print(chain_table[0])
#            go = False
        
        if not go:
            break
        
        now_zipped = 0
        side = np.random.choice([-1, 1])

        
        while now_zipped < zip_len:
            
            inc_mon_type(resist_matrix, chain_table, n_mon)
            
            now_zipped += 1
            
            zip_lens[pos] += 1
            
            n_next_mon = n_mon + side
            
            ## if chain is not ended, zip this chain further
            if n_next_mon >= 0 and n_next_mon < now_len-1:
                    n_mon = n_next_mon
            
            else:
                now_xi, now_yi, now_zi, _ = chain_table[n_mon, :mon_type_ind].astype(int)
                
                now_monomer_positions = np.where(resist_matrix[now_xi,\
                            now_yi, now_zi, :, mon_type_ind] - 1e+6 < 0)[0]
                
                cnt = 0
                
                
                while len(now_monomer_positions) == 0:
                    
                    if cnt > 100:
                        unzip_lens[pos] += 1000 - now_zipped
                        unzip_cells[pos] = np.array((now_xi, now_yi, now_zi))
                        break
                    
                    ## chooice one of neighbours
                    shifts = np.random.choice([-1, 0, 1], size=3,\
                                              replace=True)
                    new_coords = np.array([now_xi, now_yi, now_zi]) + shifts
                    
                    if np.all(new_coords >= 0) and np.all(new_coords + 1 < resist_shape):
                        now_xi, now_yi, now_zi = new_coords
                        now_monomer_positions =\
                            np.where(resist_matrix[now_xi, now_yi, now_zi, :, mon_type_ind]\
                                     - 3 < 0)[0]
                    
                    cnt += 1
                
                if len(now_monomer_positions) == 0:
                    break
                
                now_monomer_pos = np.random.choice(now_monomer_positions)
                
                n_chain, n_mon, _ = resist_matrix[now_xi, now_yi, now_zi, now_monomer_pos,\
                                                  :].astype(int)
                
                chain_table = chain_tables[n_chain]
                now_len = len(chain_table)
                
                side = np.random.choice([-1, 1])
        
        
        n_events -= 1


#%%
full_mat = np.zeros(np.shape(e_matrix))
mono_mat = np.zeros(np.shape(e_matrix))


for xi, yi, zi in product(range(resist_shape[0]),\
                          range(resist_shape[1]),\
                          range(resist_shape[2])):
    
    if yi == zi == 0:
        mu.pbar(xi, resist_shape[0])
    
    full_mat[xi, yi, zi] = len(np.where(resist_matrix[xi, yi, zi, :, 0] != uint32_max)[0])
    mono_mat[xi, yi, zi] = len(np.where(resist_matrix[xi, yi, zi, :,\
            mon_type_ind] == free_mon)[0])


#%%
#np.save('2um_CT/full_mat_dose3.npy', full_mat)
#np.save('2um_CT/mono_mat_dose3.npy', mono_mat)
