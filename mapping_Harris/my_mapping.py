#%% Import
import numpy as np
import importlib

import my_utilities as mu
mu = importlib.reload(mu)


#%% resist_matrix
n_chain_pos = 0
beg_mon, mid_mon, end_mon = 0, 1, 2
free_mon, free_rad_mon = 10, 20

#%% chain_table
x_pos, y_pos, z_pos = 0, 1, 2
mon_line_pos_pos = 3
mon_type_pos = -1
uint16_max = 65535


#%%
## changes monomer type
def rewrite_mon_type(resist_matrix, chain_table, n_chain, n_mon, new_type):
    ## change mon_type in chain_inv_matrix
    chain_table[n_chain, n_mon, mon_type_pos] = new_type
    ## define x, y, z of monomer
    x_pos, y_pos, z_pos, mon_line_pos = chain_table[n_chain, n_mon, :mon_type_pos]
    ## if we aren't outside the resist area of interest
    if not x_pos == y_pos == z_pos == mon_line_pos == uint16_max:
        resist_matrix[x_pos, y_pos, z_pos, mon_line_pos, mon_type_pos] = new_type


## choose one of existing particles to interact electron with
def get_resist_part_pos(resist_matrix, x_pos, y_pos, z_pos):
    ## indexes of existing particle lines
    resist_part_poss = np.where(resist_matrix[x_pos, y_pos, z_pos, :, n_chain_pos] !=\
                         uint16_max)[0]
    ## if no free particles
    if len(resist_part_poss) == 0:
            return -1
        
    return np.random.choice(resist_part_poss)


## choose of monomer type
#def get_mon_type():
#    return np.random.choice([0, 2])


## get neede chain_table line
#def get_chain_table_line(chain_table, n_chain, n_mon):
#    return chain_table[n_chain, n_mon]


## get n events in the cell
#def get_n_events(e_matrix, x_pos, y_pos, z_pos):
#    return e_matrix[x_pos, y_pos, z_pos]


## get particle line from 
#def get_resist_part_line(resist_matrix, x_pos, y_pos, z_pos, resist_part_pos):
#    return resist_matrix[x_pos, y_pos, z_pos, resist_part_pos, :]


## convert monomer type to monomer kind
#def mon_type_to_kind(mon_type):
#    ## with ester group
#    if mon_type in [-1, 0, 1]:
#        return mon_type
#    ## W/O ester group
#    else:
#        return mon_type - 10


## convert 65536 to -1
#def correct_mon_type(mon_type):
#    if mon_type == uint16_max:
#        return -1
#    return mon_type


## calculate local AVG chain length distribution
def get_local_chain_len(res_shape, N_mon_max, chain_table, N_chains):
    
    chain_sum_len_matrix = np.zeros(res_shape)
    n_chains_matrix = np.zeros(res_shape)
    
    for idx, chain in enumerate(chain_table):
        
        mu.pbar(idx, N_chains)
        
        beg_pos = 0
        
        while True:
            
            if beg_pos >= N_mon_max or chain[beg_pos, mon_type_pos] == uint16_max:
                break
                        
            if chain[beg_pos, mon_type_pos] in [free_mon, free_rad_mon]:
                beg_pos += 1
                continue
            
            if chain[beg_pos, mon_type_pos] != beg_mon:
                print('mon_type', chain[beg_pos, mon_type_pos])
                print('idx, beg_pos', idx, beg_pos)
                print('chain indexing error!')
            
            where_result = np.where(chain[beg_pos:, mon_type_pos] == end_mon)[0]
            
            if len(where_result) == 0:
                break
            
            end_pos = beg_pos + where_result[0]
            now_chain_len = end_pos - beg_pos
            
            inds_list = []
            
            for mon_line in chain[beg_pos:end_pos+1]:
                
                x_pos, y_pos, z_pos = mon_line[:3]
                
                if x_pos == y_pos == z_pos == uint16_max:
                    continue
                
                now_poss = [x_pos, y_pos, z_pos]
                
                if now_poss in inds_list:
                    continue
                
                chain_sum_len_matrix[x_pos, y_pos, z_pos] += now_chain_len
                n_chains_matrix[x_pos, y_pos, z_pos] += 1
                
                inds_list.append(now_poss)
            
            beg_pos = end_pos + 1
            
    return chain_sum_len_matrix, n_chains_matrix


## calculate final L distribution
def get_L_final(chain_table):
    
    L_final = []

    for i, now_chain in enumerate(chain_table):
        
        mu.pbar(i, len(chain_table))
        cnt = 0
        
        for line in now_chain:
            
            if np.all(line == uint16_max):
                break
            
            mon_type = line[mon_type_pos]
                    
            if mon_type == 0:
                cnt == 1
            
            elif mon_type == 1:
                cnt += 1
            
            elif mon_type == 2:
                cnt += 1
                L_final.append(cnt)            
                cnt = 0
    
    return np.array(L_final)

