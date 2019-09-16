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

os.chdir(mc.sim_folder + 'mapping')


#%%
beg_mon, mid_mon, end_mon = 0, 1, 2
free_mon = 10

n_chain_ind = 0
mon_line_ind = 3
mon_type_ind = -1
#uint16_max = 65535
uint32_max = 4294967295

before_msg = '#################################################\n'
after_msg = 'Result:\n'


def rewrite_mon_type(resist_matrix, chain_table, n_mon, new_type):

    chain_table[n_mon, mon_type_ind] = new_type

    xi, yi, zi, mon_line_pos = chain_table[n_mon, :mon_type_ind].astype(int)
    resist_matrix[xi, yi, zi, mon_line_pos, mon_type_ind] = new_type


def write_log_table(chain_table, n_chain, n_mon, msg=''):
    
    msg += 'n_chain = ' + str(n_chain) + '\n\n'
    msg += 'n_mon = ' + str(n_mon) + '\n'
    
    if n_mon-3 >= 0 and n_mon+3 < len(chain_table):
        msg += 'chain table part:\n' +\
            str(int(n_mon-3)) + ' ' + str(int(chain_table[n_mon-3, -1])) + '\n' +\
            str(int(n_mon-2)) + ' ' + str(int(chain_table[n_mon-2, -1])) + '\n' +\
            str(int(n_mon-1)) + ' ' + str(int(chain_table[n_mon-1, -1])) + '\n' +\
            str(int(n_mon))   + ' ' + str(int(chain_table[n_mon  , -1])) + '\n' +\
            str(int(n_mon+1)) + ' ' + str(int(chain_table[n_mon+1, -1])) + '\n' +\
            str(int(n_mon+2)) + ' ' + str(int(chain_table[n_mon+2, -1])) + '\n' +\
            str(int(n_mon+3)) + ' ' + str(int(chain_table[n_mon+3, -1])) + '\n'
    
    msg += '\n'
    
    with open('logfile.txt', 'a') as myfile:
        myfile.write(msg)


def write_log_var(mon_type, n_next_mon, next_mon_type, next_mon_new_type):
    
    msg = 'mon_type = ' + str(mon_type) + '\n' +\
        'n_next_mon = ' + str(n_next_mon) + '\n' +\
        'next_mon_type = ' + str(next_mon_type) + '\n' +\
        'next_mon_new_type = ' + str(next_mon_new_type) + '\n'
    
    with open('logfile.txt', 'a') as myfile:
        myfile.write(msg)


#%%
#e_matrix = np.load(mc.sim_folder + 'e-events_matrix/Harris_e_matrix_val_+-1.npy')
e_matrix = np.load(mc.sim_folder + 'e-events_matrix/Harris_e_matrix_val_Dapor_NEW.npy')
resist_matrix = np.load(mc.sim_folder + 'PMMA_sim/MATRIX_resist_Harris_fit.npy')


chain_tables_folder = '/Volumes/ELEMENTS/Harris_chain_tables_fit/'
files = os.listdir(chain_tables_folder)

chain_tables = []

N_chains_total = 1606

for i in range(N_chains_total):
    
    mu.pbar(i, len(files))
    
    chain_tables.append(np.load(chain_tables_folder + 'chain_table_' + str(i) + '.npy'))


N_chains_total  = len(chain_tables)

resist_shape = np.shape(resist_matrix)[:3]


#%%
log_fname = 'logfile.txt'

if os.path.exists(log_fname):
    os.remove(log_fname)


for xi, yi, zi in product(range(resist_shape[0]), range(resist_shape[1]), range(resist_shape[2])):
    
    if yi == zi == 0:
        mu.pbar(xi, resist_shape[0])
    
    n_events = e_matrix[xi, yi, zi]
    
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
        
        n_chain, n_mon, mon_type = resist_matrix[xi, yi, zi, monomer_pos, :]
        
        chain_table = chain_tables[n_chain]
        
        
        if mon_type != chain_table[n_mon, -1]:
            print('FUKKK!!', n_chain, n_mon)
            print(mon_type, chain_table[n_mon, -1])
        
        
        if len(chain_table) == 1:
            continue
        
#        write_log_table(chain_table, n_chain, n_mon, before_msg)
        
        
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
                
#                write_log_var(mon_type, n_next_mon, next_mon_type, next_mon_new_type)
#                write_log_table(chain_table, n_chain, n_mon, after_msg)
            
            
            ## if next monomer is full bonded
            elif next_mon_type == mid_mon:
                
                next_mon_new_type = next_mon_type - (new_mon_type - 1)
                
                rewrite_mon_type(resist_matrix, chain_table, n_next_mon, next_mon_new_type)
                
#                write_log_var(mon_type, n_next_mon, next_mon_type, next_mon_new_type)
#                write_log_table(chain_table, n_chain, n_mon, after_msg)
            
            else:
                print('\nerror!')
                print('n_chain', n_chain)
                print('n_mon', n_mon)
                print('next_mon_type', next_mon_type)
                
#                write_log_table(chain_table, n_chain, n_mon, 'Pizdos!\n' + after_msg)
                
                
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
                
#                write_log_var(mon_type, n_next_mon, next_mon_type, next_mon_new_type)
#                write_log_table(chain_table, n_chain, n_mon, after_msg)
            
            ## if next monomer is full bonded !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            elif next_mon_type == mid_mon:
                
                next_mon_new_type = next_mon_type + (mon_type - 1)
                
                rewrite_mon_type(resist_matrix, chain_table, n_next_mon, next_mon_new_type)
                
#                write_log_var(mon_type, n_next_mon, next_mon_type, next_mon_new_type)
#                write_log_table(chain_table, n_chain, n_mon, after_msg)
                
            else:
                print('error 2', next_mon_type)
                
#                write_log_table(chain_table, n_chain, n_mon, 'Pizdos2' + after_msg)
        
        else:
            continue


#%%
lens = []

for i, now_chain in enumerate(chain_tables):
    
    mu.pbar(i, len(chain_tables))
    cnt = 0
    
    if len(now_chain) == 1:
        lens.append(cnt) 
        continue
    
    for line in now_chain:
        
        mon_type = line[mon_type_ind]
                
        if mon_type == 0:
            cnt == 1
        
        elif mon_type == 1:
            cnt += 1
        
        elif mon_type == 2:
            cnt += 1
            lens.append(cnt)            
            cnt = 0


chain_lens = np.array(lens)


#%%
#chain_lens_new = np.load('mapping_chain_lens_+-1_fit.npy')

xx = np.load('../PMMA_sim/harris_x_after.npy')
#yy = np.load('../PMMA_sim/harris_y_after_SZ.npy')
yy = np.load('../PMMA_sim/harris_y_after_fit.npy')

mass = np.array(chain_lens)*100

bins = np.logspace(2, 7.1, 21)

plt.hist(mass, bins, label='simulation')
plt.gca().set_xscale('log')

plt.plot(xx, yy*7.1e+7, label='Harris (paper)')

plt.title('Harris final molecular weight distribution FIT')
plt.xlabel('molecular weight')
plt.ylabel('density')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Harris_final_weight_distr_Dapor_E_bind_3p3_fit.png', dpi=300)


#%% Test integral distributions
yy_int = np.cumsum(yy)

plt.hist(mass, bins, cumulative=True, label='simulation')

plt.plot(xx, yy_int*157000)

plt.gca().set_xscale('log')

