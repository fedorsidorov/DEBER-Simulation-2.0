#%% Import
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import importlib
#import copy
from itertools import product

import my_utilities as mu
mu = importlib.reload(mu)

import my_constants as mc
mc = importlib.reload(mc)

os.chdir(os.path.join(mc.sim_folder, 'mapping_Harris'))


#%%
beg_mon, mid_mon, end_mon = 0, 1, 2
free_mon = 10

n_chain_ind = 0
mon_line_ind = 3
mon_type_ind = -1
#uint16_max = 65535
uint32_max = 4294967295

def rewrite_mon_type(resist_matrix, chain_table, n_mon, new_type):

    chain_table[n_mon, mon_type_ind] = new_type

    xi, yi, zi, mon_line_pos = chain_table[n_mon, :mon_type_ind].astype(int)
    resist_matrix[xi, yi, zi, mon_line_pos, mon_type_ind] = new_type


#%%
e_matrix = np.load(os.path.join(mc.sim_folder, 'e-matrix_Harris', '2020',\
                'Harris_e_matrix_val_2СС+05ester.npy'))

resist_matrix = np.load(os.path.join(mc.sim_folder, 'PMMA_sim_Harris',\
                'MATRIX_resist_Harris_2020.npy'))

chain_tables_folder = os.path.join(mc.sim_folder, 'PMMA_sim_Harris', 'Harris_chain_tables_2020')
files = os.listdir(chain_tables_folder)

chain_tables = []

N_chains_total = 1370


for i in range(N_chains_total):
    
    mu.pbar(i, len(files))
    
    chain_tables.append(np.load(os.path.join(chain_tables_folder,
                'chain_table_' + str(i) + '.npy')))


N_chains_total  = len(chain_tables)

resist_shape = np.shape(resist_matrix)[:3]


#%%
lens_before = np.zeros(len(chain_tables))

for i in range(len(chain_tables)):
    
    lens_before[i] = len(chain_tables[i])


#%%
for xi, yi, zi in product(range(resist_shape[0]), range(resist_shape[1]),\
                          range(resist_shape[2])):
    
    if yi == zi == 0:
        mu.pbar(xi, resist_shape[0])
    
    n_events = e_matrix[xi, yi, zi]
    
    for i in range(int(n_events)):
        
        monomer_positions = np.where(resist_matrix[xi, yi, zi, :, n_chain_ind]\
                                     != uint32_max)[0]
    
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
lens_final = []


for i, now_chain in enumerate(chain_tables):
    
    mu.pbar(i, len(chain_tables))
    cnt = 0
    
    if len(now_chain) == 1:
        lens_final.append(cnt+1) 
        continue
    
    for line in now_chain:
        mon_type = line[mon_type_ind]
                
        if mon_type == 0:
            cnt == 1
        
        elif mon_type == 1:
            cnt += 1
        
        elif mon_type == 2:
            cnt += 1
            lens_final.append(cnt)            
            cnt = 0


chain_lens_final = np.array(lens_final)


#%%
np.save('lens_final_2CC+05ester_2020.npy', chain_lens_final)


#%%
chain_lens_initial = np.load(os.path.join(mc.sim_folder,
            'PMMA_sim_Harris', 'Harris_chain_lens_2020.npy'))

#chain_lens_final = np.load('lens_final_2CC_2020.npy')

xx = np.load('../PMMA_sim_Harris/PMMA_sim_2019/harris_x_after.npy')
#yy_SZ = np.load('../PMMA_sim_Harris/harris_y_after_SZ.npy')
yy = np.load('../PMMA_sim_Harris/PMMA_sim_2019/harris_y_after_fit.npy')

mass = np.array(chain_lens_final)*100

#bins = np.logspace(2, 7.1, 21)
bins = np.logspace(2, 7.1, 21)

plt.hist(mass, bins, label='simulation')

plt.plot(xx, yy*2.2e+7, label='experiment')

plt.title('Harris final molecular weight distribution FIT')
plt.xlabel('molecular weight')
plt.ylabel('N$_{entries}$')

plt.xlim(1e+2, 1e+6)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Harris_final_weight_distr_E_bind_4p94.png', dpi=300)
#plt.savefig('fig2.png', dpi=600)


#%% get G(S)
M0 = 100
Mn0 = np.average(np.load(os.path.join(mc.sim_folder, 'PMMA_sim_Harris',\
                                      'Harris_chain_lens_2020.npy')) * M0)
Mn = np.average(chain_lens_final * M0)

ps = (1/Mn - 1/Mn0)*M0

e_matrix_dE = np.load(os.path.join(mc.sim_folder,
            'e-matrix_Harris', '2020', 'Harris_e_matrix_dE_2СС+05ester.npy'))
Gs = (ps*5.95e-5*6.02e+23) / (np.sum(e_matrix_dE)*1e+10)

print('Gs =', Gs)


