#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import copy
from itertools import product

import my_functions as mf
import my_variables as mv
import my_indexes as mi
import my_constants as mc
import my_mapping as mm

os.chdir(mv.sim_path_MAC + 'mapping')

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mi = importlib.reload(mi)
mc = importlib.reload(mc)
mm = importlib.reload(mm)

#%%
e_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/MATRIX_6e-5C_cm2_C_exc.npy')
resist_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/MATRIX_resist_Sharma.npy')
chain_table = np.load(mv.sim_path_MAC + 'MATRIXES/TABLE_chains_Sharma.npy')

N_chains_total = 12703
N_mon_chain_max = 8294

resist_shape = np.shape(resist_matrix)

#%%
e_test = e_matrix[20]
resist_test = resist_matrix[20, 20, 0]

#%%
#chain_sum_len_matrix, n_chains_matrix = mm.get_local_chain_len(np.shape(resist_matrix),
#                                        N_mon_chain_max, chain_table)
#
##%%
#np.save('chain_sum_len_after.npy', chain_sum_len_matrix)
#np.save('n_chains_after.npy', n_chains_matrix)

#%%
p_scission = 0.5
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
            elif next_mon_type == 1:
                next_mon_new_type = next_mon_type - (new_mon_type - 1)
                mm.rewrite_mon_type(resist_matrix, chain_table,\
                                 n_chain, n_next_mon, next_mon_new_type)
            
            else:
                print('error 1, next_mon_type =', next_mon_type)
                print(x_ind, y_ind, z_ind)
            
            n_scissions += 1

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
            
###############################################################################
        elif mon_type == mm.free_mon: ## free monomer with ester group ########
###############################################################################
            
            ## only ester group deatachment is possible
            mm.rewrite_mon_type(resist_matrix, chain_table,\
                             n_chain, n_mon, mm.free_rad_mon)
            
            n_scissions += 1
        
        elif mon_type == mm.free_rad_mon:
            continue
        
        else:
            print('WTF', mon_type)


#%%
#chain_test_inds = np.random.choice(N_chains_total, 100, replace=False)
#chain_test_table = chain_table[chain_test_inds]
#
#resist_matrix_test = resist_matrix[:, 25, 30]

#%%
L_final = []
#radical_matrix = np.zeros((len(chain_table), len(chain_table[0])))

for i, now_chain in enumerate(chain_table):
    
    mf.upd_progress_bar(i, N_chains_total)
    cnt = 0
    
#    radical_matrix[i] = now_chain[:, mm.mon_type_ind]
    
    for line in now_chain:
        
        if np.all(line == mc.uint16_max):
            break
        
        mon_type = line[mm.mon_type_ind]
                
        if mon_type == 0:
            cnt == 1
        
        elif mon_type == 1:
            cnt += 1
        
        elif mon_type == 2:
            cnt += 1
            L_final.append(cnt)            
            cnt = 0

#%%
L_final_arr = np.array(L_final)

log_mw = np.log10(L_final_arr * 100)
plt.hist(log_mw, bins=20, label='sample', normed=True)

data_Sharma = np.loadtxt(mv.sim_path_MAC +\
                         'L_distribution_simulation/curves/Sharma_peak_A.dat')

x_Sharma = data_Sharma[:, 0]
y_Sharma = data_Sharma[:, 1]

plt.plot(np.log10(x_Sharma), y_Sharma/y_Sharma.max()*1.1, label='model')

plt.title('Chain mass distribution')
plt.xlabel('log(m$_w$)')
plt.ylabel('probability')
plt.ylim((0, 1.2))
plt.legend()
plt.grid()
plt.show()

#%% Sharma G-value
N_el_dep = 6e-5 / 1.6e-19 * (100e-7)**2
E_dep =  N_el_dep * 25e+3
#G_value = n_scissions / (E_dep / 100)
G_value = n_events_total / (E_dep / 100)
print(G_value * 100)

#%%
chain_sum_len_matrix, n_chains_matrix = mm.get_local_chain_len(chain_table)

#%%
#np.save('MATRIX_radicals.npy', radical_matrix)
np.save('final_L_2.5C_exc.npy', np.array(L_final))

#%%
#radical_matrix_uint16 = np.array(radical_matrix, dtype=np.uint16)

#%%
L_final_arr = np.array(L_final)
L_test = L_final_arr[:1000]

#%%
plt.hist(np.log10(L_final_arr * 100))

#%% get monomers
monomer_matrix = np.zeros((s_0, s_1, s_2))

for x_ind, y_ind, z_ind in product(range(s_0), range(s_1), range(s_2)):
    
    if y_ind == z_ind == 0:
        mf.upd_progress_bar(x_ind, s_0)
    
    now_cube = resist_matrix[x_ind, y_ind, z_ind]
    
    inds_2 = np.where(now_cube[:, mi.mon_type] == 2)[0]
    inds_12 = np.where(now_cube[:, mi.mon_type] == 12)[0]
    
    monomer_matrix[x_ind, y_ind, z_ind] += len(inds_2) + len(inds_12)

#%% drawing
plt.figure()
plt.semilogy(x_grid_2nm, np.sum(monomer_matrix[:, 25, :], axis = 1), label='500 pC/cm')
plt.xlabel('x, nm')
plt.ylabel('N monomers')
plt.title('Monomer coordinate distribution, 2 nm')
plt.legend()
plt.grid()
plt.show()
plt.savefig('LOG monomers 2nm.png', dpi=300)

#%%
monomer_matrix_xz = np.sum(monomer_matrix, axis=1)

plt.figure()
plt.semilogy(x_grid_2nm, np.sum(monomer_matrix_xz, axis = 1), label='500 pC/cm')
plt.xlabel('x, nm')
plt.ylabel('N monomers')
plt.title('Monomer coordinate distribution, 100 nm')
plt.legend()
plt.grid()
plt.show()
plt.savefig('LOG monomers 100nm.png', dpi=300)

#%% destroy some ester groups
#ester_part = 0.5
#
#add_CO = n_ester * ester_part * d_CO
#n_CO_final = n_CO + add_CO
#
#add_CO2 = n_ester * ester_part * d_CO2
#n_CO2_final = n_CO2 + add_CO2
#
#n_CH4_final = n_CH4 + (add_CO + add_CO2)
#n_ester_final = n_ester - (add_CO + add_CO2)
#
##%%
#print('n_CO', n_CO)
#print('n_CO2', n_CO2)
#print('n_CH4', n_CH4)
#print('n_ester', n_ester)
