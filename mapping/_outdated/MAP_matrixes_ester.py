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

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mi = importlib.reload(mi)
mc = importlib.reload(mc)

os.chdir(mv.sim_path_MAC + 'map_matrixes')

#%% functions
## changes monomer type
def rewrite_mon_type(n_chain, n_mon, new_type):
    ## change mon_type in chain_inv_matrix
    chain_table[n_chain, n_mon, mi.mon_type] = new_type
    ## define x, y, z of monomer
    x_ind, y_ind, z_ind, mon_line_pos = chain_table[n_chain, n_mon, :mi.mon_type]
    ## if we aren't outside the resist area of interest
    if not x_ind == y_ind == z_ind == mon_line_pos == mc.uint16_max:
        resist_matrix[x_ind, y_ind, z_ind, mon_line_pos, mi.mon_type] = new_type


## add ester group to resist array
def add_ester_group(x_ind, y_ind, z_ind):
    ## looking for a free space
    ester_ind = np.where(resist_matrix[x_ind, y_ind, z_ind, :, mi.n_chain] ==\
                         mc.uint16_max)[0][0]
    ## and write to it ester group line
    resist_matrix[x_ind, y_ind, z_ind, ester_ind, :] = 100
    return 1


## remove ester group from resist_matrix
def remove_ester_group(x_ind, y_ind, z_ind, ester_line_pos):
    ## make free corresponding line of resist_matrix
    resist_matrix[x_ind, y_ind, z_ind, ester_line_pos, :] = mc.uint16_max
    ## make a desicion on ester group decay
    ester_add, CO_add, CO2_add, CH4_add = -1, 0, 0, 1
    easter_decay = mf.choice((mv.ester_CO, mv.ester_CO2), p=(d_CO, d_CO2))
    if easter_decay == mv.ester_CO:
        CO_add = 1
    else:
        CO2_add = 1
    return ester_add, CO_add, CO2_add, CH4_add


## choose one of existing particles to interact electron with
def get_resist_part_ind(x_ind, y_ind, z_ind):
    ## indexes of existing particle lines
    resist_part_inds = np.where(resist_matrix[x_ind, y_ind, z_ind, :, mi.n_chain] !=\
                         mc.uint16_max)[0]
    ## if no free particles
    if len(resist_part_inds) == 0:
            return -1
    ## else
    return mf.choice(resist_part_inds)


## choose the way of ester decay
def get_ester_decay():
    return mf.choice((mv.ester_CO, mv.ester_CO2), p=(d_CO, d_CO2))


## choose of the process
def get_process_3():
    probs = (d1, d2, d3)/np.sum((d1, d2, d3))
    return mf.choice((mv.sci_ester, mv.sci_direct, mv.ester), p=probs)
    

## choose of monomer kind
def get_mon_kind():
    return mf.choice([-1, 1])


## get neede chain_table line
def get_chain_table_line(n_chain, n_mon):
    return chain_table[n_chain, n_mon]


## get n events in the cell
def get_n_events(x_ind, y_ind, z_ind):
    return e_matrix[x_ind, y_ind, z_ind]


## get particle line from 
def get_resist_part_line(x_ind, y_ind, z_ind, resist_part_ind):
    return resist_matrix[x_ind, y_ind, z_ind, resist_part_ind, :]


## convert monomer type to monomer kind
def mon_type_to_kind(mon_type):
    ## with ester group
    if mon_type in [-1, 0, 1]:
        return mon_type
    ## W/O ester group
    else:
        return mon_type - 10


## convert 65536 to -1
def correct_mon_type(mon_type):
    if mon_type == mc.uint16_max:
        return -1
    return mon_type


## calculate local chain lengths distribution
#def get_local_chain_len():
#    
#    chain_sum_len_matrix = np.zeros(np.shape(e_matrix))
#    n_chains_matrix = np.zeros(np.shape(e_matrix))
#    
#    for chain in chain_table:
#        
#        beg_ind = 0
#        
#        while True:
#            
#            if chain[beg_ind, mi.mon_type] not in [mc.uint16_max, 9]:
#                print('chain indexing error!')
#            
#            st_1 = chain[beg_ind:, mi.mon_type] == 1
#            st_2 = chain[beg_ind:, mi.mon_type] == 11
#            
#            where_result = np.where(np.logical_or(st_1, st_2))[0]
#            
#            if len(where_result) == 0:
#                break
#            
#            end_ind = where_result[0]
#            
#            now_chain_len = end_ind - beg_ind
#            print(now_chain_len)
#            
#            inds_list = []
#            
#            for mon_line in chain[beg_ind:end_ind+1]:
#                
#                x_ind, y_ind, z_ind = mon_line[:3]
#                now_inds = [x_ind, y_ind, z_ind]
#                
#                if now_inds in inds_list:
#                    continue
#                
#                chain_sum_len_matrix[x_ind, y_ind, z_ind] += now_chain_len
#                n_chains_matrix[x_ind, y_ind, z_ind] += 1
#                
#                inds_list.append(now_inds)
#            
#            beg_ind = end_ind + 1
#    
#    return chain_sum_len_matrix, n_chains_matrix

#%%
l_xyz = np.array((600, 100, 122))

space = 50
beam_d = 1

x_beg, y_beg, z_beg = (-l_xyz[0]/2, 0, 0)
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

step_2nm = 2

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)

bins_2nm = x_bins_2nm, y_bins_2nm, z_bins_2nm

x_grid_2nm = (x_bins_2nm[:-1] + x_bins_2nm[1:]) / 2
y_grid_2nm = (y_bins_2nm[:-1] + y_bins_2nm[1:]) / 2
z_grid_2nm = (z_bins_2nm[:-1] + z_bins_2nm[1:]) / 2

#%%
n_scission = 0
n_CO = 0
n_CO2 = 0
n_CH4 = 0
n_ester = 0

e_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/MATRIX_e_500_pC_cm_C.npy')
resist_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/MATRIX_resist.npy')
chain_table = np.load(mv.sim_path_MAC + 'MATRIXES/TABLE_chains.npy')

s_0, s_1, s_2 = np.shape(e_matrix)

## probabilities
p1 = 0.3 ## ester group detachment with scissions
p2 = 0.5 ## sure lol
p3 = 0.7 ## ester group detachment w/o scissions

#p1 = 0.1
#p2 = 2/5 ## scission threshold
#p3 = 0.9

d1, d2, d3 = p1 - 0, p2 - p1, p3 - p2 

## scission ways
k_CO = 25.3
k_CO2 = 13

d_CO, d_CO2 = (k_CO, k_CO2) / np.sum((k_CO, k_CO2))

N_chains_total = 63306

#%%
chain_sum_len_matrix = np.zeros(np.shape(e_matrix))
n_chains_matrix = np.zeros(np.shape(e_matrix))

for idx, chain in enumerate(chain_table):
    
    mf.upd_progress_bar(idx, N_chains_total)
    
    beg_ind = 0
    
#    while True:
    for ii in range(3):
        
        if beg_ind >= 9780:
            break
        
#        print('beg_ind', beg_ind)
        
        if chain[beg_ind, mi.mon_type] in [2, 12]:
            beg_ind += 1
            continue
        
        if chain[beg_ind, mi.mon_type] not in [mc.uint16_max, 9]:
            print('mon_type', chain[beg_ind, mi.mon_type])
            print('chain indexing error!')
        
        st_1 = chain[beg_ind:, mi.mon_type] == 1
        st_2 = chain[beg_ind:, mi.mon_type] == 11
        
        where_result = np.where(np.logical_or(st_1, st_2))[0]
        
        if len(where_result) == 0:
            break
        
#        end_ind = where_result[0]
        end_ind = beg_ind + where_result[0]
#        print('end_ind', end_ind)
        
        now_chain_len = end_ind - beg_ind
#        print('len', now_chain_len)
        
        inds_list = []
        
        for mon_line in chain[beg_ind:end_ind+1]:
            
            x_ind, y_ind, z_ind = mon_line[:3]
            
            if x_ind == y_ind == z_ind == mc.uint16_max:
                continue
            
            now_inds = [x_ind, y_ind, z_ind]
            
            if now_inds in inds_list:
                continue
            
            chain_sum_len_matrix[x_ind, y_ind, z_ind] += now_chain_len
            n_chains_matrix[x_ind, y_ind, z_ind] += 1
            
            inds_list.append(now_inds)
        
        beg_ind = end_ind + 1

#%%
ans = chain_sum_len_matrix / n_chains_matrix

#%%
for x_ind, y_ind, z_ind in product(range(s_0), range(s_1), range(s_2)):
    
    if y_ind == z_ind == 0:
        mf.upd_progress_bar(x_ind, s_0)
    
    n_events = get_n_events(x_ind, y_ind, z_ind)
    
    for i in range(n_events):
        
        ## only ... C atoms of 5 are of interest
        if mf.random() >= p3:
            continue
        
        resist_part_ind = get_resist_part_ind(x_ind, y_ind, z_ind)
        
        if resist_part_ind == -1:
            continue
        
        n_chain, n_mon, mon_type =\
            get_resist_part_line(x_ind, y_ind, z_ind, resist_part_ind)
        
        ## convert 65535 to -1
        mon_type = correct_mon_type(mon_type)
        
        if mon_type == mc.uint16_max:
            print('MON_TYPE ERROR!')
        
###############################################################################
        if mon_type == 100: # ester group ####################################
###############################################################################
            
            ester_add, CO_add, CO2_add, CH4_add =\
                remove_ester_group(x_ind, y_ind, z_ind, resist_part_ind)
            
            n_ester += ester_add
            n_CO += CO_add
            n_CO2 += CO2_add
            n_CH4 += CH4_add
            
            continue

############################################################################### 
        elif mon_type in [0, 10]: # bonded monomer with ester group ###########
###############################################################################
            
            if mon_type == 0:
                process = get_process_3()
            else:
                if mf.random() > (d1 + d2) / (d1 + d2 + d3):
                    continue
                process = mv.sci_direct
            
            if process == mv.ester:
                
                rewrite_mon_type(n_chain, n_mon, 10)
                n_ester += add_ester_group(x_ind, y_ind, z_ind)
            
            ## deal with monomer types
            elif process in [mv.sci_ester, mv.sci_direct]:
                
                new_mon_kind = get_mon_kind()
                new_mon_type = mon_type + new_mon_kind
                
                if new_mon_type not in [-1, 1, 9, 11]:
                    print('new_mon_type error!', new_mon_type)
                
                rewrite_mon_type(n_chain, n_mon, new_mon_type)
                
                n_next_mon = n_mon + new_mon_kind
                
                next_x_ind, next_y_ind, next_z_ind, _, next_mon_type =\
                    get_chain_table_line(n_chain, n_next_mon)
                
                next_mon_type = correct_mon_type(next_mon_type)
                
                ## if next monomer was at the end
                if next_mon_type in [-1, 1]:
                    rewrite_mon_type(n_chain, n_next_mon, 2)
                
                elif next_mon_type in [9, 11]:
                    rewrite_mon_type(n_chain, n_next_mon, 12)
                
                ## if next monomer is full bonded
                elif next_mon_type in [0, 10]:
                    next_mon_new_type = next_mon_type - new_mon_kind
                    rewrite_mon_type(n_chain, n_next_mon, next_mon_new_type)
                
                else:
                    print('error 1, next_mon_type =', next_mon_type)
                    print(x_ind, y_ind, z_ind)
                
                n_scission += 1
            
            ## scission with ester group deatachment
            if process == mv.sci_ester:
                
                rewrite_mon_type(n_chain, n_mon, new_mon_type + 10)
                n_ester += add_ester_group(x_ind, y_ind, z_ind)

###############################################################################
        elif mon_type in [-1, 1, 9, 11]: # half-bonded monomer with or w/o ester group
###############################################################################
            
            if mon_type in [-1, 1]:
                process = get_process_3()
            else:
                if mf.random() > (d1 + d2) / (d1 + d2 + d3):
                    continue
                process = mv.sci_direct
            
            if process == mv.ester:
                
                rewrite_mon_type(n_chain, n_mon, mon_type + 10)
                n_ester += add_ester_group(x_ind, y_ind, z_ind)
            
            ## deal with monomer types
            elif process in [mv.sci_ester, mv.sci_direct]:
                
                mon_kind = mon_type_to_kind(mon_type)
                
                if mon_kind in [-1, 1]:
                    new_mon_type = 2
                else:
                    new_mon_type = 12
                
                rewrite_mon_type(n_chain, n_mon, new_mon_type)
                
                n_next_mon = n_mon - mon_kind ## minus, Karl!
                
                next_x_ind, next_y_ind, next_z_ind, _, next_mon_type =\
                    get_chain_table_line(n_chain, n_next_mon)
                
                next_mon_type = correct_mon_type(next_mon_type)
                
                ## if next monomer was at the end
                if next_mon_type in [-1, 1]:
                    rewrite_mon_type(n_chain, n_next_mon, 2)
                
                elif next_mon_type in [9, 11]:
                    rewrite_mon_type(n_chain, n_next_mon, 12)
                
                ## if next monomer is full bonded
                elif next_mon_type in [0, 10]:
                    next_mon_new_type = next_mon_type + mon_kind
                    rewrite_mon_type(n_chain, n_next_mon, next_mon_new_type)
                    
                else:
                    print('error 2', next_mon_type)
                
                n_scission += 1
            
            ## scission with ester group deatachment
            if process == mv.sci_ester:
                
                rewrite_mon_type(n_chain, n_mon, new_mon_type + 10)
                n_ester += add_ester_group(x_ind, y_ind, z_ind)
        
###############################################################################
        elif mon_type == 2: # free monomer with ester group ###################
###############################################################################
            
            ## only ester group deatachment is possible
            rewrite_mon_type(n_chain, n_mon, 12)
            n_ester += add_ester_group(x_ind, y_ind, z_ind)
            
###############################################################################
        elif mon_type == 12: # free monomer w/o ester group ###################
###############################################################################
            
            continue
        
        else:
            print('WTF', mon_type)

#%%
chain_test_inds = np.random.choice(N_chains_total, 100, replace=False)
chain_test_table = chain_table[chain_test_inds]

#%%
for chain in chain_table:
    

#%%
L_final = []
radical_matrix = np.zeros((len(chain_table), len(chain_table[0])))

for i, now_chain in enumerate(chain_table):
    
    mf.upd_progress_bar(i, N_chains_total)
    cnt = 0
    
    radical_matrix[i] = now_chain[:, mi.mon_type]
    
    for line in now_chain:
        
        if np.all(line == mc.uint16_max):
            break
        
        mon_type = line[-1]
                
        if mon_type in [mc.uint16_max, 9]:
            cnt == 1
        
        elif mon_type in [0, 10]:
            cnt += 1
        
        elif mon_type in [1, 11]:
            cnt += 1
            L_final.append(cnt)            
            cnt = 0

#%%
np.save('MATRIX_radicals.npy', radical_matrix)
np.save('final_L_2.5C_exc.npy', np.array(L_final))

#%%
radical_matrix_uint16 = np.array(radical_matrix, dtype=np.uint16)

#%%
L_final_arr = np.array(L_final)

#%%
plt.hist(np.log(L_final_arr * 100))

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

#%% Sharma G-value
#N_el_dep = 6e-5 / 1.6e-19 * 1e-10
#E_dep =  N_el_dep * 25e+3
#G_value = n_scission / (E_dep / 100)
#print(G_value * 100)

#%%


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
