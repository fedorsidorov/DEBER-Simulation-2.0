#%% Import
import numpy as np
import os
import importlib
from itertools import product

import my_functions as mf
import my_variables as mv

mf = importlib.reload(mf)
mv = importlib.reload(mv)

os.chdir(mv.sim_path_MAC + 'map_matrixes')

#%% functions
def rewrite_mon_type(n_chain, n_mon, new_type):
    chain_inv_matrix[n_chain, n_mon, -1] = new_type
    
    Z, XY, x, y, z = chain_inv_matrix[n_chain, n_mon, :-2].astype(int)
    pos = chain_inv_matrix[n_chain, n_mon, -2]
    
    if not np.isnan(pos):
        part_matrix[Z, XY, x, y, z, int(pos), -1] = new_type


def add_ester_group(cell_coords):
    Z, XY, x, y, z = cell_coords
    ester_ind = np.where(np.isnan(part_matrix[Z, XY, x, y, z, :, 0]))[0][0]
    part_matrix[Z, XY, x, y, z, ester_ind, :] = -100
    return 1


def delete_ester_group(cell_coords, ind):
    Z, XY, x, y, z = cell_coords
    part_matrix[Z, XY, x, y, z, ind, :] = np.nan
    
    ester_add = -1
    CO_add = 0
    CO2_add = 0
    CH4_add = 1
    
    easter_decay = mf.choice((mv.ester_CO, mv.ester_CO2), p=(d_CO, d_CO2))
    
    if easter_decay == mv.ester_CO:
        CO_add = 1
    else:
        CO2_add = 1
    
    return ester_add, CO_add, CO2_add, CH4_add


def get_part_ind(cell_coords):    
    Z, XY, x, y, z = cell_coords
    part_inds = np.where(np.logical_not(np.isnan(part_matrix[Z, XY, x, y, z, :, 0])))[0]
    
    if len(part_inds) == 0:
            return -1
    
    return mf.choice(part_inds)


def get_ester_decay():
    return mf.choice((mv.ester_CO, mv.ester_CO2), p=(d_CO, d_CO2))


def get_process_3():
    probs = (d1, d2, d3)/np.sum((d1, d2, d3))
    return mf.choice((mv.sci_ester, mv.sci_direct, mv.ester), p=probs)
    

def get_mon_kind():
    return mf.choice([-1, 1])


def get_inv_line(n_chain, n_mon):
    return chain_inv_matrix[n_chain, n_mon]


def get_n_events(cell_coords):
    Z, XY, x, y, z = cell_coords
    return e_matrix[Z, XY, x, y, z].astype(int)


def get_part_line(cell_coords, part_ind):
    Z, XY, x, y, z = cell_coords
    return part_matrix[Z, XY, x, y, z, part_ind, :]


def mon_type_to_kind(mon_type):
    if mon_type in [-1, 0, 1]:
        return mon_type
    else:
        return mon_type - 10


#%%
part_matrix = np.load('../chain_DATA/chain_matrix_short.npy')
chain_inv_matrix = np.load('../chain_DATA/chain_matrix_inv_short.npy')

n_scission = 0
n_CO = 0
n_CO2 = 0
n_CH4 = 0
n_ester = 0

#%%
e_matrix = np.load('../e_DATA/matrixes/Aktary/e_matrix_25uC_C_exc.npy')

#%
s0, s1, s2, s3, s4 = np.shape(e_matrix)

## probabilities
p1 = 0.3 ## ester group detachment with scissions
p2 = 0.5 ## sure lol
p3 = 0.7 ## ester group detachment w/o scissions

d1, d2, d3 = p1 - 0, p2 - p1, p3 - p2 

## scission ways
k_CO = 25.3
k_CO2 = 13

d_CO, d_CO2 = (k_CO, k_CO2) / np.sum((k_CO, k_CO2))

#%%
sci_matrix = np.zeros(np.shape(e_matrix))

#len_matrix = np.zeros(np.shape(e_matrix))
#
#for Z, XY, x, y, z in product(range(s0), range(s1), range(s2), range(s3), range(s4)):
#    
#    part_inds = np.where(np.logical_not(np.isnan(part_matrix[Z, XY, x, y, z, :, 0])))[0]
#    len_matrix[Z, XY, x, y, z] = len(part_inds)

#%%
for Z, XY, x, y, z in product(range(s0), range(s1), range(s2), range(s3), range(s4)):
    
    if XY == z == y == x == 0:
        print('Z =', Z)
    
    cell_coords = Z, XY, x, y, z
    
    n_events = get_n_events(cell_coords)
    
    for i in range(n_events):
        
        if mf.random() >= p3:
            continue
        
        part_ind = get_part_ind(cell_coords)
        
        if part_ind == -1:
            continue
        
        part_line = get_part_line(cell_coords, part_ind)
        n_chain, n_mon, mon_type = list(map(int, part_line))
        
###############################################################################
        if mon_type == -100: # ester group ####################################
###############################################################################
            
            ester_add, CO_add, CO2_add, CH4_add = delete_ester_group(cell_coords, part_ind)
            
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
                n_ester += add_ester_group(cell_coords)
            
            ## deal with monomer types
            elif process in [mv.sci_ester, mv.sci_direct]:
                
                new_mon_kind = get_mon_kind()
                new_mon_type = mon_type + new_mon_kind
                rewrite_mon_type(n_chain, n_mon, new_mon_type)
                
                n_next_mon = n_mon + new_mon_kind
                next_mon_inv_line = get_inv_line(n_chain, n_next_mon)
                
#                new_cell_coords = next_mon_inv_line[:5].astype(int)
                next_mon_pos, next_mon_type = next_mon_inv_line[5:]
                
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
                    print('error 1', next_mon_type)
                
                n_scission += 1
                sci_matrix[Z, XY, x, y, z] = 1
            
            ## scission with ester group deatachment
            if process == mv.sci_ester:
                
                rewrite_mon_type(n_chain, n_mon, new_mon_type + 10)
                n_ester += add_ester_group(cell_coords)
                
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
                n_ester += add_ester_group(cell_coords)
            
            ## deal with monomer types
            elif process in [mv.sci_ester, mv.sci_direct]:
                
                mon_kind = mon_type_to_kind(mon_type)
                
                if mon_kind in [-1, 1]:
                    new_mon_type = 2
                else:
                    new_mon_type = 12
                
                rewrite_mon_type(n_chain, n_mon, new_mon_type)
                
                n_next_mon = n_mon - mon_kind ## minus, Karl!
                next_mon_inv_line = get_inv_line(n_chain, n_next_mon)
                next_mon_pos, next_mon_type = next_mon_inv_line[5:]
                
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
                sci_matrix[Z, XY, x, y, z] = 1
            
            ## scission with ester group deatachment
            if process == mv.sci_ester:
                
                rewrite_mon_type(n_chain, n_mon, new_mon_type + 10)
                n_ester += add_ester_group(cell_coords)
        
###############################################################################
        elif mon_type == 2: # free monomer with ester group ###################
###############################################################################
            
            ## only ester group deatachment is possible
            rewrite_mon_type(n_chain, n_mon, 12)
            n_ester += add_ester_group(cell_coords)
            
###############################################################################
        elif mon_type == 12: # free monomer w/o ester group ###################
###############################################################################
            
            continue
        
        else:
            print('WTF', mon_type)

#%%
L_final = []

n = 0

for chain in chain_inv_matrix:
    
    mf.upd_progress_bar(n, 12709)
    n += 1
    cnt = 0
    
    for line in chain:
        
        if np.all(np.isnan(line)):
            break
        
        mon_type = line[-1]
                
        if mon_type == -1:
            cnt == 0
        
        elif mon_type == 0:
            cnt += 1
        
        elif mon_type == 1:
            L_final.append(cnt)            
            cnt = 0

#%
np.save('final_L_2C_exc.npy', np.array(L_final))

#%% Sharma G-value
#N_el_dep = 6e-5 / 1.6e-19 * 1e-10
#E_dep =  N_el_dep * 25e+3
#G_value = n_scission / (E_dep / 100)
#print(G_value * 100)

#%%
#% Get radicals
#radical_matrix = np.zeros((mv.n_chains_short*3, mv.chain_len_max_short))
#
#for i in range(mv.n_chains_short):
#    
#    mf.upd_progress_bar(i, mv.n_chains_short*3)
#    
#    radical_matrix[i] = chain_inv_matrix[i, :, -1]
#
#np.save('Wall/radical_matrix_' + n_el + '.npy', radical_matrix)

#%% Get L distribution
#L_final = []
#
#n = 0
#
#for chain in chain_inv_matrix:
#    
#    mf.upd_progress_bar(n, 12709)
#    n += 1
#    cnt = 0
#    
#    for line in chain:
#        
#        if np.all(np.isnan(line)):
#            break
#        
#        mon_type = line[-1]
#                
#        if mon_type == -1:
#            cnt == 0
#        
#        elif mon_type == 0:
#            cnt += 1
#        
#        elif mon_type == 1:
#            L_final.append(cnt)            
#            cnt = 0
#
##%%
#np.save('Sharma/final_L_new.npy', np.array(L_final))
#
##%% destroy some ester groups
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
