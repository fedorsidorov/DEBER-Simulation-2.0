#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import my_functions as mf
import my_variables as mv
import copy
from itertools import product

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


#%% total arrays
e_matrix = np.load('../e_DATA/matrixes/Wall/e_matrix_C_exc.npy')
part_matrix_0 = np.load('../chain_DATA/chain_matrix_short.npy')
chain_inv_matrix_0 = np.load('../chain_DATA/chain_matrix_inv_short.npy')

scission_matrix = np.zeros(np.shape(e_matrix))
monomer_matrix = np.zeros(np.shape(e_matrix))

#% load and correct part_matrix - 45 s
part_matrix = np.zeros((mv.n_Z_new, mv.n_XY, mv.n_x, mv.n_y, mv.n_z, mv.n_part_max, 3))*np.nan

print('loading part_matrix')

for i in range(3):
    
    mf.upd_progress_bar(i, 3)
    
    for i0, i1, i2, i3, i4 in product(range(mv.n_Z * i, mv.n_Z*(i + 1)),\
        range(mv.n_XY), range(mv.n_x), range(mv.n_y), range(mv.n_z)):
    
        part_matrix[i0, i1, i2, i3, i4] = part_matrix_0[i0 - mv.n_Z * i, i1, i2, i3, i4]
        part_matrix[i0, i1, i2, i3, i4] += mv.n_chains_short * i, 0, 0


#% load chain_inv_matrix - 2 m
chain_inv_matrix = np.zeros((mv.n_chains_short*3, mv.chain_len_max_short, 7))*np.nan

print('loading chain_inv_matrix')

for i in range(3):
    
    mf.upd_progress_bar(i, 3)
    
    chain_inv_matrix[mv.n_chains_short * i : mv.n_chains_short * (i+1)] = chain_inv_matrix_0
    
    for i0 in range(mv.n_chains_short * i, mv.n_chains_short*(i + 1)):
        chain_inv_matrix[i0] += mv.n_Z * i, 0, 0, 0, 0, 0, 0

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

n_CO = 0
n_CO2 = 0
n_CH4 = 0
n_ester = 0

n_scission = 0

#%
for Z, XY, x, y, z in product(range(s0), range(s1), range(s2), range(s3), range(s4)):
    
    if XY == z == y == x == 0:
        print('Z =', Z)
    
    cell_coords = Z, XY, x, y, z
    
    n_events = get_n_events(cell_coords)
    
    for i in range(n_events):
        
        ## only ... C atoms of 5 are of interest
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
                
                new_cell_coords = next_mon_inv_line[:5].astype(int)
                next_mon_pos, next_mon_type = next_mon_inv_line[5:]
                
                ## if next monomer was at the end
                if next_mon_type in [-1, 1]:
                    rewrite_mon_type(n_chain, n_next_mon, 2)
                    monomer_matrix[Z, XY, x, y, z] += 1
                
                elif next_mon_type in [9, 11]:
                    rewrite_mon_type(n_chain, n_next_mon, 12)
                    monomer_matrix[Z, XY, x, y, z] += 1
                
                ## if next monomer is full bonded
                elif next_mon_type in [0, 10]:
                    
                    next_mon_new_type = next_mon_type - new_mon_kind
                    rewrite_mon_type(n_chain, n_next_mon, next_mon_new_type)
                        
                else:
                    print('error 1', next_mon_type)
                    
                scission_matrix[Z, XY, x, y, z] += 1
            
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
                    monomer_matrix[Z, XY, x, y, z] += 1
                
                elif next_mon_type in [9, 11]:
                    rewrite_mon_type(n_chain, n_next_mon, 12)
                    monomer_matrix[Z, XY, x, y, z] += 1
                
                ## if next monomer is full bonded
                elif next_mon_type in [0, 10]:
                    next_mon_new_type = next_mon_type + mon_kind
                    rewrite_mon_type(n_chain, n_next_mon, next_mon_new_type)
                        
                else:
                    print('error 2', next_mon_type)
                
                scission_matrix[Z, XY, x, y, z] += 1
            
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
np.save('Wall_scission_matrix.npy', scission_matrix)
np.save('Wall_monomer_matrix.npy', monomer_matrix)

#%%
monomer_matrix = np.load('Wall_monomer_matrix.npy')

res_monomer_matrix = np.zeros((240, 50))

for Z, XY, x, y, z in product(range(s0), range(s1), range(s2), range(s3), range(s4)):
    
#    if (40 <= XY < 50 and y == 4) or (50 <= XY < 60 and y == 0):
    if True:
    
        ind_x = (XY % 10) * 5 + x
        ind_z = Z * 5 + z
        
        res_monomer_matrix[ind_z, ind_x] += monomer_matrix[Z, XY, x, y, z]

plt.imshow(res_monomer_matrix)
plt.colorbar(orientation='vertical')
ax = plt.gca()
ax.set_xticklabels((0, 0, 50))
ax.set_yticklabels((0, 0, 100, 200, 300, 400))
plt.xlabel('x, nm')
plt.ylabel('z, nm')

#%%
radical_matrix = np.load('Wall_scission_matrix.npy')

res_radical_matrix = np.zeros((240, 50))

for Z, XY, x, y, z in product(range(s0), range(s1), range(s2), range(s3), range(s4)):
    
    ## slice
#    if (40 <= XY < 50 and y == 4) or (50 <= XY < 60 and y == 0):
    if True:
        
        ind_x = (XY % 10) * 5 + x
        ind_z = Z * 5 + z
        
        res_radical_matrix[ind_z, ind_x] += radical_matrix[Z, XY, x, y, z]

plt.imshow(res_radical_matrix)
plt.colorbar(orientation='vertical')
ax = plt.gca()
ax.set_xticklabels((0, 0, 50))
ax.set_yticklabels((0, 0, 100, 200, 300, 400))
plt.xlabel('x, nm')
plt.ylabel('z, nm')

#%% Get radicals
radical_matrix = np.zeros((mv.n_chains_short*3, mv.chain_len_max_short))

for i in range(mv.n_chains_short*3):
    
    mf.upd_progress_bar(i, mv.n_chains_short*3)
    
    radical_matrix[i] = chain_inv_matrix[i, :, -1]

np.save('Wall_radical_matrix.npy', radical_matrix)

#%% Get L distribution
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

#%%
np.save('Sharma/final_L_new.npy', np.array(L_final))

#%% destroy some ester groups
ester_part = 0.5

add_CO = n_ester * ester_part * d_CO
n_CO_final = n_CO + add_CO

add_CO2 = n_ester * ester_part * d_CO2
n_CO2_final = n_CO2 + add_CO2

n_CH4_final = n_CH4 + (add_CO + add_CO2)
n_ester_final = n_ester - (add_CO + add_CO2)

#%%
print('n_CO', n_CO)
print('n_CO2', n_CO2)
print('n_CH4', n_CH4)
print('n_ester', n_ester)
