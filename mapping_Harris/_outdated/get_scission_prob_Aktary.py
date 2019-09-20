#%% Import
import matplotlib.pyplot as plt
import numpy as np
import os
import importlib
from itertools import product
import my_functions as mf
import my_variables as mv
import mapping_functions as mapf

mf = importlib.reload(mf)
mv = importlib.reload(mv)

os.chdir(mv.sim_path_MAC + 'map_matrixes')

#%%
s0, s1, s2, s3, s4 = 16, 100, 5, 5, 5

#%% functions
def get_scission_matrix(cnt):
    
    e_matrix = np.load('../e_DATA/matrixes/Aktary/100uC_rot/e_matrix_' + str(cnt) + '.npy')
    part_matrix = np.load('../chain_DATA/chain_matrix_short.npy')
    chain_inv_matrix = np.load('../chain_DATA/chain_matrix_inv_short.npy')
    
    ## probabilities
    p1 = 0.3 ## ester group detachment with scissions
    p2 = 0.5 ## sure lol
    p3 = 0.7 ## ester group detachment w/o scissions
    
    d1, d2, d3 = p1 - 0, p2 - p1, p3 - p2 
    
    ## scission ways
    k_CO = 25.3
    k_CO2 = 13
    
    d_CO, d_CO2 = (k_CO, k_CO2) / np.sum((k_CO, k_CO2))
    
    n_scission = 0
    n_CO = 0
    n_CO2 = 0
    n_CH4 = 0
    n_ester = 0
    
    sci_matrix = np.zeros(np.shape(e_matrix))

    for Z, XY, x, y, z in product(range(s0), range(s1), range(s2), range(s3), range(s4)):
        
#        if XY == z == y == x == 0:
#            print('Z =', Z)
        
        cell_coords = Z, XY, x, y, z
        
        n_events = mapf.get_n_events(e_matrix, cell_coords)
        
        for i in range(n_events):
            
            if mf.random() >= p3:
                continue
            
            part_ind = mapf.get_part_ind(part_matrix, cell_coords)
            
            if part_ind == -1:
                continue
            
            part_line = mapf.get_part_line(part_matrix, cell_coords, part_ind)
            n_chain, n_mon, mon_type = list(map(int, part_line))
            
    ###############################################################################
            if mon_type == -100: # ester group ####################################
    ###############################################################################
                
                ester_add, CO_add, CO2_add, CH4_add =\
                    mapf.delete_ester_group(part_matrix, cell_coords, part_ind, d_CO, d_CO2)
                
                n_ester += ester_add
                n_CO += CO_add
                n_CO2 += CO2_add
                n_CH4 += CH4_add
                
                continue
            
    ############################################################################### 
            elif mon_type in [0, 10]: # bonded monomer with ester group ###########
    ###############################################################################
                
                if mon_type == 0:
                    process = mapf.get_process_3(d1, d2, d3)
                else:
                    if mf.random() > (d1 + d2) / (d1 + d2 + d3):
                        continue
                    process = mv.sci_direct
                
                if process == mv.ester:
                    
                    mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_mon, 10)
                    n_ester += mapf.add_ester_group(part_matrix, cell_coords)
                
                ## deal with monomer types
                elif process in [mv.sci_ester, mv.sci_direct]:
                    
                    new_mon_kind = mapf.get_mon_kind()
                    new_mon_type = mon_type + new_mon_kind
                    mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_mon, new_mon_type)
                    
                    n_next_mon = n_mon + new_mon_kind
                    next_mon_inv_line =\
                        mapf.get_inv_line(chain_inv_matrix, n_chain, n_next_mon)
                    
                    next_mon_pos, next_mon_type = next_mon_inv_line[5:]
                    
                    ## if next monomer was at the end
                    if next_mon_type in [-1, 1]:
                        mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_next_mon, 2)
                    
                    elif next_mon_type in [9, 11]:
                        mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_next_mon, 12)
                    
                    ## if next monomer is full bonded
                    elif next_mon_type in [0, 10]:
                        
                        next_mon_new_type = next_mon_type - new_mon_kind
                        mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_next_mon, next_mon_new_type)
                            
                    else:
                        print('error 1', next_mon_type)
                    
                    n_scission += 1
                    sci_matrix[Z, XY, x, y, z] += 1
                
                ## scission with ester group deatachment
                if process == mv.sci_ester:
                    
                    mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_mon, new_mon_type + 10)
                    n_ester += mapf.add_ester_group(part_matrix, cell_coords)
                    
    ###############################################################################
            elif mon_type in [-1, 1, 9, 11]: # half-bonded monomer with or w/o ester group
    ###############################################################################
                
                if mon_type in [-1, 1]:
                    process = mapf.get_process_3(d1, d2, d3)
                else:
                    if mf.random() > (d1 + d2) / (d1 + d2 + d3):
                        continue
                    process = mv.sci_direct
                
                if process == mv.ester:
                    
                    mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_mon, mon_type + 10)
                    n_ester += mapf.add_ester_group(part_matrix, cell_coords)
                
                ## deal with monomer types
                elif process in [mv.sci_ester, mv.sci_direct]:
                    
                    mon_kind = mapf.mon_type_to_kind(mon_type)
                    
                    if mon_kind in [-1, 1]:
                        new_mon_type = 2
                    else:
                        new_mon_type = 12
                    
                    mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_mon, new_mon_type)
                    
                    n_next_mon = n_mon - mon_kind ## minus, Karl!
                    next_mon_inv_line =\
                        mapf.get_inv_line(chain_inv_matrix, n_chain, n_next_mon)
                    next_mon_pos, next_mon_type = next_mon_inv_line[5:]
                    
                    ## if next monomer was at the end
                    if next_mon_type in [-1, 1]:
                        mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_next_mon, 2)
                    
                    elif next_mon_type in [9, 11]:
                        mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_next_mon, 12)
                    
                    ## if next monomer is full bonded
                    elif next_mon_type in [0, 10]:
                        next_mon_new_type = next_mon_type + mon_kind
                        mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_next_mon, next_mon_new_type)
                        
                    else:
                        print('error 2', next_mon_type)
                    
                    n_scission += 1
                    sci_matrix[Z, XY, x, y, z] += 1
                
                ## scission with ester group deatachment
                if process == mv.sci_ester:
                    
                    mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_mon, new_mon_type + 10)
                    n_ester += mapf.add_ester_group(part_matrix, cell_coords)
            
    ###############################################################################
            elif mon_type == 2: # free monomer with ester group ###################
    ###############################################################################
                
                ## only ester group deatachment is possible
                mapf.rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_mon, 12)
                n_ester += mapf.add_ester_group(part_matrix, cell_coords)
                
    ###############################################################################
            elif mon_type == 12: # free monomer w/o ester group ###################
    ###############################################################################
                
                continue
            
            else:
                print('WTF', mon_type)
    
    return sci_matrix
    
#%%
total_sci_matrix = np.zeros([s0, s1, s2, s3, s4])
for cnt in range(100):
    
    mf.upd_progress_bar(cnt, 100)
    total_sci_matrix += get_scission_matrix(cnt)

#%%
len_matrix = np.zeros([s0, s1, s2, s3, s4])

part_matrix_0 = np.load('../chain_DATA/chain_matrix_short.npy')

for Z, XY, x, y, z in product(range(s0), range(s1), range(s2), range(s3), range(s4)):
    
    part_inds = np.where(np.logical_not(np.isnan(part_matrix_0[Z, XY, x, y, z, :, 0])))[0]
    len_matrix[Z, XY, x, y, z] = len(part_inds)
    
#%%
total_sci_matrix = np.load('total_sci_matrix_100_add.npy')
res_matrix = np.zeros([s0, s1, s2, s3, s4])

#%%
for Z, XY, x, y, z in product(range(s0), range(s1), range(s2), range(s3), range(s4)):
    
    if len_matrix[Z, XY, x, y, z] == 0:
        res_matrix[Z, XY, x, y, z] = 0
        continue
    
    res_matrix[Z, XY, x, y, z] +=\
        total_sci_matrix[Z, XY, x, y, z] / len_matrix[Z, XY, x, y, z]

#%%
#fname = 'total_sci_matrix_100.npy'
#total_sci_matrix = np.load(fname)

total_slice = np.zeros((80, 50))

for Z, XY, x, y, z in product(range(s0), range(s1), range(s2), range(s3), range(s4)):
    
#    if 0 <= XY < 10 and y == 4 or 10 <= XY < 20 and y == 0:
#    if 0 <= XY < 10 and y == 4:
    if 0 <= XY < 20:
        
        ind_x = (XY % 10) * 5 + x
        ind_z = Z * 5 + z
        
        total_slice[ind_z, ind_x] += res_matrix[Z, XY, x, y, z] / 100 / 10

plt.imshow(total_slice[:50, 10:40])
plt.colorbar(orientation='vertical')
ax = plt.gca()
ax.set_xticklabels((0, 0, 20, 40))
ax.set_yticklabels((0, 0, 20, 40, 60))
plt.xlabel('x, nm')
plt.ylabel('z, nm')
