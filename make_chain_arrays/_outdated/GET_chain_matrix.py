#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import my_functions as mf
import my_variables as mv

mf = importlib.reload(mf)
mv = importlib.reload(mv)
os.chdir(mv.sim_path_MAC + 'make_chain_matrix')

#%% chains
chain_source_dir = 'Sharma/CHAINS_cubed/'
chain_lens = np.load('Sharma/initial_L.npy')

L_list = []

particles_matrix = np.zeros((mv.n_Z, mv.n_XY, mv.n_x, mv.n_y, mv.n_z, mv.n_mon_max, 3))*np.nan
pos_matrix = np.zeros((mv.n_Z, mv.n_XY, mv.n_x, mv.n_y, mv.n_z))
#easy_mon_matrix = np.zeros((mv.n_Z_new, mv.n_XY, mv.n_x, mv.n_y,\
#    mv.n_z, mv.n_part_max, 1))*np.nan

#chain_inv_matrix = np.zeros((mv.n_chains, mv.chain_len_max, 7))*np.nan
chain_inv_matrix_short = np.zeros((mv.n_chains_short, 6000, 7))*np.nan

n_chain_short = 0

for n_chain in range(mv.n_chains):
#for n_chain in range(1):
    
    if n_chain % 1000 == 0:
        print(n_chain)
    
    chain = np.load(chain_source_dir + 'chain_shift_' + str(n_chain) + '_cubed.npy')
    
    l_chain = len(chain)
    
    if l_chain > 6000:
        continue
    
    L_list.append(l_chain)
    
    for n_mon in range(l_chain):
#    for n_mon in range(1):
        
        line = chain[n_mon, :]
                
        XY, Z, x, y, z = list(map(int, line))
        
        ## define type of monomer
        mon_type = 0
        
        if l_chain == 1:
            mon_type == -2
            
        elif n_mon == 0:
            mon_type = -1
            
        elif n_mon == l_chain - 1:
            mon_type = 1
        
        ## in case of oversize
        if Z >= mv.n_Z or XY >= mv.n_XY or x >= mv.n_x or y >= mv.n_y or z >= mv.n_z:
            
#            chain_inv_matrix[n_chain, n_mon, :-1] = np.nan
#            chain_inv_matrix[n_chain, n_mon, -1] = mon_type
            
            chain_inv_matrix_short[n_chain, n_chain_short, :-1] = np.nan
            chain_inv_matrix_short[n_chain, n_chain_short, -1] = mon_type
            
            continue
        
        pos = pos_matrix[Z, XY, x, y, z].astype(int)
        
        particles_matrix[Z, XY, x, y, z, pos, :] = n_chain, n_mon, mon_type
#        easy_mon_matrix[Z, XY, x, y, z, pos] = 1
        pos_matrix[Z, XY, x, y, z] += 1
        
#        chain_inv_matrix[n_chain, n_mon] = Z, XY, x, y, z, pos, mon_type
        
        chain_inv_matrix_short[num, n_mon] = Z, XY, x, y, z, pos, mon_type
  
#%%
#np.save('Sharma/MATRIX_particles.npy', particles_matrix)
np.save('Sharma/MATRIX_chains_inv_short.npy', chain_inv_matrix)
#np.save('Sharma/MATRIX_easy_mon.npy', easy_mon_matrix)

#%% make extended matrices
particle_matrix = np.load('Sharma/MATRIX_particles.npy')
chain_inv_matrix = np.load('Sharma/MATRIX_chains_inv.npy')

particle_matrix_ext = np.zeros((mv.n_Z_new, mv.n_XY, mv.n_x, mv.n_y, mv.n_z,\
                                mv.n_mon_max, 3))*np.nan
chain_inv_matrix_ext = np.zeros((mv.N_chains*3, mv.chain_len_max, 7))*np.nan

#%%
for i in range(3):
    
    particle_matrix_ext[mv.n_Z * i : mv.n_Z * (i+1)] =\
        particle_matrix[:, :, :, :, :, :mv.n_mon_max, :] 
    chain_inv_matrix_ext[mv.N_chains * i : mv.N_chains * (i+1)] = chain_inv_matrix

#%%
np.save('Courtney/MATRIX_particles.npy', particle_matrix_ext)
np.save('Courtney/MATRIX_chains_inv.npy', chain_inv_matrix_ext)

#%% check n_events
e_matrix = np.load('Sharma/MATRIX_e_data_02_05.npy')

cnt_0 = 0 ## total
cnt_1 = 0 ## monomer + event
cnt_2 = 0 ## monomer
cnt_3 = 0 ## event
cnt_4 = 0 ## nothing

shape = np.shape(particles_matrix)

for i0 in range(shape[0]):
    
    print(i0)
    
    for i1 in range(shape[1]):
        for i2 in range(shape[2]):
            for i3 in range(shape[3]):
                for i4 in range(shape[4]):
    
                    cnt_0 += 1
                    
                    if e_matrix[i0, i1, i2, i3, i4] > 0:
                        
                        if particles_matrix[i0, i1, i2, i3, i4] > 0:
                            cnt_1 += 1
                        else:
                            cnt_3 += 1
                    
                    else:
                        if particles_matrix[i0, i1, i2, i3, i4] > 0:
                            cnt_2 += 1
                        else:
                            cnt_4 += 1
