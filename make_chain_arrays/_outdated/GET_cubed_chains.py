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

#%%
## cube parameters
cube_size = 10.
cell_size = 2.

n_cells = cube_size / cell_size

eps = 1e-3

## min and max coordinates
x_min, x_max = 0., 100.
y_min, y_max = 0., 100.
z_min, z_max = 0., 122.

## cubes numbers
n_x = int(np.round((x_max - x_min) / cube_size))
n_y = int(np.round((y_max - y_min) / cube_size))
n_z = int(np.round((z_max - z_min) / cube_size))

## cubes number
n_xy = n_x * n_y
n_total = n_x * n_y * n_z

## directories
source_dir = mv.sim_path_MAC + 'CHAINS/CHAINS_950K_122nm_10k_shifted/'
dest_dir = mv.sim_path_MAC + 'CHAINS/CHAINS_950K_122nm_10k_cubed/'

#%%
N_chains = 11701

#N0 = 0
#N1 = N0 + 2000

for n in range(N_chains):
#for n in range(N0, N_chains):
    
    if n % 1000 == 0:
        print(n, 'chains are cubed')
    
    fname = 'chain_shift_' + str(n) + '.npy'
    
    chain_arr = np.load(source_dir + fname)
    chain_arr_cubed = np.zeros((len(chain_arr), 5))
    
    ## write DATA_Pn_cut to total_arr
    for i in range(len(chain_arr)):
        
        x, y, z = chain_arr[i, :]
        
        ## if monomer is out of range, set exceeding indices
        if x < x_min or  x > x_max or y < y_min or y > y_max:
            chain_arr_cubed[i, :] = n_xy, n_z, n_cells, n_cells, n_cells
            continue
        
        ## x, y and z shifts in nm
        x_coord = x - x_min
        y_coord = y - y_min
        z_coord = z - z_min
        
        ## get cube coordinates
        cube_x = np.floor(x_coord / cube_size)
        cube_y = np.floor(y_coord / cube_size)
        cube_z = np.floor(z_coord / cube_size)
        
        ## get cube position in xy plane
        cube_xy = int(cube_x + cube_y * n_x)
        
        if cube_xy >= n_xy:
            print('!!!!!')
            print(x, y, z)
        
        ## get cell coordinates in cube
        cell_x = int(np.floor((x_coord - cube_x * cube_size) / cell_size))
        cell_y = int(np.floor((y_coord - cube_y * cube_size) / cell_size))
        cell_z = int(np.floor((z_coord - cube_z * cube_size) / cell_size))
        
        ## write cell coordinates into total_arr
        chain_arr_cubed[i, :] = cube_xy, cube_z, cell_x, cell_y, cell_z
    
    np.save(dest_dir + fname.replace('.npy', '_cubed.npy'), chain_arr_cubed)
