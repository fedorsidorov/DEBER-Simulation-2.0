#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

import numpy.random as rnd

import my_constants as mc
import my_utilities as mu
import scission_functions_2020 as sf
import e_matrix_functions as emf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
sf = importlib.reload(sf)
emf = importlib.reload(emf)

os.chdir(os.path.join(mc.sim_folder, 'e-matrix_Harris'))


#%%
path = os.path.join(mc.sim_folder, 'e_DATA', 'Harris')

folders = ['Harris_2020_MAC_0_G', 'Harris_2020_MAC_1_G', 'Harris_2020_UBU_0_G']


#%%
DATA_PMMA_list = []


for now_ind in range(len(folders)):
    
    mu.pbar(now_ind, len(folders))
    
    now_folder = os.path.join(path, folders[now_ind])
    now_folder_files = os.listdir(now_folder)
    
    pos = 0
    
    for file in now_folder_files:
        
        if 'DS' in file:
            continue
        
        now_DATA_PMMA = np.load(os.path.join(now_folder, file))
        now_DATA_PMMA[:, 5:8] *= 1e+7
        DATA_PMMA_list.append(now_DATA_PMMA)


#%%
l_xyz = np.array((100, 100, 500))
lx, ly, lz = l_xyz

x_beg, y_beg, z_beg = -lx/2, -ly/2, 0
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

step_2nm = 2

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)

bins_2nm = x_bins_2nm, y_bins_2nm, z_bins_2nm


#%%
def get_scissions(b_map_sc, now_DATA_PMMA):
    
    scissions = np.zeros(len(now_DATA_PMMA))
    
    
    for b in b_map_sc:
        
        pr_ind = 10 + np.where(sf.Eb_Nel[:, 0] == sf.MMA_bonds[b][0])[0][0]
        
#        inds = np.where(now_DATA_PMMA[:, 3] == pr_ind)[0]
        inds = np.where(np.abs(now_DATA_PMMA[:, 3]) == pr_ind)[0] ## ABS!!
        
        scissions[inds] = rnd.rand(len(inds)) < np.ones(len(inds)) *\
                b_map_sc[b] / sf.MMA_bonds[b][1]
    
    
    return scissions


#%%
#bond = "C-C2"
#bond = "C-C'"
#bond = "C-C3"
bond = "C3-H"

step = 0.2


for w in np.arange(step, sf.MMA_bonds[bond][1] + step, step):
    
    bond_dict_sc = {bond: w}
    
    print(bond_dict_sc)
    
    e_matrix_shape = (len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1)
    
    e_matrix_ss = np.zeros(e_matrix_shape)
    e_matrix_dE = np.zeros(e_matrix_shape)
    
    borders_nm = 250
    
    x_min, x_max = x_beg - borders_nm, x_end + borders_nm
    y_min, y_max = y_beg - borders_nm, y_end + borders_nm
    
    n_electrons_required = emf.get_n_electrons_2D(1e-4, lx, ly, borders_nm)
    n_electrons = 0
    
    
    while n_electrons < n_electrons_required:
    
        mu.pbar(n_electrons, n_electrons_required)
        
        electrons_in_file = 10
        n_files = 100
        
        inds = rnd.choice(len(DATA_PMMA_list), size=n_files, replace=False)
        
        now_DATA_PMMA = np.vstack(list(DATA_PMMA_list[i] for i in inds))
        
        phi=2*np.pi*rnd.random()
        
        emf.rotate_DATA(now_DATA_PMMA, phi)
            
        x_shift, y_shift = rnd.uniform(x_min, x_max), rnd.uniform(y_min, y_max)
        
        emf.add_xy_shift_easy(now_DATA_PMMA, x_shift, y_shift)
        
        scissions = get_scissions(bond_dict_sc, now_DATA_PMMA)
        
        e_matrix_ss += np.histogramdd(now_DATA_PMMA[:, 5:8], bins=bins_2nm,
                                      weights=scissions.astype(int))[0]
        
        e_matrix_dE += np.histogramdd(now_DATA_PMMA[:, 5:8], bins=bins_2nm,
                                      weights=now_DATA_PMMA[:, 8])[0]
        
        n_electrons += n_files * electrons_in_file
    
    
    np.save(os.path.join('sweep', bond, 'Harris_ss_' + bond + '_' + str(w) + '.npy'), e_matrix_ss)
    np.save(os.path.join('sweep', bond, 'Harris_dE_' + bond + '_' + str(w) + '.npy'), e_matrix_dE)

