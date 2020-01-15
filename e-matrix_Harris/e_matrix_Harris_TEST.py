#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

import numpy.random as rnd

import my_constants as mc
import my_utilities as mu
import my_arrays_Dapor as ma

mc = importlib.reload(mc)
mu = importlib.reload(mu)
ma = importlib.reload(ma)

os.chdir(mc.sim_folder + 'e-matrix_Harris')

import e_matrix_functions as emf
emf = importlib.reload(emf)

import scission_functions as sf
sf = importlib.reload(sf)


#%%
path = '../e_DATA/Harris/'

folders = ['e_DATA_Harris_MY_PRE', 'e_DATA_Harris_MY_MAC', 'e_DATA_Harris_MY_FTIAN']


#%%
DATA_PMMA_list = []
DATA_PMMA_val_list = []
DATA_PMMA_dE_list = []
DATA_PMMA_dE_total_list = []


for now_ind in range(len(folders)):
    
    mu.pbar(now_ind, len(folders))
    
    now_folder = path + folders[now_ind]
    now_folder_files = os.listdir(now_folder)
    
    pos = 0
    
    for file in now_folder_files:
        
        if 'DS' in file:
            continue
        
        now_DATA_PMMA = np.load(now_folder + '/' + file)

        now_DATA_PMMA[:, 5:8] *= 1e+7
        
        DATA_PMMA_list.append(now_DATA_PMMA)
        
        now_DATA_PMMA_val = now_DATA_PMMA[np.where(now_DATA_PMMA[:, 3] == 1)]
        DATA_PMMA_val_list.append(now_DATA_PMMA_val)
        
        now_DATA_PMMA_dE = copy.deepcopy(now_DATA_PMMA_val)
        now_DATA_PMMA_dE[np.where(now_DATA_PMMA_dE[:, 3] == 1)[0], -1] = ma.PMMA_E_bind
        DATA_PMMA_dE_list.append(now_DATA_PMMA_dE)


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
e_matrix_shape = (len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1)

e_matrix_val = np.zeros(e_matrix_shape)
e_matrix_dE = np.zeros(e_matrix_shape)
e_matrix_dE_total = np.zeros(e_matrix_shape)

borders_nm = 250

x_min, x_max = x_beg - borders_nm, x_end + borders_nm
y_min, y_max = y_beg - borders_nm, y_end + borders_nm

n_electrons_required = emf.get_n_electrons_2D(1e-4, lx, ly, borders_nm)
n_electrons = 0


scission_probs_gryz = np.load('../all_Gryzinski/probs_gryzinski.npy')


while n_electrons < n_electrons_required:

    mu.pbar(n_electrons, n_electrons_required)
    
    now_folder_ind = rnd.randint(len(folders))
    
    electrons_in_file = 10
    n_files = 100
    
    inds = rnd.choice(len(DATA_PMMA_dE_list), size=n_files, replace=False)
    
    now_DATA_PMMA_val = np.vstack(list(DATA_PMMA_val_list[i] for i in inds))
    now_DATA_PMMA_dE = np.vstack(list(DATA_PMMA_dE_list[i] for i in inds))
    
    phi=2*np.pi*rnd.random()
    
    emf.rotate_DATA(now_DATA_PMMA_val, phi)
    emf.rotate_DATA(now_DATA_PMMA_dE, phi)
        
    x_shift, y_shift = rnd.uniform(x_min, x_max), rnd.uniform(y_min, y_max)
    
    emf.add_xy_shift_easy(now_DATA_PMMA_val, x_shift, y_shift)
    emf.add_xy_shift_easy(now_DATA_PMMA_dE, x_shift, y_shift)
    
    ## !!! ##
    now_EE = now_DATA_PMMA_val[:, 4]
    
    scission_probs = np.interp(now_EE, mc.EE, scission_probs_gryz)
    
    scissions = rnd.rand(len(now_EE)) < scission_probs
#    scissions = rnd.rand(len(now_EE)) < sf.scission_probs_2CC(now_EE)
#    scissions = rnd.rand(len(now_EE)) < sf.scission_probs_2CC_ester(now_EE)
#    scissions = rnd.rand(len(now_EE)) < sf.scission_probs_2CC_ester_H(now_EE)
#    scissions = rnd.rand(len(now_EE)) < sf.scission_probs_2CC_3H(now_EE)
#    scissions = rnd.rand(len(now_EE)) < sf.scission_probs_2CC_2H(now_EE)
#    scissions = rnd.rand(len(now_EE)) < sf.scission_probs_2CC_1p5H(now_EE)
    ## !!! ##
    
    e_matrix_val += np.histogramdd(now_DATA_PMMA_val[:, 5:8], bins=bins_2nm,
                                   weights=scissions.astype(int))[0]
    
    e_matrix_dE += np.histogramdd(now_DATA_PMMA_dE[:, 5:8], bins=bins_2nm,
                                  weights=now_DATA_PMMA_dE[:, -1])[0]
    
    n_electrons += n_files * electrons_in_file


#%%
folder = '2C-C+1.5H/'

np.save(folder + 'e-matrix_val.npy', e_matrix_val)
np.save(folder + 'e-matrix_dE.npy', e_matrix_dE)


#%%
print('G(S) =', np.sum(e_matrix_val) / np.sum(e_matrix_dE) * 100)

