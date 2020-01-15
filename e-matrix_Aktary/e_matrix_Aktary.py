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

os.chdir(mc.sim_folder + 'e-matrix_Aktary')

import e_matrix_functions as emf
emf = importlib.reload(emf)


#%%
def get_w_scission(EE):
    
    result = np.zeros(len(EE))
    
    result = np.ones(len(EE)) * 4/40
    result[np.where(EE < 815 * 0.0103)] = 4/(40 - 8)
    result[np.where(EE < 420 * 0.0103)] = 4/(40 - 8 - 4)
    result[np.where(EE < 418 * 0.0103)] = 4/(40 - 8 - 4 - 12)
    result[np.where(EE < 406 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4)
    result[np.where(EE < 383 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2)
    result[np.where(EE < 364 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4)
    result[np.where(EE < 356 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4 - 2)
    result[np.where(EE < 354 * 0.0103)] = 0
    
    return result


def get_scissions(EE):
    
    return rnd.rand(len(EE)) < get_w_scission(EE)


def get_scissions_ones(EE):
    
    return np.ones(len(EE))


#%%
path = '../e_DATA/'

folders = ['Aktary']


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
        
        now_DATA_PMMA_dE_total = copy.deepcopy(now_DATA_PMMA)
        now_DATA_PMMA_dE_total[np.where(now_DATA_PMMA_dE_total[:, 3] == 1)[0], -1] =\
            ma.PMMA_E_bind
        DATA_PMMA_dE_total_list.append(now_DATA_PMMA_dE_total)
        
        now_DATA_PMMA_val = now_DATA_PMMA[np.where(now_DATA_PMMA[:, 3] == 1)]
        DATA_PMMA_val_list.append(now_DATA_PMMA_val)
        
        now_DATA_PMMA_dE = copy.deepcopy(now_DATA_PMMA_val)
        now_DATA_PMMA_dE[np.where(now_DATA_PMMA_dE[:, 3] == 1)[0], -1] = ma.PMMA_E_bind
        DATA_PMMA_dE_list.append(now_DATA_PMMA_dE)


#%%
l_xyz = np.array((100, 100, 100))
lx, ly, lz = l_xyz

x_beg, y_beg, z_beg = -lx/2, -ly/2, 0
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

#step_2nm = 2
step_2nm = 1

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)

bins_2nm = x_bins_2nm, y_bins_2nm, z_bins_2nm

##
e_matrix_shape = (len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1)

e_matrix_val = np.zeros(e_matrix_shape)
e_matrix_dE = np.zeros(e_matrix_shape)
e_matrix_dE_total = np.zeros(e_matrix_shape)

#borders_nm = 250
borders_nm = 100

y_min, y_max = y_beg - borders_nm, y_end + borders_nm

#n_electrons_required = emf.get_n_electrons_2D(1e-4, lx, ly, borders_nm)
#n_electrons_required = 10000 * 3
n_electrons_required = 12500 * 3

#shift_array = np.zeros((150000, 2))

#i = 0


for pos in (-40, -20, 0, 20, 40):
    
    print(pos)
    
    n_electrons = 0
    
    while n_electrons < n_electrons_required:
    
        mu.pbar(n_electrons, n_electrons_required)
        
        now_folder_ind = rnd.randint(len(folders))
        
        electrons_in_file = 10
        n_files = 100
        
        inds = rnd.choice(len(DATA_PMMA_dE_list), size=n_files, replace=False)
        
        now_DATA_PMMA_val = np.vstack(list(DATA_PMMA_val_list[i] for i in inds))
        now_DATA_PMMA_dE = np.vstack(list(DATA_PMMA_dE_list[i] for i in inds))
        now_DATA_PMMA_dE_total = np.vstack(list(DATA_PMMA_dE_total_list[i] for i in inds))
        
        phi=2*np.pi*rnd.random()
        
        emf.rotate_DATA(now_DATA_PMMA_val, phi)
        emf.rotate_DATA(now_DATA_PMMA_dE, phi)
        emf.rotate_DATA(now_DATA_PMMA_dE_total, phi)
            
        x_shift = pos
        y_shift = rnd.uniform(y_min, y_max)
        
#        shift_array[i] = x_shift, y_shift
#        i += 1
        
        emf.add_xy_shift_easy(now_DATA_PMMA_val, x_shift, y_shift)
        emf.add_xy_shift_easy(now_DATA_PMMA_dE, x_shift, y_shift)
        emf.add_xy_shift_easy(now_DATA_PMMA_dE_total, x_shift, y_shift)
        
        ## !!! ##
        scissions = get_scissions(now_DATA_PMMA_val[:, 4]).astype(int)
        ## !!! ##
        
        e_matrix_val += np.histogramdd(now_DATA_PMMA_val[:, 5:8], bins=bins_2nm,
                                       weights=scissions)[0]
        
        e_matrix_dE += np.histogramdd(now_DATA_PMMA_dE[:, 5:8], bins=bins_2nm,
                                      weights=now_DATA_PMMA_dE[:, -1])[0]
        
        e_matrix_dE_total += np.histogramdd(now_DATA_PMMA_dE_total[:, 5:8], bins=bins_2nm,
                                      weights=now_DATA_PMMA_dE_total[:, -1])[0]
        
        n_electrons += n_files * electrons_in_file



np.save('Aktary_e_matrix_val_1nm_100_10.npy', e_matrix_val)
#np.save('Aktary_e_matrix_dE_1nm_100.npy', e_matrix_dE)
#np.save('Aktary_e_matrix_dE_total_1nm_100.npy', e_matrix_dE_total)


#%%
print('G value =', np.sum(e_matrix_val) / np.sum(e_matrix_dE) * 100)


#%%
G_mat = np.sum(e_matrix_val, axis=1) / np.sum(e_matrix_dE, axis=1) * 100

