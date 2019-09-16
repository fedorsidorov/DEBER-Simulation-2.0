#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
#import copy

import numpy.random as rnd

import my_constants as mc
import my_utilities as mu
import my_arrays_Dapor as ma

mc = importlib.reload(mc)
mu = importlib.reload(mu)
ma = importlib.reload(ma)

os.chdir(mc.sim_folder + 'e-events_matrix')

import e_matrix_functions as emf
emf = importlib.reload(emf)


#%%
def get_w_scission(EE):
    
    result = np.zeros(len(EE))
    
    result = (64-EE) * (4/8-4/40) / (64-14) + 4/40
    result[np.where(EE > 64)] = 4/40
    result[np.where(EE < 3.6)] = 0
    
    return result


def get_scission(EE):
    
    return rnd.rand(len(EE)) < get_w_scission(EE)


def get_w_scission_easy(EE):
    
    result = np.zeros(len(EE))
    
    result = np.ones(len(EE)) * 4/40
    result[np.where(EE < 3.6)] = 0
    
    return result


def get_scission_easy(EE):
    
    return rnd.rand(len(EE)) < get_w_scission_easy(EE)


#%%
l_xyz = np.array((100, 100, 500))

x_beg, y_beg, z_beg = -l_xyz[0]/2, -l_xyz[1]/2, 0
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

step_2nm = 2

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)

bins_2nm = x_bins_2nm, y_bins_2nm, z_bins_2nm

#x_grid_2nm = (x_bins_2nm[:-1] + x_bins_2nm[1:]) / 2
#y_grid_2nm = (y_bins_2nm[:-1] + y_bins_2nm[1:]) / 2
#z_grid_2nm = (z_bins_2nm[:-1] + z_bins_2nm[1:]) / 2


#%%
path = '../e_DATA/'

folders = ['e_DATA_Harris_my_E_bind_inel']


#%%
DATA_PMMA_val_list = []
DATA_PMMA_dE_list = []


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

        now_DATA_PMMA_val = now_DATA_PMMA[np.where(now_DATA_PMMA[:, 3] == 1)]
        
#        now_DATA_PMMA_E_loss = now_DATA_PMMA[np.where(now_DATA_PMMA[])]
        now_DATA_PMMA_dE = now_DATA_PMMA_val
        
        now_DATA_PMMA_dE[np.where(now_DATA_PMMA_dE[:, 3] == 1)[0], -1] = ma.PMMA_E_bind
        
        DATA_PMMA_val_list.append(now_DATA_PMMA_val)
        DATA_PMMA_dE_list.append(now_DATA_PMMA_dE)


#%%
e_matrix_val = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))
e_matrix_val_easy = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))
e_matrix_dE = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))

borders_nm = 250

n_electrons_required = emf.get_n_electrons(1e-4, 100, 100, 250)
n_electrons = 0

x_min, x_max = x_beg - borders_nm, x_end + borders_nm
y_min, y_max = y_beg - borders_nm, y_end + borders_nm


while n_electrons < n_electrons_required:

    mu.pbar(n_electrons, n_electrons_required)
    
    now_folder_ind = rnd.randint(len(folders))
    
    inds = rnd.choice(len(DATA_PMMA_dE_list), size=100, replace=False)
    
    now_DATA_PMMA_val = np.vstack(list(DATA_PMMA_val_list[i] for i in inds))
    now_DATA_PMMA_dE = np.vstack(list(DATA_PMMA_dE_list[i] for i in inds))
    
    phi=2*np.pi*rnd.random()
    
    emf.rotate_DATA(now_DATA_PMMA_val, phi)
    emf.rotate_DATA(now_DATA_PMMA_dE, phi)
    
#    emf.shift_DATA(now_DATA_PMMA, (x_min, x_max), (y_min, y_max))
    
    x_shift, y_shift = rnd.uniform(x_min, x_max), rnd.uniform(y_min, y_max)
    
    emf.add_xy_shift_easy(now_DATA_PMMA_val, x_shift, y_shift)
    emf.add_xy_shift_easy(now_DATA_PMMA_dE, x_shift, y_shift)
    
    scissions = get_scission(now_DATA_PMMA_val[:, 4]).astype(int)
    scissions_easy = get_scission_easy(now_DATA_PMMA_val[:, 4]).astype(int)
    
    
    e_matrix_val += np.histogramdd(now_DATA_PMMA_val[:, 5:8], bins=bins_2nm,
                                   weights=scissions)[0]
    
    e_matrix_val_easy += np.histogramdd(now_DATA_PMMA_val[:, 5:8], bins=bins_2nm,
                                   weights=scissions_easy)[0]
    
    e_matrix_dE += np.histogramdd(now_DATA_PMMA_dE[:, 5:8], bins=bins_2nm,
                                  weights=now_DATA_PMMA_dE[:, -1])[0]
    
    n_electrons += 1000


#%%
np.save('Harris_e_matrix_val_Dapor_NEW.npy', e_matrix_val)
np.save('Harris_e_matrix_val_Dapor_NEW_easy.npy', e_matrix_val_easy)

np.save('Harris_e_matrix_dE_Dapor_NEW_1.npy', e_matrix_dE)


#%%
e_mat = np.load('Harris_e_matrix_val_Dapor_NEW.npy')
dE_mat = np.load('Harris_e_matrix_dE_Dapor_NEW.npy')


#%%
print('G value =', np.sum(e_matrix_val_easy) / np.sum(e_matrix_dE) * 100)





