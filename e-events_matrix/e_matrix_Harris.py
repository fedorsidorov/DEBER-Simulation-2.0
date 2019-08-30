#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
#import copy

import numpy.random as rnd

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)

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


#%%
l_xyz = np.array((100, 100, 500))

x_beg, y_beg, z_beg = -l_xyz[0]/2, -l_xyz[0]/2, 0
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

folders = ['e_DATA_Harris_PMMA_cut_1.5e-4', 'e_DATA_Harris_PMMA_cut_1e-4',
           'Harris_cut_PMMA', 'Harris_PMMA', 'DATA_PMMA_Lenovo']

sizes = [10, 10, 1, 1, 1]


#%%
DATA = [None] * len(folders)

for now_ind in list(range(len(folders))):
    
    mu.upd_progress_bar(now_ind, len(folders))

    mult = int(100 / sizes[now_ind])
    print(mult)
    
    now_list = []
    
    now_folder = path + folders[now_ind]
    now_folder_files = os.listdir(now_folder)
    
    pos = 0
    
    
    while pos < len(now_folder_files):
        
        if pos + mult >= len(now_folder_files):
            break
        
        now_DATA_PMMA = np.load(now_folder + '/' + 'DATA_PMMA_' + str(pos) + '.npy')
        now_DATA_PMMA = now_DATA_PMMA[np.where(now_DATA_PMMA[:, 3] == 1)]
        pos += 1
        
        for _ in range(mult-1):
            
            add_DATA_PMMA = np.load(now_folder + '/' + 'DATA_PMMA_' + str(pos) + '.npy')
            add_DATA_PMMA = add_DATA_PMMA[np.where(add_DATA_PMMA[:, 3] == 1)]
            
            now_DATA_PMMA = np.vstack((now_DATA_PMMA, add_DATA_PMMA))
            
            pos += 1
        
        now_DATA_PMMA[:, 5:8] *= 1e+7
        now_list.append(now_DATA_PMMA)
        
    
    DATA[now_ind] = now_list


#%%
e_matrix_val = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))
e_matrix_dE = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))

borders_nm = 250

n_electrons_required = emf.get_n_electrons(1e-4, 100, 100, 250)
n_electrons = 0

while n_electrons < n_electrons_required:

    mu.upd_progress_bar(n_electrons, n_electrons_required)
    
    now_folder_ind = rnd.randint(len(folders))
    
    now_DATA_PMMA = rnd.choice(DATA[now_folder_ind])
    
    x_min, x_max = x_beg - borders_nm, x_end + borders_nm
    y_min, y_max = y_beg - borders_nm, y_end + borders_nm
    
    emf.rotate_DATA(now_DATA_PMMA)
#    emf.shift_DATA(now_DATA_PMMA, (x_min, x_max), (y_min, y_max))
    emf.add_xy_shift_easy(now_DATA_PMMA, rnd.uniform(x_min, x_max), rnd.uniform(y_min, y_max))
    
    
    scissions = get_scission(now_DATA_PMMA[:, 4]).astype(int)
    
    e_matrix_val += np.histogramdd(now_DATA_PMMA[:, 5:8], bins=bins_2nm,
                                   weights=scissions)[0]
    
    e_matrix_dE += np.histogramdd(now_DATA_PMMA[:, 5:8], bins=bins_2nm,
                                  weights=now_DATA_PMMA[:, -1])[0]
    
    n_electrons += 100

            
#%%
data = np.load('../e_DATA/Harris_cut_PMMA/DATA_PMMA_102.npy')


#%%
#sum_C = np.sum(e_matrix_C)
sum_C_ion = np.sum(e_matrix_C_ion)
sum_C_exc = np.sum(e_matrix_C_exc)

sum_dE = np.sum(e_matrix_dE)
        
#%%
#e_matrix_C = np.array(e_matrix_C, dtype=np.uint8)
e_matrix_C_exc = np.array(e_matrix_C_exc, dtype=np.uint8)
e_matrix_C_ion = np.array(e_matrix_C_ion, dtype=np.uint8)

print('e_matrix size, Mb:', e_matrix_C_exc.nbytes / 1024**2)

#%%
#np.save('MATRIX_Harris_100uC_C.npy', e_matrix_C)
np.save('MATRIX_Harris_100uC_C_exc_bad.npy', e_matrix_C_exc)
np.save('MATRIX_Harris_100uC_C_ion_bad.npy', e_matrix_C_ion)

np.save('MATRIX_dE_Harris_100uC_bad.npy', e_matrix_dE)

#%%
#e_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/MATRIX_6e-5_pC_cm2_C_exc.npy')

#%% drawing
plt.figure()
plt.semilogy(x_grid_2nm, np.sum(e_matrix_dE[:, 25, :], axis=1))

plt.xlabel('x, nm')
plt.ylabel('N events')
plt.title('Event coordinate distribution, 2 nm')
plt.legend()
plt.grid()
plt.show()
#plt.savefig('LOG events 2nm.png', dpi=300)

#%% Test E development (kyser1975)
e_matrix_dE_avg = np.sum(e_matrix_dE[:, :, :], axis=1)/50
e_matrix_dE_avg_develop = e_matrix_dE_avg - 6.8 * 2**3
e_matrix_dE_avg_develop_mono = e_matrix_dE_avg_develop / np.abs(e_matrix_dE_avg_develop)

plt.imshow(e_matrix_dE_avg_develop_mono.transpose())
plt.show()
