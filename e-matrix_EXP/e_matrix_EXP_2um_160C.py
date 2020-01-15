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

os.chdir(mc.sim_folder + 'e-matrix_EXP')

import e_matrix_functions as emf
emf = importlib.reload(emf)

import scission_functions as sf
sf = importlib.reload(sf)

import matplotlib


n_dose = 2


#%%
def scission_probs_2CC(EE):
    
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


#%%
plt.figure(figsize=[5, 4.])

font_size = 11

matplotlib.rcParams['font.family'] = 'Times New Roman'

plt.plot(ma.EE[:237], scission_probs_2CC(ma.EE[:237]))

#plt.title('PMMA chain scission probability')
plt.xlabel('энергия налетающего электрона, эВ')
plt.ylabel('вероятность разрыва')

plt.grid()

plt.savefig('fig_1.png', dpi=600)


#%%
l_xyz = np.array((2000, 10, 900))

x_beg, y_beg, z_beg = -l_xyz[0]/2, -l_xyz[1]/2, 0
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

step_2nm = 2

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)

x_centers_2nm = (x_bins_2nm[:-1] + x_bins_2nm[1:]) / 2

bins_2nm = x_bins_2nm, y_bins_2nm, z_bins_2nm


#%%
path = '../e_DATA/EXP/'

folders = ['e_DATA_EXP_MAC', 'e_DATA_EXP_FTIAN', 'e_DATA_EXP_new', 'e_DATA_EXP_new2']


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
def get_n_electrons_1D(dose_C_cm, ly_nm, y_borders_nm):
    
    q_el_C = 1.6e-19
    L_cm = (ly_nm + y_borders_nm*2) * 1e-7
    Q_C = dose_C_cm * L_cm
    
    return int(np.round(Q_C / q_el_C))


e_matrix_val_my = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))
e_matrix_dE = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))

borders_nm = 250

## bruk2016.pdf

doses_C_cm2 = [0.05e-6, 0.2e-6, 0.87e-6]

dose_C_cm2 = doses_C_cm2[n_dose]

dose_C_cm = dose_C_cm2 / 2000

n_electrons_required = get_n_electrons_1D(dose_C_cm, l_xyz[1], borders_nm)

n_electrons = 0

y_min, y_max = y_beg - borders_nm, y_end + borders_nm


#%%
n_electrons_in_file = 10

while n_electrons < n_electrons_required:

    mu.pbar(n_electrons, n_electrons_required)
    
    now_folder_ind = rnd.randint(len(folders))
    
    n_files = 1
    
    inds = rnd.choice(len(DATA_PMMA_dE_list), size=n_files, replace=False)
    
    now_DATA_PMMA_val = np.vstack(list(DATA_PMMA_val_list[i] for i in inds))
    now_DATA_PMMA_dE = np.vstack(list(DATA_PMMA_dE_list[i] for i in inds))
    
    phi=2*np.pi*rnd.random()
    
    emf.rotate_DATA(now_DATA_PMMA_val, phi)
    emf.rotate_DATA(now_DATA_PMMA_dE, phi)
    
    x_shift, y_shift = rnd.normal(0, 200), rnd.uniform(y_min, y_max)
    
    emf.add_xy_shift_easy(now_DATA_PMMA_val, x_shift, y_shift)
    emf.add_xy_shift_easy(now_DATA_PMMA_dE, x_shift, y_shift)
    
    now_EE = now_DATA_PMMA_val[:, 4]
    scissions = rnd.rand(len(now_EE)) < sf.scission_probs_2CC_2H(now_EE)
    
    e_matrix_val_my += np.histogramdd(now_DATA_PMMA_val[:, 5:8], bins=bins_2nm,
                                  weights=scissions.astype(int))[0]
    
    e_matrix_dE += np.histogramdd(now_DATA_PMMA_dE[:, 5:8], bins=bins_2nm,
                                  weights=now_DATA_PMMA_dE[:, -1])[0]
    
    n_electrons += n_electrons_in_file * n_files


#%%
np.save('EXP_2um_160C/EXP_e_matrix_val_MY_dose' + str(n_dose) + '.npy', e_matrix_val_my)
np.save('EXP_2um_160C/EXP_e_matrix_dE_MY_dose'  + str(n_dose) + '.npy', e_matrix_dE)

print('G value =', np.sum(e_matrix_val_my) / np.sum(e_matrix_dE) * 100)

