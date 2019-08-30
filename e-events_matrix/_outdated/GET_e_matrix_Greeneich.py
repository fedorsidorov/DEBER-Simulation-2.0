#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

import my_functions as mf
mf = importlib.reload(mf)

import my_variables as mv
mv = importlib.reload(mv)

import my_indexes as mi
mi = importlib.reload(mi)

os.chdir(mv.sim_path_MAC + 'make_e_matrix')

import e_matrix_functions as emf
emf = importlib.reload(emf)

#%% calculate required files amount
l_xyz = np.array((100, 100, 300))

x_beg, y_beg, z_beg = (-l_xyz[0]/2, -l_xyz[0]/2, 0)
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

step_2nm = 2

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)

bins_2nm = x_bins_2nm, y_bins_2nm, z_bins_2nm

x_grid_2nm = (x_bins_2nm[:-1] + x_bins_2nm[1:]) / 2
y_grid_2nm = (y_bins_2nm[:-1] + y_bins_2nm[1:]) / 2
z_grid_2nm = (z_bins_2nm[:-1] + z_bins_2nm[1:]) / 2

Lx = 400
Ly = 400

#%%
e_matrix_C_exc = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))
e_matrix_C_ion = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))
e_matrix_dE    = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))

source_dir = mv.sim_path_MAC + 'e_DATA/DATA_Pn_20keV_300nm/'

#%%
n_files_total = 181
n_files_required = int(emf.get_n_electrons(1e-4, Lx, Ly) / 100)

#%%
n_events = 0

for i in range(n_files_required):
    
    mf.upd_progress_bar(i, n_files_required)
    
    filename = 'DATA_Pn_' + str(i % n_files_total) + '.npy'
    
    now_DATA_Pn = np.load(source_dir + filename)
    
    emf.rotate_DATA(now_DATA_Pn)
    
    n_events += len(now_DATA_Pn)
    
    x_min, x_max = -Lx/2, Lx/2
    y_min, y_max = -Ly/2, Ly/2
    
    emf.shift_DATA(now_DATA_Pn, (x_min, x_max), (y_min, y_max))

    e_matrix_dE += np.histogramdd(now_DATA_Pn[:, 5:8], bins=bins_2nm,
                                  weights=now_DATA_Pn[:, mi.e_dE])[0]
    
    now_DATA_Pn_C_exc = now_DATA_Pn[np.where(np.logical_and(
            now_DATA_Pn[:, mi.atom_id] == mi.C,
            now_DATA_Pn[:, mi.coll_id] == mi.exc))]
    
    now_DATA_Pn_C_ion = now_DATA_Pn[np.where(np.logical_and(
            now_DATA_Pn[:, mi.atom_id] == mi.C,
            now_DATA_Pn[:, mi.coll_id] > mi.exc))]
    
    e_matrix_C_exc += np.histogramdd(now_DATA_Pn_C_exc[:, 5:8], bins=bins_2nm)[0]
    e_matrix_C_ion += np.histogramdd(now_DATA_Pn_C_ion[:, 5:8], bins=bins_2nm)[0]
            
#%%
sum_C_ion = np.sum(e_matrix_C_ion)
sum_C_exc = np.sum(e_matrix_C_exc)
sum_dE = np.sum(e_matrix_dE)

#%%
e_matrix_C_exc = np.array(e_matrix_C_exc, dtype=np.uint8)
e_matrix_C_ion = np.array(e_matrix_C_ion, dtype=np.uint8)

print('e_matrix size, Mb:', e_matrix_C_exc.nbytes / 1024**2)

#%%
np.save('MATRIX_Greeneich_100uC_C_exc.npy', e_matrix_C_exc)
np.save('MATRIX_Greeneich_100uC_C_ion.npy', e_matrix_C_ion)
np.save('MATRIX_dE_Greeneich_100uC.npy', e_matrix_dE)

#%%
e_matrix_dE = np.load('MATRIX_dE_Greeneich_100uC.npy')

e_matrix_dE_xy = np.sum(e_matrix_dE, axis=2) / (2 * 2 * 300 * 1e-21)
e_matrix_dE_z = np.sum(e_matrix_dE[:, :, :], axis=(0, 1)) / (100 * 100 * 2 * 1e-21)

dE_avg = np.average(e_matrix_dE_z)

#%%
lambda_emp = 1e-4 / 1.6e-19 * 20e+3 / (4.6e-6 / 1.19 * 20**1.75) * 0.74

#%% drawing
#plt.figure()
plt.semilogy(z_grid_2nm, e_matrix_dE_z)

plt.xlabel('z, nm')
plt.ylabel('energy density, eV/cm^3')
plt.title('Energy dissipation, 2 nm')
plt.legend()
plt.grid()
plt.show()
#plt.savefig('LOG events 2nm.png', dpi=300)

#%% calculate Mf
Mf = 1 / (1/100e+3 + 1.9e-2 * e_matrix_dE_z / (1.19 * 6.02e+23))

Mf_avg = np.average(Mf)

#%% development
R0 = 84
beta = 3.14e+8
n = 1.5

R = R0 + beta / Mf_avg**n

#%% test for Mf = 3e+3
Mf = 0.5e+3
Mn = 950e+3
g = 1.9e-2
rho = 1.19
Na = 6.02e+23

eps = (1/Mf - 1/Mn) * rho * Na / g




