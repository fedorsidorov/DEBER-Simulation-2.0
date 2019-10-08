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
l_xyz = np.array((100, 100, 100))

space = 50
linewidth = 2

x_beg, y_beg, z_beg = (-l_xyz[0]/2, 0, 0)
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

#%%
e_matrix_dE = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))

e_matrix_C = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))
e_matrix_C_exc = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))
e_matrix_C_ion = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))

source_dir = mv.sim_path_MAC + 'e_DATA/DATA_Pn_10keV_100nm/'

#%%
n_files_total = 600

Q = 100e-6 ## C/cm^2
q = 1.6e-19 ## C

n_files_line = int(Q * 2e-7 * (100 + space*2)*1e-7 / q / 100)

#%%
#n_files_required = int(20 * (100 + 50*2) * (1e-7)**2 * 100e-6 / 1.6e-19 / 100)

n_events = 0

x_positions = [-80, -60, -40, -20, 0, 20, 40, 60, 80]
#x_positions = [0]

for x_pos in x_positions:
    
    print('x_pos =', x_pos)
    
    file_nums = mf.choice(range(n_files_total), n_files_line, replace=False)
    
    files_used = 0
    
    for i in file_nums:
        
        files_used += 1
        
        mf.upd_progress_bar(files_used, n_files_line)
        
        filename = 'DATA_Pn_' + str(i) + '.npy'
        
        DATA_Pn = np.load(source_dir + filename)
        
        n_rotations = 1
        
        for _ in range(n_rotations):
            
            now_DATA_Pn = copy.deepcopy(DATA_Pn)
            
            emf.rotate_DATA(now_DATA_Pn)
            
            n_events += len(now_DATA_Pn)
            
            emf.shift_DATA(now_DATA_Pn, (x_pos - linewidth/2, x_pos + linewidth/2),\
                           (y_beg-space, y_end+space))
            
            now_DATA_Pn = now_DATA_Pn[np.where(np.logical_and(
                    now_DATA_Pn[:, mi.e_dE] > 1,
                    now_DATA_Pn[:, mi.e_dE] < 1000))]
    
            e_matrix_dE += np.histogramdd(now_DATA_Pn[:, 5:8], bins=bins_2nm,
                                          weights=now_DATA_Pn[:, mi.e_dE])[0]
            
            now_DATA_Pn_C = now_DATA_Pn[np.where(now_DATA_Pn[:, mi.atom_id] == mi.C)]
            
            now_DATA_Pn_C_exc = now_DATA_Pn[np.where(np.logical_and(
                    now_DATA_Pn[:, mi.atom_id] == mi.C,
                    now_DATA_Pn[:, mi.coll_id] == mi.exc))]
            
            now_DATA_Pn_C_ion = now_DATA_Pn[np.where(np.logical_and(
                    now_DATA_Pn[:, mi.atom_id] == mi.C,
                    now_DATA_Pn[:, mi.coll_id] > mi.exc))]
            
            e_matrix_C += np.histogramdd(now_DATA_Pn_C[:, 5:8], bins=bins_2nm)[0]
            e_matrix_C_exc += np.histogramdd(now_DATA_Pn_C_exc[:, 5:8], bins=bins_2nm)[0]
            e_matrix_C_ion += np.histogramdd(now_DATA_Pn_C_ion[:, 5:8], bins=bins_2nm)[0]
            
#%%
#sum_C     = np.sum(e_matrix_C)
#sum_C_ion = np.sum(e_matrix_C_ion)
#sum_C_exc = np.sum(e_matrix_C_exc)

sum_dE = np.sum(e_matrix_dE)
        
#%%
e_matrix_C = np.array(e_matrix_C, dtype=np.uint8)
e_matrix_C_exc = np.array(e_matrix_C_exc, dtype=np.uint8)
e_matrix_C_ion = np.array(e_matrix_C_ion, dtype=np.uint8)

print('e_matrix size, Mb:', e_matrix_C_exc.nbytes / 1024**2)

np.save('MATRIX_Aktary_100uC_C_lines.npy',     e_matrix_C)
np.save('MATRIX_Aktary_100uC_C_exc_lines.npy', e_matrix_C_exc)
np.save('MATRIX_Aktary_100uC_C_ion_lines.npy', e_matrix_C_ion)

np.save('MATRIX_Aktary_100uC_dE_lines.npy', e_matrix_dE)

#%%
e_matrix_dE = np.load(mv.sim_path_MAC + 'MATRIXES/Aktary/MATRIX_Aktary_100uC_dE_lines.npy')

eps_xz = np.sum(e_matrix_dE, axis=(1)) / (100 * 2 * 2 * 1e-21)
Mf = 1 / (1/950e+3 + 1.9e-2 * eps_xz / (1.19 * 6.02e+23))

plt.figure()
plt.imshow(np.log10(Mf).transpose())
plt.colorbar()
plt.title('LOG chain M after exposure')
plt.xlabel('x/2, nm')
plt.ylabel('z/2, nm')
plt.show()
plt.savefig('Aktary LOG chain M after.png', dpi=300)

#%% development
R0 = 84
beta = 3.14e+8
n = 1.5

R = R0 + beta / Mf**n

plt.figure()
plt.imshow(R.transpose())
plt.colorbar()
plt.title('Resist dissolution rate, A/min')
plt.xlabel('x/2, nm')
plt.ylabel('z/2, nm')
plt.show()
plt.savefig('Aktary resist dissolution rate.png', dpi=300)

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
