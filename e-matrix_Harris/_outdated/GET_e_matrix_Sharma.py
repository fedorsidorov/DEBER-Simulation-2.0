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
#n_files = emf.get_n_files_with_50nm_borders(500e-12, 100)

l_xyz = np.array((100, 100, 160))

space = 50
beam_d = 1

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
e_matrix = np.zeros((len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1))
source_dir = mv.sim_path_MAC + 'e_DATA/DATA_Pn_25keV_160nm/'

#%%
n_files = emf.get_n_files_with_50nm_borders_S(6e-5, 100, 100)

n_files_total = 544
n_files_used = 500

n_events = 0

#%%
for i in range(n_files_used):
    
    mf.upd_progress_bar(i, n_files_used)
    
    DATA_Pn = np.load(source_dir + 'DATA_Pn_' + str(i) + '.npy')
    
    n_rotations = 2
    
    for _ in range(n_rotations):
        
        now_DATA_Pn = copy.deepcopy(DATA_Pn)
        
        emf.rotate_DATA(now_DATA_Pn)
        
        n_events += len(now_DATA_Pn)
        
        emf.shift_DATA(now_DATA_Pn, (-space+x_beg, x_end+space),\
                                    (-space+y_beg, y_end+space))
        
        now_DATA_Pn_C_exc = now_DATA_Pn[np.where(np.logical_and(
                now_DATA_Pn[:, mi.atom_id] == mi.C,
                now_DATA_Pn[:, mi.coll_id] == mi.exc))]
        
        e_matrix += np.histogramdd(now_DATA_Pn_C_exc[:, 5:8], bins=bins_2nm)[0]
        
#%%
e_matrix_uint16 = np.array(e_matrix, dtype=np.uint16)
print('e_matrix size, Mb:', e_matrix_uint16.nbytes / 1024**2)
np.save(mv.sim_path_MAC + 'MATRIXES/MATRIX_6e-5_pC_cm2_C_exc.npy', e_matrix_uint16)

#%%
e_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/MATRIX_6e-5_pC_cm2_C_exc.npy')

#%% drawing
plt.figure()
plt.semilogy(x_grid_2nm, np.sum(e_matrix[:, 25, :], axis=1), label='6e-5 C/cm$^2$')
plt.xlabel('x, nm')
plt.ylabel('N events')
plt.title('Event coordinate distribution, 2 nm')
plt.legend()
plt.grid()
plt.show()
#plt.savefig('LOG events 2nm.png', dpi=300)

#%%
plt.figure()
e_matrix_xz = np.sum(e_matrix, axis=1)
plt.plot(x_grid_2nm, np.sum(e_matrix_xz, axis=1), label='5e-5 C/cm$^2$')
plt.xlabel('x, nm')
plt.ylabel('N events')
plt.title('Event coordinate distribution, 100 nm')
plt.legend()
plt.grid()
plt.show()
#plt.savefig('events 100nm.png', dpi=300)
