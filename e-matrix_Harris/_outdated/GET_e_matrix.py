#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt

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
n_files = emf.get_n_files_with_50nm_borders_L(500e-12, 100)

l_xyz = np.array((600, 100, 122))

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
#n_mon_max = 400
e_matrix = np.zeros((len(x_grid_2nm), len(y_grid_2nm), len(z_grid_2nm)))
z_matrix = np.zeros(len(z_grid_2nm))

source_dir = mv.sim_path_MAC + 'e_DATA/DATA_Pn_20keV_122nm/'

n_files_used = 625

n_events = 0

for i in range(n_files_used):
#for i in range(1):
    
    mf.upd_progress_bar(i, n_files_used)
    
    now_DATA_Pn = np.load(source_dir + 'DATA_Pn_' + str(i) + '.npy')
    
    n_rotations = 0
    
    for i in range(n_rotations + 1):
        
        if n_rotations > 0:
            emf.rotate_DATA(now_DATA_Pn)
        
        n_events += len(now_DATA_Pn)
        
        emf.shift_DATA(now_DATA_Pn, (-beam_d/2, beam_d/2), (-space+y_beg, y_end+space))
        
#        now_DATA_Pn_C = now_DATA_Pn[np.where(now_DATA_Pn[:, mi.atom_id] == mi.C)]
        
        now_DATA_Pn_C_exc = now_DATA_Pn[
                np.where(np.logical_and(
                        np.logical_and(
                                now_DATA_Pn[:, mi.atom_id] == mi.C,
                                now_DATA_Pn[:, mi.coll_id] == mi.exc
                                ),
                        now_DATA_Pn[:, mi.e_dE] != 0)
                        )
                ]
        
        e_matrix += np.histogramdd(now_DATA_Pn_C_exc[:, 5:8], bins=bins_2nm)[0]
        z_matrix += np.histogramdd(now_DATA_Pn_C_exc[:, 7:8], bins=bins_2nm[2:])[0]
        
#%%
#plt.plot(z_grid_2nm, z_matrix, label='1st event z coord')
#plt.xlabel('z, nm')
#plt.ylabel('N events')
#plt.title('Event coordinate distribution, 2 nm')
#plt.legend()
#plt.grid()
#plt.show()
#plt.savefig('LOG events 2nm.png', dpi=300)

#%%
e_matrix_uint16 = np.array(e_matrix, dtype=np.uint16)
print('e_matrix size, Mb:', e_matrix_uint16.nbytes / 1024**2)
np.save(mv.sim_path_MAC + 'MATRIXES/MATRIX_500pC_cm_C_exc.npy', e_matrix_uint16)

#%%
e_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/MATRIX_500pC_cm_C_exc.npy')

#%% drawing
plt.figure()
plt.semilogy(x_grid_2nm, np.sum(e_matrix[:, 25, :], axis=1), label='500 pC/cm')
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
plt.plot(x_grid_2nm, np.sum(e_matrix_xz, axis=1), label='500 pC/cm')
plt.xlabel('x, nm')
plt.ylabel('N events')
plt.title('Event coordinate distribution, 100 nm')
plt.legend()
plt.grid()
plt.show()
#plt.savefig('events 100nm.png', dpi=300)
