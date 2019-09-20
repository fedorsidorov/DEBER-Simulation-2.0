#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import importlib
import my_functions as mf
import my_variables as mv

mf = importlib.reload(mf)
mv = importlib.reload(mv)
os.chdir(mv.sim_path_MAC + 'mapping')

#%%
l_xyz = np.array((100, 100, 100))

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
c_before = np.load('chain_sum_len_matrix_before.npy')
n_before = np.load('n_chains_matrix_before.npy')

c_after = np.load('chain_sum_len_matrix_C_1.npy')
n_after = np.load('n_chains_matrix_C_1.npy')

#%% before 2 nm
c_x_2_b = np.sum(c_before[:, 25, :], axis=1)
n_x_2_b = np.sum(n_before[:, 25, :], axis=1)

plt.plot(x_grid_2nm, c_x_2_b / n_x_2_b)
plt.xlabel('x, nm')
plt.ylabel('AVG local chain length')
plt.title('Chain local length distribution BEFORE EXPOSURE, 2 nm')
plt.legend()
plt.grid()
plt.show()
plt.savefig('AVG chains local 2nm before.png', dpi=300)

#%% before 100 nm
c_x_100_b = np.sum(np.sum(c_before, axis=1), axis=1)
n_x_100_b = np.sum(np.sum(n_before, axis=1), axis=1)

plt.plot(x_grid_2nm, c_x_100_b / n_x_100_b)
plt.xlabel('x, nm')
plt.ylabel('AVG local chain length')
plt.title('Chain local length distribution BEFORE EXPOSURE, 100 nm')
plt.legend()
plt.grid()
plt.show()
plt.savefig('AVG chains local 100nm before.png', dpi=300)

#%% after 2 nm imshow
n_after_mod = copy.deepcopy(n_after)
n_after_mod[np.where(n_after_mod == 0)] = 1

chain_len_avg_after = c_after / n_after_mod
chain_len_avg_after_mono = (chain_len_avg_after - 100) / np.abs(chain_len_avg_after - 100)

plt.imshow(np.sum(chain_len_avg_after_mono[:, :, :], axis=1).transpose() / 50)
plt.colorbar()

#%% after 2 nm
c_x_2_a = np.sum(c_after[:, 25, :], axis=1)
n_x_2_a = np.sum(n_after[:, 25, :], axis=1)

plt.plot(x_grid_2nm, c_x_2_a / n_x_2_a)
plt.xlabel('x, nm')
plt.ylabel('AVG local chain length')
plt.title('Chain local length distribution AFTER EXPOSURE, 2 nm')
#plt.legend()
plt.grid()
plt.show()
plt.savefig('AVG chains local 2nm after.png', dpi=300)

#%% after 100 nm
c_x_100_a = np.sum(np.sum(c_after, axis=1), axis=1)
n_x_100_a = np.sum(np.sum(n_after, axis=1), axis=1)

plt.plot(x_grid_2nm, c_x_100_a / n_x_100_a)
plt.xlabel('x, nm')
plt.ylabel('AVG local chain length')
plt.title('Chain local length distribution AFTER EXPOSURE, 100 nm')
#plt.legend()
plt.grid()
plt.show()
plt.savefig('AVG chains local 100nm after.png', dpi=300)
