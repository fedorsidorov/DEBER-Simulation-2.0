#%% Import
import numpy as np
from matplotlib import pyplot as plt, cm
import os
import importlib
import copy
from itertools import product

import my_functions as mf
import my_variables as mv
import my_indexes as mi
import my_constants as mc
import my_mapping as mm

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mi = importlib.reload(mi)
mc = importlib.reload(mc)
mm = importlib.reload(mm)

os.chdir(mv.sim_path_MAC + 'diffusion')

#%%
mon_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/MATRIX_mon.npy')
rad_mon_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/MATRIX_rad_mon.npy')

l_xyz = np.array((600, 100, 122))

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

resist_shape = len(x_grid_2nm), len(y_grid_2nm), len(z_grid_2nm)


D = 10
dx = dy = dz = 2

dt = dx**2 / (2 * D) / 100

x_g = x_grid_2nm
z_g = z_grid_2nm

d_coord = dx

u_mon = copy.deepcopy(mon_matrix)
u_rad = copy.deepcopy(rad_mon_matrix)

add_x = np.zeros((1, len(u_mon[0]), len(u_mon[0, 0])))
u_mon = np.concatenate((add_x, u_mon, add_x), axis=0)
u_rad = np.concatenate((add_x, u_rad, add_x), axis=0)

add_y = np.zeros((len(u_mon), 1, len(u_mon[0, 0])))
u_mon = np.concatenate((add_y, u_mon, add_y), axis=1)
u_rad = np.concatenate((add_y, u_rad, add_y), axis=1)

add_z = np.zeros((len(u_mon), len(u_mon[0]), 1))
u_mon = np.concatenate((add_z, u_mon, add_z), axis=2)
u_rad = np.concatenate((add_z, u_rad, add_z), axis=2)

u_tot = u_mon + u_rad


u_shape = np.shape(u_mon)
range_x, range_y, range_z = range(u_shape[0]), range(u_shape[1]),\
                            range(u_shape[2])

rad_mon_table = np.zeros((int(np.sum(u_rad)), 3), dtype=np.int16)

pos = 0

for i, j, k in product(range_x, range_y, range_z):
    
    if u_rad[i, j, k] != 0:
        for ii in range(len(rad_mon_table[int(pos):int(pos+u_rad[i, j, k])])):
            rad_mon_table[int(pos+ii)] = i, j, k
    
        pos += u_rad[i, j, k]

#for line in rad_mon_table:
#    print(u_rad[line[0], line[1], line[2]])

#%%
nt = 10000

for n in range(nt):
    
    mf.upd_progress_bar(n, nt)
    
    un = copy.deepcopy(u_mon)
    
    u_mon[:, :, -1] = u_mon[:, :, -2]
    
    u_mon[1:-1, 1:-1, 1:-1] = un[1:-1, 1:-1, 1:-1] + D * dt / d_coord**2 * (
        un[2:, 1:-1, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[0:-2, 1:-1, 1:-1] +
        un[1:-1, 2:, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 0:-2, 1:-1] +
        un[1:-1, 1:-1, 2:] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 1:-1, 0:-2]
        )
    
    u_tot = u_mon + u_rad
    
    if n % 100 == 0:
    
        for idx, rad_line in enumerate(rad_mon_table):
            
            x, y, z = rad_line
            
            if x == 0 or y == 0 or z == 0 or x == 301 or y == 51 or z == 62:
                continue
    
            arr = u_tot[x-1:x+2, y-1:y+2, z-1:z+2]
            arr_line = arr.reshape((27,))
            
            arr_line[np.where(arr_line < 0)] = 0
            
            if np.all(arr_line == 0):
                arr_line_normed = 1 / 27
            else:
                arr_line_pre = (np.sum(arr_line) - arr_line)
                arr_line_norm = arr_line_pre / np.sum(arr_line_pre)
            
            pos = mf.choice(list(range(27)), p=arr_line_norm)
            
            pos_x = pos // 9
            pos_y = (pos - pos_x*9) // 3
            pos_z = pos - pos_x*9 - pos_y*3
            
            new_x = x + pos_x - 1
            new_y = y + pos_y - 1
            new_z = z + pos_z - 1

            rad_mon_table[idx] = new_x, new_y, new_z
            
            u_rad[x, y, z] -= 1
            u_rad[new_x, new_y, new_z] += 1

#%%
fig = plt.figure()
ax = fig.gca(projection='3d')

X, Z = np.meshgrid(x_g, z_g)

u_mon_sum = np.sum(u_mon[1:-1, 1:-1, 1:-1], axis=1)
u_rad_sum = np.sum(u_rad[1:-1, 1:-1, 1:-1], axis=1)

surf = ax.plot_surface(X, Z, u_mon_sum.transpose(),\
    rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=True)
#ax.set_zlim(0, 2.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$z$')
