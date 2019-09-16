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
mon_rad_matrix = np.load(mv.sim_path_MAC + 'MATRIXES/MATRIX_rad_mon.npy')

#%%
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

resist_shape = len(x_grid_2nm), len(y_grid_2nm), len(z_grid_2nm)

#%%
#nx = 31
#ny = 31
#nt = 17
D = 2
#dx = 2 / (nx - 1)
#dy = 2 / (ny - 1)
dx = 2
dy = 2
sigma = .25
dt = sigma * dx * dy / D / 100

#x = numpy.linspace(0, 2, nx)
#y = numpy.linspace(0, 2, ny)

x = x_grid_2nm
y = y_grid_2nm
z = z_grid_2nm

d_coord = dx

#%% 2D
u = copy.deepcopy(np.sum(mon_rad_matrix, axis=1))
#u = copy.deepcopy(mon_rad_matrix[:, 25, :] / (2e-7)**3)

#%%
u = np.hstack((np.zeros(np.shape(u[:, :1])), u))

#%%
nt = 100000

for n in range(nt):
    
    mf.upd_progress_bar(n, nt)
    
    un = copy.deepcopy(u)
    
    u[1:-1, 1:-1] = un[1:-1, 1:-1] + D * dt / d_coord**2 * (
        un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1] +
        un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2] )


#plt.imshow(np.log10(u[125:175:, 1:-1]))
#plt.colorbar()
#plt.show()

#%%
fig = plt.figure()
ax = fig.gca(projection='3d')

#X, Y = np.meshgrid(x, y)
X, Y = np.meshgrid(x, z)

surf = ax.plot_surface(X, Y, u[:, 1:].transpose(), rstride=1,\
    cstride=1, cmap=cm.viridis, linewidth=0, antialiased=True)
#ax.set_zlim(0, 2.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$z$')

#%% 3D
u = copy.deepcopy(mon_rad_matrix / (2e-7)**3)

nt = 10000

for n in range(nt):
    
    mf.upd_progress_bar(n, nt)
    
    un = copy.deepcopy(u)
    
#    for i, j, k in product(range(1, len(x)-1), range(1, len(y)-1), range(1, len(z)-1)):
                           
#        u[i, j, k] = un[i, j, k]
#        add_x = un[i+1, j, k] - 2 * un[i, j, k] + un[i-1, j, k]
#        add_y = un[i, j+1, k] - 2 * un[i, j, k] + un[i, j-1, k]
#        add_z = un[i, j, k+1] - 2 * un[i, j, k] + un[i, j, k-1]
#        u[i, j, k] += D * (add_x + add_y + add_z) * dt / d_coord**2
#        if u[i, j, k] < 0:
#            print(n, i, j, k, add_x, add_y, add_z)
#            break
        
    u[1:-1, 1:-1, 1:-1] = un[1:-1, 1:-1, 1:-1] + D * dt / d_coord**2 * (
        un[2:, 1:-1, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[0:-2, 1:-1, 1:-1] +
        un[1:-1, 2:, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 0:-2, 1:-1] +
        un[1:-1, 1:-1, 2:] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 1:-1, 0:-2]
        )
    
#    u[np.where(u <= 0)] = 0

#plt.imshow(np.sum(u[125:175:, :, :], axis=2))
#plt.colorbar()
#plt.show()

#%%
fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(x, z)

surf = ax.plot_surface(X, Y, np.sum(u, axis=1).transpose(), rstride=1,\
    cstride=1, cmap=cm.viridis, linewidth=0, antialiased=True)
#ax.set_zlim(0, 2.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$z$')
