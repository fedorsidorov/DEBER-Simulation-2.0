#%% Import
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib

import my_constants as mc
mc = importlib.reload(mc)

os.chdir(os.path.join(mc.sim_folder, 'SE_files'))


#%%
step_l = 6
half_l = step_l / 2

arr_y = np.linspace(0, 50, 20)
arr_z = 10*(1 - np.cos(2*np.pi/50 * arr_y)/2)

#arr_y = np.array([0, 1, 2, 3, 4, 5])
#arr_z = np.array([1, 1, 0.8, 0.8, 1, 1])

n = len(arr_y)

#plt.plot(arr_y, arr_z)


#%% vertices
V = np.zeros(((n+2)*2, 1+3))


V[  0, :] = 101,  half_l, 0, 0
V[n+2, :] = 201, -half_l, 0, 0


for i in range(0, n):
    V[  i+1, :] = 100+i+2,  half_l, arr_y[i], arr_z[i]
    V[i+n+3, :] = 200+i+2, -half_l, arr_y[i], arr_z[i]


V[  n+1, :] = 100+n+2,  half_l, arr_y[n-1], 0
V[2*n+3, :] = 200+n+2, -half_l, arr_y[n-1], 0


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot(V[:, 1], V[:, 2], V[:, 3], 'o')
#plt.show()


#%% edges
E = np.zeros((8+n+2*(n-1), 3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i in range(1, n+3):
    
    beg_V_ind = 100+i
    end_V_ind = 200+i
    
    E[i-1, :] = i, beg_V_ind, end_V_ind
    
    beg_x, beg_y, beg_z = V[np.where(V[:, 0] == beg_V_ind)[0], 1:].reshape((3, 1))
    end_x, end_y, end_z = V[np.where(V[:, 0] == end_V_ind)[0], 1:].reshape((3, 1))
    
    dx = np.array([beg_x, end_x])[:, 0]
    dy = np.array([beg_y, end_y])[:, 0]
    dz = np.array([beg_z, end_z])[:, 0]
    
    ax.plot(dx, dy, dz)


for i in range(1, n+3-1):
    
    beg_V_100_ind = 100+i
    end_V_100_ind = 100+i+1
    
    beg_V_200_ind = 200+i
    end_V_200_ind = 200+i+1
    
    E[n+2 + i-1, :] = 100+i, beg_V_100_ind, end_V_100_ind
    E[2*n+3 + i, :] = 200+i, beg_V_200_ind, end_V_200_ind

    beg_x_100, beg_y_100, beg_z_100 = V[np.where(V[:, 0] == beg_V_100_ind)[0], 1:].reshape((3, 1))
    end_x_100, end_y_100, end_z_100 = V[np.where(V[:, 0] == end_V_100_ind)[0], 1:].reshape((3, 1))
    
    beg_x_200, beg_y_200, beg_z_200 = V[np.where(V[:, 0] == beg_V_200_ind)[0], 1:].reshape((3, 1))
    end_x_200, end_y_200, end_z_200 = V[np.where(V[:, 0] == end_V_200_ind)[0], 1:].reshape((3, 1))
    
    dx_100 = np.array([beg_x_100, end_x_100])[:, 0]
    dy_100 = np.array([beg_y_100, end_y_100])[:, 0]
    dz_100 = np.array([beg_z_100, end_z_100])[:, 0]
    
    dx_200 = np.array([beg_x_200, end_x_200])[:, 0]
    dy_200 = np.array([beg_y_200, end_y_200])[:, 0]
    dz_200 = np.array([beg_z_200, end_z_200])[:, 0]
    
    ax.plot(dx_100, dy_100, dz_100)
    ax.plot(dx_200, dy_200, dz_200)


E[2*n + 3] = 100+n+3-1, 100+n+2, 101
E[3*n + 5] = 200+n+3-1, 200+n+2, 201

beg_x_100, beg_y_100, beg_z_100 = V[np.where(V[:, 0] == 100+n+2)[0], 1:].reshape((3, 1))
end_x_100, end_y_100, end_z_100 = V[np.where(V[:, 0] == 101    )[0], 1:].reshape((3, 1))

beg_x_200, beg_y_200, beg_z_200 = V[np.where(V[:, 0] == 200+n+2)[0], 1:].reshape((3, 1))
end_x_200, end_y_200, end_z_200 = V[np.where(V[:, 0] == 201    )[0], 1:].reshape((3, 1))

dx_100 = np.array([beg_x_100, end_x_100])[:, 0]
dy_100 = np.array([beg_y_100, end_y_100])[:, 0]
dz_100 = np.array([beg_z_100, end_z_100])[:, 0]

dx_200 = np.array([beg_x_200, end_x_200])[:, 0]
dy_200 = np.array([beg_y_200, end_y_200])[:, 0]
dz_200 = np.array([beg_z_200, end_z_200])[:, 0]
    

ax.plot(dx_100, dy_100, dz_100)
ax.plot(dx_200, dy_200, dz_200)
    
    
#%% faces
F_1 = np.zeros((n+2, 5))
F_2 = np.zeros((2, n+2+1))


for i in range(1, n+3-1):
    
    e1_ind = 100+i
    e2_ind = i+1
    e3_ind = 200+i
    e4_ind = i
    
    F_1[i-1] = i, e1_ind, e2_ind, -e3_ind, -e4_ind


F_1[n+1] = n+2, 1, -(200+n+2), -(n+2), 100+n+2


F_2[0, 0] = 301
F_2[1, 0] = 401


for j in range(1, n+3):
    
    F_2[0, j] = -(100+n+2+1-j)
    F_2[1, j] = 200+j


#%% datafile
file = 'vertices\n'


for i in range(len(V)):
    
    V_ind = str(int(V[i, 0]))
    V_coords = str(V[i, 1:])[1:-1]
    
    file += V_ind + '\t' + V_coords + '\n'


file += '\n' + 'edges' + '\n'


for i in range(len(E)):
    
    E_ind = str(int(E[i, 0]))
    beg = str(int(E[i, 1]))
    end = str(int(E[i, 2]))
    file += E_ind + '\t' + beg + ' ' + end + '\n'


file += '\n' + 'faces' + '\n'


for i in range(len(F_1)):
    
    F_ind = str(int(F_1[i, 0]))
    F_edges = str(F_1[i, 1:].astype(int))[1:-1]
    
    file += F_ind + '\t' + F_edges + '\n'


for i in range(len(F_2)):
    
    F_ind = str(int(F_2[i, 0]))
    F_edges = str(F_2[i, 1:].astype(int))[1:-1]
    
    file += F_ind + '\t' + F_edges + '\n'


file += '\n' + 'bodies' + '\n' + '1' + '\t'
    

for i in range(len(F_1)):
    
    file += str(i+1) + ' '


file += str(301) + ' ' + str(401) + '\n'


with open('SE_input.txt', 'w') as myfile:
    myfile.write(file)


