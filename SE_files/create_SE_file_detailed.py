#%% Import
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib

import my_constants as mc
mc = importlib.reload(mc)

os.chdir(os.path.join(mc.sim_folder, 'SE_files'))


#%% read datafile
profile = np.loadtxt('profile.txt')

arr_y_pre = profile[::3, 0]
arr_z_pre = profile[::3, 1] * 5

#step_w = 20
step_w = 8

dummy_y = np.linspace(-step_w/2, step_w/2, 20)
dummy_z = np.ones(len(dummy_y)) * arr_z_pre.max()

inds_1 = np.where(dummy_y < arr_y_pre[0])[0]
inds_2 = np.where(dummy_y > arr_y_pre[-1])[0]

arr_y = np.concatenate((dummy_y[inds_1], arr_y_pre, dummy_y[inds_2]))
arr_z = np.concatenate((dummy_z[inds_1], arr_z_pre, dummy_z[inds_2]))


shift = arr_y.max()

arr_y_final = np.concatenate((arr_y, arr_y + shift, arr_y+shift*2, arr_y+shift*3,\
                              ))
arr_z_final = np.concatenate((arr_z, arr_z, arr_z, arr_z))


arr_y = arr_y_final
arr_z = arr_z_final


n = len(arr_y)

step_l = 10
half_l = step_l / 2


#plt.plot(arr_y, arr_z)


## vertices
V = np.zeros((4*n, 1+3))

for i in range(n):
    V[0*n + i, :] = 100+i,  half_l, arr_y[i], 0
    V[1*n + i, :] = 200+i, -half_l, arr_y[i], 0
    V[2*n + i, :] = 300+i,  half_l, arr_y[i], arr_z[i]
    V[3*n + i, :] = 400+i, -half_l, arr_y[i], arr_z[i]


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot(V[:, 1], V[:, 2], V[:, 3], 'o')
#plt.show()


## edges
E = np.zeros((4*n + 4*(n-1), 1+2))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')


for i in range(n):
    
    V_ind_100 = 100+i
    V_ind_200 = 200+i
    V_ind_300 = 300+i
    V_ind_400 = 400+i
    
    E[0*n + i, :] = 1200+i, V_ind_100, V_ind_200
    E[1*n + i, :] = 1300+i, V_ind_100, V_ind_300
    E[2*n + i, :] = 2400+i, V_ind_200, V_ind_400
    E[3*n + i, :] = 3400+i, V_ind_300, V_ind_400
    
    
    for j in range(4):
        
        beg_x, beg_y, beg_z = V[np.where(V[:, 0] == E[n*j + i, 1])[0], 1:].reshape((3, 1))
        end_x, end_y, end_z = V[np.where(V[:, 0] == E[n*j + i, 2])[0], 1:].reshape((3, 1))
    
        dx = np.array([beg_x, end_x])[:, 0]
        dy = np.array([beg_y, end_y])[:, 0]
        dz = np.array([beg_z, end_z])[:, 0]
        
#        ax.plot(dx, dy, dz)


for i in range(n-1):
    
    for j in range(1, 5):
        
        E[4*n + (j-1)*(n-1) + i] = 100*int(str(j)*2) + i, 100*j + i, 100*j + i+1
        
        beg_x, beg_y, beg_z = V[np.where(V[:, 0] == 100*j + i  )[0], 1:].reshape((3, 1))
        end_x, end_y, end_z = V[np.where(V[:, 0] == 100*j + i+1)[0], 1:].reshape((3, 1))
        
        dx = np.array([beg_x, end_x])[:, 0]
        dy = np.array([beg_y, end_y])[:, 0]
        dz = np.array([beg_z, end_z])[:, 0]
        
#        ax.plot(dx, dy, dz)

    
## faces
F = np.zeros((4*(n-1) + 2, 1+4))


for i in range(n-1):
    
    F[0*(n-1) + i] = 100+i,  (1300 + i+1), -(3300 + i), -(1300 + i),  (1100 + i)
    F[1*(n-1) + i] = 200+i,  (3400 + i+1), -(4400 + i), -(3400 + i),  (3300 + i)
    F[2*(n-1) + i] = 300+i, -(2400 + i+1), -(2200 + i),  (2400 + i),  (4400 + i)
    F[3*(n-1) + i] = 400+i, -(1200 + i+1), -(1100 + i),  (1200 + i),  (2200 + i)


F[4*(n-1)    , :] = 500,   1300,         3400,        -2400,         -1200
F[4*(n-1) + 1, :] = 501, -(1300 + n-1), (1200 + n-1), (2400 + n-1), -(3400 + n-1)



## datafile
file = ''

file += 'PARAMETER step_l = ' + str(step_l) + '\n'
file += 'PARAMETER max_y = ' + str(arr_y.max()) + '\n'
file += 'PARAMETER min_y = ' + str(arr_y.min()) + '\n'
file += 'PARAMETER max_z = ' + str(arr_z.max()) + '\n\n'

file += '/*--------------------CONSTRAINTS START--------------------*/\n'
file += 'constraint 1 /* fixing the resist on the substrate surface */\n'
file += 'formula: x3 = 0\n\n'
file += 'constraint 13 /* mirror plane, resist on front-side wall */\n'
file += 'formula: x1 = 0.5*step_l\n\n'
file += 'constraint 24 /* mirror plane, resist on back-side wall */\n'
file += 'formula: x1 = -0.5*step_l\n\n'
file += 'constraint 2\n'
file += 'formula: x2 = min_y\n\n'
file += 'constraint 3\n'
file += 'formula: x2 = max_y\n\n'
file += 'constraint 4 nonpositive\n'
file += 'formula: x2 = max_z\n'
file += '/*--------------------CONSTRAINTS END--------------------*/\n\n'


file += '/*--------------------VERTICES START--------------------*/\n'
file += 'vertices\n'


for i in range(len(V)):
        
    V_ind = str(int(V[i, 0]))
    V_coords = str(V[i, 1:])[1:-1]
    
    file += V_ind + '\t' + V_coords
    
    V0 = V_ind[0]
    
    if V_ind == '100':
        file += '\tconstraints 1 13 2'
    elif V_ind == '200':
        file += '\tconstraints 1 24 2'
    elif V_ind == '300':
        file += '\tconstraints 13 2'
#        file += '\tconstraints 13 2 4'
    elif V_ind == '400':
        file += '\tconstraints 24 2'
#        file += '\tconstraints 24 2 4'
    elif V_ind == str(100+n-1):
        file += '\tconstraints 1 13 3'
    elif V_ind == str(200+n-1):
        file += '\tconstraints 1 24 3'
    elif V_ind == str(300+n-1):
        file += '\tconstraints 13 3'
#        file += '\tconstraints 13 3 4'
    elif V_ind == str(400+n-1):
        file += '\tconstraints 24 3'
#        file += '\tconstraints 24 3 4'
    elif V0 == '1':
        file += '\tconstraints 1 13'
    elif V0 == '2':
        file += '\tconstraints 1 24'
    elif V0 == '3':
        file += '\tconstraint 13'
    elif V0 == '4':
        file += '\tconstraint 24'
    
    
    
    file += '\n'

file += '/*--------------------VERTICES END--------------------*/\n\n'


file += '/*--------------------EDGES START--------------------*/\n'
file += 'edges' + '\n'


for i in range(len(E)):
    
    E_ind = str(int(E[i, 0]))
    
    beg = str(int(E[i, 1]))
    end = str(int(E[i, 2]))
    
    file += E_ind + '\t' + beg + ' ' + end
    
    E00 = E_ind[:2]
    
    if E_ind == '1200':
        file += '\tconstraints 1 2'
    elif E_ind == '1300':
        file += '\tconstraints 2 13'
    elif E_ind == '3400':
        file += '\tconstraints 2'
#        file += '\tconstraints 2 4'
    elif E_ind == '2400':
        file += '\tconstraints 2 24'
    elif E_ind == str(1200+n-1):
        file += '\tconstraints 1 3'
    elif E_ind == str(1300+n-1):
        file += '\tconstraints 3 13'
    elif E_ind == str(3400+n-1):
        file += '\tconstraints 3'
#        file += '\tconstraints 3 4'
    elif E_ind == str(2400+n-1):
        file += '\tconstraints 3 24'
    elif E00 == '11':
        file += '\tconstraints 1 13'
    elif E00 == '22':
        file += '\tconstraints 1 24'
    elif E00 in ['13', '33']:
        file += '\tconstraint 13'
    elif E00 in ['24', '44']:
        file += '\tconstraint 24'
    elif E00 == '12':
        file += '\tconstraint 1'
    
    file += '\n'


file += '/*--------------------EDGES END--------------------*/\n\n'


file += '/*--------------------FACES START--------------------*/\n'
file += 'faces' + '\n'


for i in range(len(F)):
    
    F_ind = str(int(F[i, 0]))
    F_edges = str(F[i, 1:].astype(int))[1:-1]
    
    file += F_ind + '\t' + F_edges
    
    if F_ind[0] in ['2', '4']:
        file += '\tcolor brown'
    else:
        file += '\tcolor yellow'
        
    file += '\n'


file += '/*--------------------FACES END--------------------*/\n\n'


file += '/*--------------------BODY--------------------*/\n'
file += 'bodies' + '\n' + '1' + '\t'


for fn in F[:, 0]:
    
    file += str(int(fn)) + ' '


## write to file
with open('SE_input.txt', 'w') as myfile:
    myfile.write(file)

