#%% Import
import numpy as np
import os
import importlib
import my_utilities as mu
import my_constants as mc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from random import uniform

mu = importlib.reload(mu)
mc = importlib.reload(mc)

os.chdir(os.path.join(mc.sim_folder, 'random_walk'))

from numpy import sin, cos, arccos


#%%
def check_chain(chain_coords, now_mon_coords, step_2):
    for mon_coords in chain_coords[:-1, :]:    
        if np.sum((mon_coords - now_mon_coords)**2) < step_2:
            return False
    return True


def get_On(phi, theta, O_pre):
    
    Wn = np.mat([
                [            cos(phi),             sin(phi),          0],
                [-sin(phi)*cos(theta),  cos(phi)*cos(theta), sin(theta)],
                [ sin(phi)*sin(theta), -cos(phi)*sin(theta), cos(theta)]
                ])
    On = np.matmul(Wn, O_pre)
    
    return On


def make_PMMA_chain(chain_len):

    step = 1
    step_2 = step**2
    
    chain_len = 200
    
    chain_coords = np.zeros((chain_len, 3))
    chain_coords[0, :] = 0, 0, 0
    
    ## collision counter
    jam_cnt = 0
    ## collision link number
    jam_pos = 0
    
    i = 1
    
    On = np.eye(3)
    x_prime = np.array([0, 0, 1])
    
    On_list = [None] * chain_len
    On_list[0] = On
    
    while i < chain_len:
        
        mu.pbar(i, chain_len)
        
        while True:
            
            phi = uniform(0, 2*np.pi)
            theta = np.deg2rad(180-109)
            
            On = get_On(phi, theta, On_list[i-1])
            xn = np.matmul(On.transpose(), x_prime)
            
            chain_coords[i, :] = chain_coords[i-1, :] + step*xn
            On_list[i] = On
            
            st = check_chain(chain_coords[:i, :], chain_coords[i, :], step_2)
            
            if st:
                break
            
            else: ## if no free space
                
                if np.abs(jam_pos - i) < 10: ## if new jam is near current link
                    jam_cnt += 1 ## increase collision counter
                
                else: ## if new jam is on new link
                    jam_pos = i ## set new collision position
                    jam_cnt = 0 ## set collision counter to 0
                
                print(i, ': No free space,', jam_cnt)
                
                ## if possible, make rollback proportional to jam_cnt
                rollback_step = jam_cnt // 10
                
                if i - (rollback_step + 1) >= 0:
                    i -= rollback_step
                    continue
                
                else:
                    print('Jam in very start!')
                    break
        
        i += 1
    
    return chain_coords


def plot_chain(chain_arr, beg=0, end=-1):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(chain_arr[beg:end, 0], chain_arr[beg:end, 1], chain_arr[beg:end, 2], 'bo-')
    
    ax.set_xlabel('x, nm')
    ax.set_ylabel('y, nm')
    ax.set_zlabel('z, nm')


def check_angles(chain_arr):
    
    def dotproduct(v1, v2):
      return sum((a*b) for a, b in zip(v1, v2))
    
    def length(v):
      return np.sqrt(dotproduct(v, v))
    
    def angle(v1, v2):
        return arccos(dotproduct(v1, v2) / (length(v1) * length(v2)))
    
    angles = []
    
    for i in range(len(chain_arr)-2):
        
        vector_1 =   chain_arr[i+1] - chain_arr[i]
        vector_2 = -(chain_arr[i+2] - chain_arr[i+1])
        
        angles.append(np.rad2deg(angle(vector_1, vector_2)))
    
    if np.all(angles==109):
        return True
    
    return False


#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


dxdydz = np.array([-1, -1, -1])
v0 = np.array((*np.zeros(3), *dxdydz)).reshape((2, 3))
plt.plot(v0[:, 0], v0[:, 1], v0[:, 2], 'k')

cos_theta = np.dot(dxdydz, [0, 0, -1]) / (np.linalg.norm(dxdydz) * 1)
theta = np.arccos(cos_theta)

#O = get_On(theta*2, np.pi, np.eye(3))
O = np.mat(np.eye(3))
O[2, :] *= -1

dxdydz1 = np.matmul(O.transpose(), dxdydz).A1
v1 = np.array((*np.zeros(3), *dxdydz1)).reshape((2, 3))
plt.plot(v1[:, 0], v1[:, 1], v1[:, 2], 'k--')


i0 = np.array([1, 0, 0])
j0 = np.array([0, 1, 0])
k0 = np.array([0, 0, 1])

i1 = np.matmul(O.transpose(), i0).A1
j1 = np.matmul(O.transpose(), j0).A1
k1 = np.matmul(O.transpose(), k0).A1

vi0 = np.array((*np.zeros(3), *i0)).reshape((2, 3))
vj0 = np.array((*np.zeros(3), *j0)).reshape((2, 3))
vk0 = np.array((*np.zeros(3), *k0)).reshape((2, 3))

vi1 = np.array((*np.zeros(3), *i1)).reshape((2, 3))
vj1 = np.array((*np.zeros(3), *j1)).reshape((2, 3))
vk1 = np.array((*np.zeros(3), *k1)).reshape((2, 3))

plt.plot(vi0[:, 0], vi0[:, 1], vi0[:, 2], 'r')
plt.plot(vj0[:, 0], vj0[:, 1], vj0[:, 2], 'g')
plt.plot(vk0[:, 0], vk0[:, 1], vk0[:, 2], 'b')

plt.plot(vi1[:, 0], vi1[:, 1], vi1[:, 2], 'r--')
plt.plot(vj1[:, 0], vj1[:, 1], vj1[:, 2], 'g--')
plt.plot(vk1[:, 0], vk1[:, 1], vk1[:, 2], 'b--')

ax.set_xlabel('x, nm')
ax.set_ylabel('y, nm')
ax.set_zlabel('z, nm')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

