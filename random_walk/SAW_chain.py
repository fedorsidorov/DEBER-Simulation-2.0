#%% Import
import numpy as np
from numpy import sin, cos
import os
import importlib
import my_constants as mc
import my_utilities as mu
import matplotlib.pyplot as plt

from random import uniform

mc = importlib.reload(mc)
mu = importlib.reload(mu)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.chdir(mc.sim_folder + 'random_walk')


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

    step = 0.28
    
    chain_coords = np.zeros((chain_len, 3))
    chain_coords[0, :] = 0, 0, 0
    
    i = 1
    
    On = np.eye(3)
    x_prime = np.array([0, 0, 1])
    
    On_list = [None] * chain_len
    On_list[0] = On
    
    while i < chain_len:
        
        mu.upd_progress_bar(i, chain_len)
        
        phi = uniform(0, 2*np.pi)
        theta = np.deg2rad(180-109)
        
        On = get_On(phi, theta, On_list[i-1])
        xn = np.matmul(On.transpose(), x_prime)
        
        chain_coords[i, :] = chain_coords[i-1, :] + step*xn
        On_list[i] = On
        
        i += 1
    
    return chain_coords


def make_PMMA_SAW_chain(chain_len):

    step = 0.28
    step_2 = step**2
    
    chain_len = 1000000
    
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
        
        mu.upd_progress_bar(i, chain_len)
        
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


#def dotproduct(v1, v2):
#    
#    return sum((a*b) for a, b in zip(v1, v2))


#def length(v):
#    
#    return np.sqrt(dotproduct(v, v))


def angle(v1, v2):
    
#    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.arccos(np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def check_chain_bonds(chain_arr):
    
    step = 0.28
    
    for i in range(len(chain_arr)-2):
    
        vector_1 = np.array(chain_arr[i+1] - chain_arr[i])
        vector_2 = np.array(-(chain_arr[i+2] - chain_arr[i+1]))
        
#        print(vector_1.shape)
#        print(vector_2.shape)
        
        if np.abs(np.linalg.norm(vector_1) - step) > 1e-4:
            print(i, 'bond length error')
            return False
        
        now_angle = np.rad2deg(angle(vector_1, vector_2))
        
        if np.abs(now_angle - 109) > 1e-4:
            print(i, 'bond angle error')
            return False
        
    return True


#%%
def rnd_ang():
    
    return 2*np.pi*np.random.random()


def rotate_chain(chain):
    
    a = rnd_ang()
    b = rnd_ang()
    g = rnd_ang()
    
    M = np.mat([
        [cos(a)*cos(g)-sin(a)*cos(b)*sin(g), -cos(a)*sin(g)-sin(a)*cos(b)*cos(g),  sin(a)*sin(b)],
        [sin(a)*cos(g)+cos(a)*cos(b)*sin(g), -sin(a)*sin(g)+cos(a)*cos(b)*cos(g), -cos(a)*sin(b)],
        [                     sin(b)*sin(g),                       sin(b)*cos(g), cos(b)        ]   
        ])
    
    return np.matmul(M, chain.transpose()).transpose()




#%%
#for i in range(100):
#    
#    now_chain = make_PMMA_chain(1000000)
#    
#    np.save('chains_1M/chain_' + str(i) + '.npy', now_chain)
#    
#    print('chain', i, 'is ready')


#%%
chain = np.load('../CHAINS/chains_1M/chain_5.npy')

chain_rot = rotate_chain(chain)

#%%
check_chain_bonds(chain_rot)

