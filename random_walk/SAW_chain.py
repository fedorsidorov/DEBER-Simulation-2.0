#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import matplotlib.pyplot as plt

from random import uniform

mf = importlib.reload(mf)
mv = importlib.reload(mv)

os.chdir(mv.sim_path_MAC + 'random_walk')

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
        
        mf.upd_progress_bar(i, chain_len)
        
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
