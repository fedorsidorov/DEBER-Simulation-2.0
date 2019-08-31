#%% Import
import numpy as np
from numpy import sin, cos
#import os
import importlib
import my_utilities as mu
import my_constants as mc
import matplotlib.pyplot as plt

mu = importlib.reload(mu)
mc = importlib.reload(mc)

from math import gamma


#%%
def schulz_zimm(x, Mn, Mw):
    
    z = Mn / (Mw - Mn)
    l =  1 / (Mw - Mn)
    
#    f = l**z / gamma(z) * np.power(x, z-1) * np.exp(-l*x)
    f = l**z / (gamma(z) * Mn) * np.power(x, z) * np.exp(-l*x)
    
    return f


def get_chain_len(m, mw):
    
   mw_norm = mw / np.sum(mw)
   
   return int(np.random.choice(m, p=mw_norm) / 100)


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


def plot_chain(chain_arr, beg=0, end=-1):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(chain_arr[beg:end, 0], chain_arr[beg:end, 1], chain_arr[beg:end, 2], 'bo-')
    
    ax.set_xlabel('x, nm')
    ax.set_ylabel('y, nm')
    ax.set_zlabel('z, nm')




