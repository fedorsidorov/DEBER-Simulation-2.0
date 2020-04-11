#%% Import
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder, 'elastic'))


#%%
def get_elsepa_theta_diff_cs(filename):
    
    with open(os.path.join(filename), 'r', encoding='utf-8') as f:
        file_lines = f.readlines()
    
    theta = np.zeros(606)
    diff_cs = np.zeros(606)
    i = 0
    
    for line in file_lines:
        if line[:2] == ' #':
            continue
        
        else:
            line_arr = line.split()
            theta[i] = line_arr[0]
            diff_cs[i] = line_arr[2]
            
            i += 1
            
    return theta, diff_cs


def get_elsepa_EE_cs(dirname):
    
    filename = os.path.join(dirname, 'tcstable.dat')
    
    with open(os.path.join(filename), 'r', encoding='utf-8') as f:
        file_lines = f.readlines()
    
    EE = np.zeros(63)
    cs = np.zeros(63)
    i = 0
    
    for line in file_lines:
        if line[:2] == ' #':
            continue
        
        else:
            line_arr = line.split()
            EE[i] = line_arr[0]
            cs[i] = line_arr[1]
            
            i += 1
    
    return EE, cs


#%%
EE = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45,
      50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500,
      600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
      5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000,
      16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000])


folder = 'material/Si'

diff_cs = np.zeros((len(EE), 606))
theta = np.zeros(606)

filenames = os.listdir(folder)


for i, E in enumerate(EE):
    
    E_str = str(E)
    
    d1 = E_str[0]
    d2 = E_str[1]
    exp = str(len(E_str) - 1)
    
    fname = 'dcs_' + d1 + 'p' + d2 + '00e0' + exp + '.dat'
    
    diff_cs[i, :] = get_elsepa_theta_diff_cs(os.path.join(folder, fname))[1]


#%%
el = 'Si'

EE, cs_at  = get_elsepa_EE_cs('atomic/' + el)
EE, cs_muf = get_elsepa_EE_cs('muffin/' + el)

#ioffe = np.load('_outdated/Ioffe/Si/u.npy') / mc.n_Si
#
plt.loglog(EE, cs_at)
plt.loglog(EE, cs_muf)
#plt.loglog(mc.EE, ioffe)
