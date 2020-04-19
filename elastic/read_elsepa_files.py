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
EE = np.array([
          10,    11,    12,    13,    14,    15,    16,    17,    18,    19, ##  0- 9
          20,    25,    30,    35,    40,    45,    50,    60,    70,    80, ## 10-19
          90,   100,   150,   200,   250,   300,   350,   400,   450,   500, ## 20-29
         600,   700,   800,   900,  1000,  1500,  2000,  2500,  3000,  3500, ## 30-39
        4000,  4500,  5000,  6000,  7000,  8000,  9000, 10000, 11000, 12000, ## 40-49
       13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, ## 50-59
       23000, 24000, 25000]) ## 60-63

a02 = 2.8e-17 ## cm


for el in ['H', 'C', 'O', 'Si']:


    folder_ea = 'easy/' + el
    folder_at = 'atomic/' + el
    folder_mu = 'muffin/' + el
    
    diff_cs_ea = np.zeros((len(EE), 606))
    diff_cs_at = np.zeros((len(EE), 606))
    diff_cs_mu = np.zeros((len(EE), 606))
    
    theta = np.zeros(606)
    
    
    for i, E in enumerate(EE):
        
        E_str = str(int(E))
        
        d1 = E_str[0]
        d2 = E_str[1]
        exp = str(len(E_str) - 1)
        
        fname = 'dcs_' + d1 + 'p' + d2 + '00e0' + exp + '.dat'
        
        theta, diff_cs_ea[i, :] = get_elsepa_theta_diff_cs(os.path.join(folder_ea, fname))
        theta, diff_cs_at[i, :] = get_elsepa_theta_diff_cs(os.path.join(folder_at, fname))
        theta, diff_cs_mu[i, :] = get_elsepa_theta_diff_cs(os.path.join(folder_mu, fname))
        
        EE, cs_ea = get_elsepa_EE_cs(folder_ea)
        EE, cs_at = get_elsepa_EE_cs(folder_at)
        EE, cs_mu = get_elsepa_EE_cs(folder_mu)
        
    
    np.save(os.path.join('raw_arrays', el, 'easy_diff_cs.npy'), diff_cs_ea)
    np.save(os.path.join('raw_arrays', el, 'atomic_diff_cs.npy'), diff_cs_at)
    np.save(os.path.join('raw_arrays', el, 'muffin_diff_cs.npy'), diff_cs_mu)
    
    np.save(os.path.join('raw_arrays', el, 'easy_cs.npy'), cs_ea)
    np.save(os.path.join('raw_arrays', el, 'atomic_cs.npy'), cs_at)
    np.save(os.path.join('raw_arrays', el, 'muffin_cs.npy'), cs_mu)


#%%
i50, i100, i500, i1k, i5k, i10k, i15k, i20k = 16, 21, 29, 34, 42, 47, 52, 58

ind = 3

# now_diff_cs_ea = diff_cs_at[ind, :] / a02
# now_diff_cs_at = diff_cs_at[ind, :] / a02
# now_diff_cs_mu = diff_cs_mu[ind, :] / a02

# print(now_diff_cs_ea[0], now_diff_cs_ea[-1])
# print(now_diff_cs_at[0], now_diff_cs_mu[-1])
# print(now_diff_cs_mu[0], now_diff_cs_mu[-1])

# plt.semilogy(theta, diff_cs_ea[ind, :] / a02, label='easy')
# plt.semilogy(theta, diff_cs_at[ind, :] / a02, label='atomic')
# plt.semilogy(theta, diff_cs_mu[ind, :] / a02, '--', label='muffin')

plt.xlim(0, 180)
# plt.ylim(1e-7, 1e+4)

plt.legend()
plt.grid()


#%%
el = 'Si'

EE, cs_at  = get_elsepa_EE_cs('atomic/' + el)
EE, cs_muf = get_elsepa_EE_cs('muffin/' + el)

#ioffe = np.load('_outdated/Ioffe/Si/u.npy') / mc.n_Si
#
plt.loglog(EE, cs_at)
plt.loglog(EE, cs_muf)
#plt.loglog(mc.EE, ioffe)
