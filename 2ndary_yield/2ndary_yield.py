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

os.chdir(os.path.join(mc.sim_folder, '2ndary_yield'))


#%%
def print_2ndaries(model, n_tracks):
    
    source_folder = os.path.join(mc.sim_folder, 'e_DATA', '2ndaries', model)
    E_str_list = os.listdir(source_folder)
    
    E_list = []
    
    
    for E_str in E_str_list:
        
        if E_str == '.DS_Store':
            continue
        
        E_list.append(int(E_str))
    
    
    E_final_list = []
    d_list = []
    
    for E in sorted(E_list):
    
        source = os.path.join(source_folder, str(E))
        
        filenames = os.listdir(source)
        
        n_total = 0
        n_2nd = 0
        
        
        for fname in filenames:
            
            if fname == '.DS_Store':
                continue
            
            DATA = np.load(os.path.join(source, fname))
            
            n_total += n_tracks
            n_2nd += DATA
        
        
        my_d = n_2nd/n_total
        
        E_final_list.append(E)
        d_list.append(my_d)
        
    
    plt.plot(E_final_list, d_list, '*-', label=model)
    
    return E_final_list, d_list


#%%
print_2ndaries('Mermin', 100)

# C_sim = np.loadtxt(os.path.join(mc.sim_folder, '2ndary_yield', 'ciappa2010.txt'))
D_sim = np.loadtxt(os.path.join(mc.sim_folder, '2ndary_yield', 'Dapor_sim.txt'))
D_exp = np.loadtxt(os.path.join(mc.sim_folder, '2ndary_yield', 'Dapor_exp.txt'))

# plt.plot(C_sim[:, 0], C_sim[:, 1], 'o-', label='Ciappa')
plt.plot(D_sim[:, 0], D_sim[:, 1], 'o-', label='Dapor')
plt.plot(D_exp[:, 0], D_exp[:, 1], 'o-', label='experiment')

plt.legend()
plt.grid()

# plt.xlim(0, 1600)
plt.ylim(0, 3)


