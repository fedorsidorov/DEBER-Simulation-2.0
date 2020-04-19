#%% Import
import numpy as np
import os
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import importlib

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder, 'elastic'))


#%%
for kind in ['easy', 'atomic', 'muffin']:    

    diff_cs_H  = np.load('final_arrays/H/'  + kind + '_diff_cs.npy')
    diff_cs_C  = np.load('final_arrays/C/'  + kind + '_diff_cs.npy')
    diff_cs_O  = np.load('final_arrays/O/'  + kind + '_diff_cs.npy')
    diff_cs_Si = np.load('final_arrays/Si/' + kind + '_diff_cs.npy')
    
    diff_cs_MMA = mc.N_H_MMA * diff_cs_H +\
                  mc.N_C_MMA * diff_cs_C +\
                  mc.N_O_MMA * diff_cs_O
    
    diff_cs_MMA_plane = np.zeros(np.shape(diff_cs_MMA))
    diff_cs_Si_plane = np.zeros(np.shape(diff_cs_Si))
    
    diff_cs_MMA_plane_norm = np.zeros(np.shape(diff_cs_MMA))
    diff_cs_Si_plane_norm = np.zeros(np.shape(diff_cs_Si))
    
    diff_cs_MMA_cumulated = np.zeros(np.shape(diff_cs_MMA))
    diff_cs_Si_cumulated = np.zeros(np.shape(diff_cs_Si))
    
    
    for i in range(len(mc.EE)):
        
        now_diff_cs_MMA_plane = diff_cs_MMA[i, :] * 2*np.pi * np.sin(mc.THETA_rad)
        now_diff_cs_Si_plane = diff_cs_Si[i, :] * 2*np.pi * np.sin(mc.THETA_rad)
        
        diff_cs_MMA_plane[i, :] = now_diff_cs_MMA_plane
        diff_cs_Si_plane[i, :] = now_diff_cs_Si_plane
        
        diff_cs_MMA_plane_norm[i, :] = now_diff_cs_MMA_plane / np.sum(now_diff_cs_MMA_plane)
        diff_cs_Si_plane_norm[i, :] = now_diff_cs_Si_plane / np.sum(now_diff_cs_Si_plane)
        
        diff_cs_MMA_cumulated[i, :] = mu.diff2int_1d(mc.THETA_rad, now_diff_cs_MMA_plane)
        diff_cs_Si_cumulated[i, :] = mu.diff2int_1d(mc.THETA_rad, now_diff_cs_Si_plane)
    
    
    np.save('final_arrays/PMMA/diff_cs_plane_norm_' + kind + '.npy', diff_cs_MMA_plane_norm)
    np.save('final_arrays/PMMA/diff_cs_cumulated_' + kind + '.npy', diff_cs_MMA_cumulated)
    
    np.save('final_arrays/Si/diff_cs_plane_norm_' + kind + '.npy', diff_cs_Si_plane_norm)
    np.save('final_arrays/Si/diff_cs_cumulated_' + kind + '.npy', diff_cs_Si_cumulated)

