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

    cs_H = np.load('final_arrays/' + kind + '/H/H_' + kind + '_cs_extrap.npy')
    cs_C = np.load('final_arrays/' + kind + '/C/C_' + kind + '_cs_extrap.npy')
    cs_O = np.load('final_arrays/' + kind + '/O/O_' + kind + '_cs_extrap.npy')
    cs_Si = np.load('final_arrays/' + kind + '/Si/Si_' + kind + '_cs_extrap.npy')

    cs_MMA = mc.N_H_MMA*cs_H + mc.N_C_MMA*cs_C + mc.N_O_MMA*cs_O
    
    u_PMMA = cs_MMA * mc.n_MMA
    u_Si  = cs_Si  * mc.n_Si

    # plt.loglog(mc.EE, cs_MMA)
    
    np.save('final_arrays/PMMA/MMA_' + kind + '_cs.npy', cs_MMA)
    np.save('final_arrays/PMMA/PMMA_' + kind + '_u.npy', u_PMMA)
    
    np.save('final_arrays/Si/Si_' + kind + '_cs.npy', cs_Si)
    np.save('final_arrays/Si/Si_' + kind + '_u.npy', u_Si)



