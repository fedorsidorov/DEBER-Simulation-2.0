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
#def get_Ruth_diff_cs(Z, E):
#    
#    alpha = mc.k_el**2 * (mc.m * mc.e**4 * np.pi**2 * Z**(2/3)) / (mc.h**2 * E * mc.eV)
#    
#    diff_cs = mc.k_el**2 * Z**2 * mc.e**4/ (4 * (E * mc.eV)**2) /\
#        np.power(1 - np.cos(mc.THETA_rad) + alpha, 2) * 1e+4
#    
#    return diff_cs
#
#
#def get_Ruth_cs(Z, E=mc.EE):
#    
#    alpha = mc.k_el**2 * (mc.m * mc.e**4 * np.pi**2 * Z**(2/3)) / (mc.h**2 * E * mc.eV)
#        
#    cs = np.pi * mc.k_el**2 * Z**2 * mc.e**4/ ((E * mc.eV)**2) / (alpha * (2 + alpha)) * 1e+4
#    
#    return cs


#%%
kind_PMMA = 'muffin'
kind_Si = 'muffin'

#cs_H = np.load('final_arrays/H/' + kind_PMMA + '_cs.npy')
#cs_C = np.load('final_arrays/C/' + kind_PMMA + '_cs.npy')
#cs_O = np.load('final_arrays/O/' + kind_PMMA + '_cs.npy')
#cs_Si = np.load('final_arrays/Si/' + kind_Si + '_cs.npy')

cs_H = np.load('final_arrays/H/' + kind_PMMA + '_cs_extrap.npy')
cs_C = np.load('final_arrays/C/' + kind_PMMA + '_cs_extrap.npy')
cs_O = np.load('final_arrays/O/' + kind_PMMA + '_cs_extrap.npy')
cs_Si = np.load('final_arrays/Si/' + kind_Si + '_cs_extrap.npy')

diff_cs_H = np.load('final_arrays/H/' + kind_PMMA + '_diff_cs.npy')
diff_cs_C = np.load('final_arrays/C/' + kind_PMMA + '_diff_cs.npy')
diff_cs_O = np.load('final_arrays/O/' + kind_PMMA + '_diff_cs.npy')
diff_cs_Si = np.load('final_arrays/Si/' + kind_Si + '_diff_cs.npy')


cs_MMA = mc.N_H_MMA*cs_H + mc.N_C_MMA*cs_C + mc.N_O_MMA*cs_O
diff_cs_MMA = mc.N_H_MMA*diff_cs_H + mc.N_C_MMA*diff_cs_C + mc.N_O_MMA*diff_cs_O

cs_MMA_test = np.zeros(len(cs_MMA))
cs_Si_test = np.zeros(len(cs_Si))

diff_cs_MMA_cumulated = np.zeros(np.shape(diff_cs_MMA))
diff_cs_Si_cumulated = np.zeros(np.shape(diff_cs_Si))

#cs_MMA_R = mc.N_H_MMA*get_Ruth_cs(mc.Z_H) +\
#           mc.N_C_MMA*get_Ruth_cs(mc.Z_C) +\
#           mc.N_O_MMA*get_Ruth_cs(mc.Z_O)
           
#diff_cs_MMA_R = np.zeros(np.shape(diff_cs_MMA))


for i in range(len(mc.EE)):
    
#    diff_cs_MMA_R[i, :] = mc.N_H_MMA*get_Ruth_diff_cs(mc.Z_H, mc.EE[i]) +\
#                          mc.N_C_MMA*get_Ruth_diff_cs(mc.Z_C, mc.EE[i]) +\
#                          mc.N_O_MMA*get_Ruth_diff_cs(mc.Z_O, mc.EE[i])
    
    now_diff_cs_MMA = diff_cs_MMA[i, :]
    now_diff_cs_Si  = diff_cs_Si[i , :]
    
    yy_MMA = now_diff_cs_MMA * 2 * np.pi * np.sin(mc.THETA_rad)
    yy_Si  = now_diff_cs_Si  * 2 * np.pi * np.sin(mc.THETA_rad)
    
    diff_cs_MMA_cumulated[i, :] = mu.diff2int_1d(yy_MMA, mc.THETA_rad)
    diff_cs_Si_cumulated[i,  :] = mu.diff2int_1d(yy_Si , mc.THETA_rad)
    
    cs_MMA_test[i] = np.trapz(yy_MMA, mc.THETA_rad)
    cs_Si_test[i]  = np.trapz(yy_Si , mc.THETA_rad)


#%%
plt.loglog(mc.EE, cs_MMA)
plt.loglog(mc.EE, cs_MMA_test, '.')
plt.loglog(mc.EE, cs_Si)
plt.loglog(mc.EE, cs_Si_test, '.')


#%%
plt.plot(mc.THETA_deg, diff_cs_MMA_cumulated[500, :])
plt.plot(mc.THETA_deg, diff_cs_Si_cumulated[500, :])


#%%
plt.semilogy(mc.THETA_deg, diff_cs_MMA[300, :])
plt.semilogy(mc.THETA_deg, diff_cs_Si[300, :])


#%%
u_MMA = cs_MMA * mc.n_MMA
u_Si  = cs_Si  * mc.n_Si


#%%
u_MMA_m = np.load('final_arrays/PMMA/u_atomic.npy')
u_Si_m = np.load('final_arrays/Si/u_atomic.npy')


#%%
plt.loglog(mc.EE, u_MMA)
plt.loglog(mc.EE, u_MMA_m)


#%%
plt.loglog(mc.EE, u_Si)
plt.loglog(mc.EE, u_Si_m)

