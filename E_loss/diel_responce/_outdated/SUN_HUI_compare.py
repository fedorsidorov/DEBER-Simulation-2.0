#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import my_constants as mc
import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mc = importlib.reload(mc)

os.chdir(mv.sim_path_MAC + 'diel_responce')

#%%
E_arr = np.load('E_PALIK.npy')
Im_arr = np.load('IM_PALIK.npy')

IMFP_inv_arr = np.zeros(len(E_arr))

#%%
diff_IMFP_inv_arr = np.zeros((len(E_arr), len(E_arr)))

for i in range(len(E_arr)):
    
    print(i)
    
    now_E = E_arr[i]

    for j in range(len(E_arr)):
        
        now_w = E_arr[j]
        
        w_min = 0
        
        if now_w <= now_E/2:
            w_min = 0
        elif now_w <= 3*now_E/4:
            w_min = 2*now_w - now_E
        else:
            continue
        
        w_max = 2*np.sqrt(now_E - now_w)*(np.sqrt(now_E) - np.sqrt(now_E - now_w))
        
        inds = np.where(np.logical_and(E_arr >= w_min, E_arr <= w_max))[0]
        
        X = E_arr[inds]*mc.eV
        Y = Im_arr[inds] * E_arr[inds]*mc.eV * 1/(now_w*mc.eV * (now_w - E_arr[inds])*mc.eV)
        
        diff_IMFP_inv_arr[i, j] = mc.k_el * mc.m * mc.e**2 /\
            (2 * np.pi * mc.hbar**2 * now_E*mc.eV) * np.trapz(Y, x=X)        

#%%    
plt.loglog(E_arr, diff_IMFP_inv_arr[4545, :] / 1e+2 * mc.eV, label='My')

#%%
sun_diel = np.loadtxt('curves/sun_diel_Si_1000.txt')

plt.loglog(sun_diel[:, 0], sun_diel[:, 1], label='sun hui')
plt.title('diff_IMFP_inv_arr at 1 keV')
plt.legend()
plt.ylim(1e+1, 1e+6)
plt.grid()

#%%
test = np.load('DIFF_MU_SI_PALIK.npy')

#%%
diff_IMFP_inv_arr = np.zeros((len(E_arr), len(E_arr)))

for i in range(len(E_arr)):
    
    now_E = E_arr[i]
    
    for j in range(len(E_arr)):
        
        now_w = E_arr[j]
        
        w_min = -1
        
        if now_w <= now_E/2:
            w_min = 0
        elif now_w <= 3*now_E/4:
            w_min = 2*now_w - now_E
        else:
            continue
        
        w_max = 2*np.sqrt(now_E - now_w)*(np.sqrt(now_E) - np.sqrt(now_E - now_w))
        
        ind_min = np.where(E_arr == w_min)[0]
        ind_max = np.where(E_arr == w_max)[0]
        
        inds = np.where(np.logical_and(E_arr >= w_min, E_arr <= w_max))[0]
        
        X = E_arr[inds]*mc.eV
        Y = Im_arr[inds] * E_arr[inds] * 1/(now_w*mc.eV * (now_w - E_arr[inds])*mc.eV)
        
        diff_IMFP_inv_arr[i, inds] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 *\
            now_E*mc.eV) * np.trapz(Y, x=X)
    
#%%
plt.loglog(E_arr, diff_IMFP_inv_arr[925, :])
