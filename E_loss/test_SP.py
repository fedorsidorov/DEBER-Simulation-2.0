#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder, 'E_loss'))


#%% Second approach
SP_arr = np.zeros((len(mc.EE), 2))
IMFP_arr = np.zeros((len(mc.EE), 2))

n_primaries = 1282


for f_num in range(n_primaries):
    
    mu.pbar(f_num, n_primaries)
    
    DATA = np.load(os.path.join(mc.sim_folder,
            'e_DATA/primary/DATA_PMMA_prim_' + str(f_num) + '.npy'
            ))
    
    for i in range(len(DATA) - 1):
        
        E_pos = 977
        
        E = DATA[i, 4]
        dE = DATA[i, 8] + DATA[i, 9]
        
        ds = np.linalg.norm((DATA[i+1, 5:8] - DATA[i, 5:8]))
        
        
        if DATA[i, 3] != 0:
            E_pos = np.argmin(np.abs(mc.EE - E))
        
        
        SP_arr[E_pos, 0] += dE
        SP_arr[E_pos, 1] += ds
        
        IMFP_arr[E_pos, 0] += ds
        IMFP_arr[E_pos, 1] += 1
        

#%%
SP_avg = np.zeros(len(mc.EE))
IMFP_avg = np.zeros(len(mc.EE))


for i in range(len(SP_avg)):
 
    if SP_arr[i, 1] == 0:
        SP_avg[i] = 0
    else:
        SP_avg[i] = SP_arr[i, 0] / SP_arr[i, 1]
    
    if IMFP_arr[i, 1] == 0:
        IMFP_avg[i] = 0
    else:
        IMFP_avg[i] = IMFP_arr[i, 0] / IMFP_arr[i, 1]


#%% SP
SP_Dapor = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Dapor_2020', 'PMMA_ee_SP_Dapor.npy'
        ))

plt.semilogx(mc.EE, SP_avg, 'ro', label='simulation')
plt.semilogx(mc.EE, SP_Dapor, label='Dapor')

plt.xlabel('E, eV')
plt.ylabel('SP, eV/$\AA$')

plt.xlim(1, 1e+4)
plt.ylim(0, 2e+9)

plt.grid()
plt.legend()
plt.show()

#plt.savefig('PMMA_SP_Dapor.png', dpi=300)


#%% IMFP
#IMFP_Tahir = np.loadtxt('curves/IMFP.txt')
#plt.loglog(ma.E_arr, IMFP_avg, 'ro', label='me')
#plt.loglog(IMFP_Tahir[:, 0], IMFP_Tahir[:, 1]*1e-8, '-b', label='Tahir')
#plt.xlim(1e+1, 3e+4)
#plt.legend()
#plt.show()

