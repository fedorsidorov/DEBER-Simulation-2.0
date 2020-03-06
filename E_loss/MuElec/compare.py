#%% Import
import numpy as np
import os
import importlib

import my_constants as mc
import my_utilities as mu
import Gryzinski as gryz

#from itertools import product


import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)
gryz = importlib.reload(gryz)


os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'MuElec'
        ))


#%%
EE = mc.EE

s1 = np.zeros(len(EE))
s2 = np.zeros(len(EE))
s3 = np.zeros(len(EE))
s4 = np.zeros(len(EE))
s5 = np.zeros(len(EE))
s6 = np.zeros(len(EE))


for i, E in enumerate(mc.EE):
    
    mu.pbar(i, len(mc.EE))
    
    s1[i] = gryz.get_Gr_u(gryz.Si_MuElec_Eb[0], E, 1, gryz.Si_MuElec_occ[0])
    s2[i] = gryz.get_Gr_u(gryz.Si_MuElec_Eb[1], E, 1, gryz.Si_MuElec_occ[1])
    s3[i] = gryz.get_Gr_u(gryz.Si_MuElec_Eb[2], E, 1, gryz.Si_MuElec_occ[2])
    s4[i] = gryz.get_Gr_u(gryz.Si_MuElec_Eb[3], E, 1, gryz.Si_MuElec_occ[3])
    s5[i] = gryz.get_Gr_u(gryz.Si_MuElec_Eb[4], E, 1, gryz.Si_MuElec_occ[4])
    s6[i] = gryz.get_Gr_u(gryz.Si_MuElec_Eb[5], E, 1, gryz.Si_MuElec_occ[5])


#%%
s_inel = np.loadtxt('microelec/sigma_inelastic_e_Si.dat')
s_inel_tot = np.zeros(len(s_inel))

EE_f = s_inel[:, 0]


for i in range(1, len(s_inel[0])):
    
    print(i)
    
    now_s_inel = s_inel[:, i]
    s_inel_tot += now_s_inel
    
    plt.loglog(EE_f, s_inel[:, i], 'o')


#plt.loglog(EE_f, s_inel_tot)

plt.xlim(1e+1, 1e+4)
plt.ylim(1e-3, 1e+3)


#plt.loglog(EE, s1*1e+18, '--')
#plt.loglog(EE, s2*1e+18, '--')
#plt.loglog(EE, s3*1e+18, '--')
#plt.loglog(EE, s4*1e+18, '--')
#plt.loglog(EE, s5*1e+18, '--')
#plt.loglog(EE, s6*1e+18, '--')


#%%
plt.loglog(EE_f, s_inel[:, 1]+s_inel[:, 2]+s_inel[:, 3], '--', label='M')
plt.loglog(EE_f, s_inel[:, 4]+s_inel[:, 5], '--', label='L')
plt.loglog(EE_f, s_inel[:, 6], '--', label='K')

plt.loglog(EE, (s1+s2+s3)*1e+18, label='Mg')
plt.loglog(EE, (s4+s5)*1e+18, label='Lg')
plt.loglog(EE, s6*1e+18, label='Kg')

plt.loglog()

plt.xlim(1e+1, 1e+4)


#%%
#valentin = np.loadtxt('valentin_Si_shells.txt')
#plt.loglog(valentin[:, 0], valentin[:, 1], '.')


#%%
diff_6 = np.load('sigmadiff_6.npy')

s_test = np.zeros((len(mc.EE), 6))


for n in range(6):
    
    now_s_diff = diff_6[n]
    
    for i in range(len(mc.EE)):
        
        E = mc.EE[i]
        
        Eb = gryz.Si_MuElec_Eb[n]
        
        if n == 0:
            inds = np.where(mc.EE <= (E+Eb)/2)
        else:
            inds = np.where(np.logical_and(mc.EE >= Eb, mc.EE <= (E+Eb)/2))
        
        s_test[i, n] = np.trapz(now_s_diff[i, inds], x=mc.EE[inds])
    
    
    plt.loglog(mc.EE, s_test[:, n])


plt.xlim(1e+1, 1e+4)
plt.ylim(1e-3, 1e+3)
