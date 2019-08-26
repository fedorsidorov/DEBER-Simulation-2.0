#%% Import
import numpy as np
import os
import importlib

import matplotlib.pyplot as plt

import MC_functions as mcf
import my_constants as mc

mc = importlib.reload(mc)
mcf = importlib.reload(mcf)

os.chdir(mc.sim_path_MAC + 'e-beam_sim')

import plot_data as pd
pd = importlib.reload(pd)


#%%
## Usual
n_files = 1
n_tracks = 1

d_PMMA = 500e-7
E0 = 10e+3

num = 0

while num < n_files:
    
    DATA = mcf.get_DATA(d_PMMA, E0, n_tracks)
    
    DATA_PMMA = DATA[np.where(np.logical_and(DATA[:, 2] == 0, np.abs(DATA[:, 3]) == 1))]
    
#    fname = 'e_DATA/DATA_' + str(num) + '.npy'
#    np.save(fname, DATA)
    
#    print('file ' + fname + ' is ready')

    num += 1


#%%
pd.plot_DATA(DATA, d_PMMA)


#%%
fig, ax = plt.subplots()

num = 900

for i in range(num, num+100):
#for i in range(860, 870):
    
    now_DATA = np.load('e_DATA/DATA_' + str(i) + '.npy')
    
    for tn in range(int(np.max(now_DATA[:, 0]))):
        
#        now_DATA = now_DATA[np.where(now_DATA[:, 7] < d_PMMA*2)]
        
        if len(np.where(now_DATA[:, 0] == tn)[0]) == 0:
            continue
        
        beg = np.where(now_DATA[:, 0] == tn)[0][0]
        end = np.where(now_DATA[:, 0] == tn)[0][-1] + 1
        
        if now_DATA[end-1, 2] == 1:
            continue
        
        ax.plot(now_DATA[beg:end, 5], now_DATA[beg:end, 7])


mult = 5

points = np.linspace(-d_PMMA*mult, d_PMMA*mult, 100)

ax.plot(points, np.zeros(len(points)), 'k')
ax.plot(points, np.ones(len(points))*d_PMMA, 'k')

ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.gca().invert_yaxis()

plt.show()


#%% Si
EE = mc.EE
Si_SP = np.load(mc.sim_path_MAC + 'E_loss/diel_responce/Palik/Si_SP_Palik.npy')

plt.loglog(EE, Si_SP)


#%%
RR = np.zeros(len(EE))

for i in range(70, len(EE)):
    
    inds = np.where(np.logical_and(EE >= 3, EE < EE[i]))
    
    RR[i] = np.trapz(Si_SP[inds]**(-1), x=mc.EE[inds])


#%%
plt.loglog(EE, RR)

plt.grid()
plt.ylim(1e-3, 1e-2)
