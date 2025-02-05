q#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt

import my_constants as mc
import my_utilities as mu
import MC_functions_osc as mcf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
mcf = importlib.reload(mcf)

os.chdir(os.path.join(mc.sim_folder, 'e-beam_sim'))

import plot_data as pd
pd = importlib.reload(pd)


#%% Harris
d_PMMA = 500e-7
E0 = 10e+3

## EXP
#d_PMMA = 900e-7
#E0 = 20e+3

## Aktari
#d_PMMA = 100e-7
#E0 = 10e+3

## Si only
#d_PMMA = -1
#E0 = 20e+3

n_files = 10000
n_tracks = 10

num = 0


while num < n_files:
    
    DATA = mcf.get_DATA(d_PMMA, E0, n_tracks)
    
    DATA_PMMA = DATA[np.where(DATA[:, 2] == 0)]
    DATA_PMMA_inel = DATA_PMMA[np.where(DATA_PMMA[:, 3] != 0)]
#    DATA_PMMA_prim = DATA_PMMA[np.where(np.isnan(DATA_PMMA[:, 1]))]
    
    fname_PMMA_inel = '../e_DATA/Harris/osc/DATA_PMMA_inel_' + str(num) + '.npy'
#    fname_PMMA_prim = '../e_DATA/primary/Si_osc/DATA_Si_prim_' + str(num) + '.npy'

    np.save(fname_PMMA_inel, DATA_PMMA_inel)
#    np.save(fname_PMMA_prim, DATA_PMMA_prim)
    
    print('file ' + fname_PMMA_inel + ' is ready')

    num += 1


#%%
pd.plot_DATA(DATA, 2e-4)


#%% 473 - max
Z_MAX = np.zeros(1000)
inds = np.zeros(1000)
j = 0

for i in range(473):
#for i in range(860, 870):
    
    mu.upd_progress_bar(i, 473)
    
    now_DATA = np.load('../e_DATA/e_DATA_Harris/DATA_' + str(i) + '.npy')
    
    for tn in range(int(np.max(now_DATA[:, 0]))):
        
#        now_DATA = now_DATA[np.where(now_DATA[:, 7] < d_PMMA*2)]
        
        if len(np.where(now_DATA[:, 0] == tn)[0]) == 0:
            continue
        
        beg = np.where(now_DATA[:, 0] == tn)[0][0]
        end = np.where(now_DATA[:, 0] == tn)[0][-1] + 1
        
        now_TRACK = now_DATA[beg:end, :]
        
        if now_TRACK[-1, 2] == 0 and np.max(now_TRACK[:, 7]) > d_PMMA*6/5:
            Z_MAX[j] = np.max(now_TRACK[:, 7])
            inds[j] = i
            j += 1
        
#        ax.plot(now_DATA[beg:end, 5], now_DATA[beg:end, 7])


#%%
fig, ax = plt.subplots()

i = 295

now_DATA = np.load('../e_DATA/e_DATA_Harris/DATA_' + str(i) + '.npy')

#for i in range(180, 181):
    
#    mu.upd_progress_bar(i, 473)
    
#    now_DATA = np.load('../e_DATA/e_DATA_test/DATA_' + str(i) + '.npy')
    
for tn in range(int(np.max(now_DATA[:, 0]))):
        
    if len(np.where(now_DATA[:, 0] == tn)[0]) == 0:
        continue
    
    beg = np.where(now_DATA[:, 0] == tn)[0][0]
    end = np.where(now_DATA[:, 0] == tn)[0][-1] + 1
    
    ax.plot(now_DATA[beg:end, 5], now_DATA[beg:end, 7])

now_DATA = np.load('../e_DATA/e_DATA_Harris/DATA_' + str(i) + '.npy')
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


#%%
for i in range(21):
    
    now_DATA = np.load('../e_DATA/e_DATA_Harris_cut_1.5e-4/DATA_' + str(num) + '.npy')
    
    now_DATA_PMMA = now_DATA[np.where(np.logical_and(now_DATA[:, 2] == 0,\
                                    np.abs(now_DATA[:, 3]) == 1))]
    
    fname = '../e_DATA/e_DATA_Harris_PMMA_cut_1.5e-4/DATA_PMMA_' + str(i) + '.npy'
    
    np.save(fname, now_DATA_PMMA)
    
    print('file ' + fname + ' is ready')

    num += 1


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
#