#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(mv.sim_path_MAC + 'make_chains')

import importlib

import my_functions as mf
mf = importlib.reload(mf)


#%%
def get_log_mw():
    r = mf.random()
    for i in range(len(S_arr) - 1):
        if r < S_arr[i + 1]:
            return X_LOG[i]

N_chains = 10000

log_mw_arr = np.zeros(N_chains)
L_arr = np.zeros(N_chains)

for i in range(N_chains):
    log_mw = get_log_mw()
    log_mw_arr[i] = log_mw
    L_arr[i] = int(10**log_mw / 100)



