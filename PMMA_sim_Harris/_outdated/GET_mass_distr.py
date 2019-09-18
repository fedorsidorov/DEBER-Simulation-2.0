#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(mv.sim_path_MAC + 'make_chains')

import importlib

import my_functions as mf
mf = importlib.reload(mf)

#%%
mat = np.loadtxt('curves/sharma_peak_B.dat')
x_log = np.log10(mat[1:, 0])
y = mat[1:, 1]
X_LOG = np.arange(x_log[0], x_log[-1], 1e-2)
Y_LOG = mf.log_interp1d(x_log, y)(X_LOG)
#plt.plot(X_LOG, Y_LOG)
plt.semilogx(X_LOG, Y_LOG, 'ro')
plt.show()

#%%
S_arr = np.zeros(np.size(X_LOG))
s_tot = np.trapz(Y_LOG, x=X_LOG)

for i in range(len(S_arr) - 1):
    S_arr[i] = np.trapz(Y_LOG[0:i+1], x=X_LOG[0:i+1])/s_tot

S_arr[-1] = 1

plt.plot(X_LOG, S_arr)
plt.show()

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



