#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
#import copy
from itertools import product

import my_functions as mf
import my_variables as mv
import my_indexes as mi
import my_constants as mc

from scipy.signal import medfilt

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mi = importlib.reload(mi)
mc = importlib.reload(mc)

os.chdir(mv.sim_path_MAC + 'mapping')

import my_mapping as mm
mm = importlib.reload(mm)

#%%
n_chains = 10000
    
## INITIAL
mat_A = np.load('Harris/mat_A_diff_Harris.npy')

x_FS = mat_A[:, 0]
y_FS = mat_A[:, 1]

x_FS_log = x_FS
S_tot = np.trapz(y_FS, x=x_FS_log)

plt.plot(x_FS_log, y_FS/S_tot, label='initial, GPC')

S_FS_log = np.zeros(len(x_FS_log))

for i in range(len(x_FS)):
    S_FS_log[i] = np.trapz(y_FS[:i+1], x=x_FS_log[:i+1]) / S_tot

#    plt.plot(x_FS_log, S_FS_log, label='sum')

n = 10000
M_arr = np.zeros(n)

for i in range(len(M_arr)):
    S_rand = mf.random()
    m = x_FS_log[mf.get_closest_el_ind(S_FS_log, S_rand)]
    M_arr[i] = 10 ** m

plt.hist(np.log10(M_arr), bins=20, normed=True, label='initial, SIM',\
         rwidth=0.8, alpha=0.5)

L_arr = M_arr / mv.u_PMMA

## FINAL
mat_B = np.load('Harris/mat_B_diff_Harris.npy')

plt.plot(mat_B[:, 0], mat_B[:, 1] * 0.7, label='final, GPC')

L_2C_ion = np.load('Harris/L_final_2.5C_exc_Harris.npy')

plt.hist(np.log10(L_2C_ion * 100), bins=20, normed=True, label='final, SIM',\
         rwidth=0.8, alpha=0.5)

plt.title('EXC on 2.5 C atoms of 5')
plt.xlabel('log(M$_n$)')
plt.ylabel('arbitrary units')
plt.legend(loc=2)
plt.ylim((0, 1))
plt.grid()
plt.show()
plt.savefig('Harris_2.5C_exc.png', dpi=300)

#%%
L_final_arr = np.load('Harris/L_final_2C_all_Harris.npy')

#%
log_mw = np.log10(L_final_arr * 100)
plt.hist(log_mw, bins=20, cumulative=False, label='sample', rwidth=0.8,\
         normed=True, alpha=0.6)

data_B = np.loadtxt(mv.sim_path_MAC + 'make_chains/harris1973_B.dat')

x_B = data_B[:, 0]
y_B = data_B[:, 1]

x_B_log = np.log10(x_B)
X = np.linspace(x_B_log[0], x_B_log[-1], 200)
Y = mf.log_interp1d(x_B_log, y_B)(X)

Y = medfilt(Y, 5)

X_diff = X[:-1]
Y_diff = np.diff(Y)

#plt.plot(np.log10(x_B), y_B, label='model')
plt.plot(X_diff, Y_diff/np.max(Y_diff), label='model')

plt.title('Harris chain mass distribution after exposure, 2C ion+exc')
plt.xlabel('log(m$_w$)')
plt.ylabel('probability')
plt.xlim((1.5, 5.5))
plt.ylim((0, 1))
plt.legend()
plt.grid()
plt.show()

#%%
chain_sum_len_matrix_C_1, n_chains_matrix_C_1 =\
    mm.get_local_chain_len(resist_shape, N_mon_chain_max, chain_table, N_chains_total)

#%%
np.save('chain_sum_len_matrix_C_1.npy', chain_sum_len_matrix_C_1)
np.save('n_chains_matrix_C_1.npy', n_chains_matrix_C_1)

#%%
sci_avg = np.average(sci_per_mol_matrix)

g = sci_avg * 1.19e-21 * 6.02e+23 / (np.sum(dE_matrix) / (100*100*500) * 9.5e+5)

print(g)
