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

os.chdir(mv.sim_path_MAC + 'E_loss')

#%%
Palik_arr = np.loadtxt('curves/Palik_Si_E_n_k.txt')[:-3, :]

E_arr_Palik = Palik_arr[::-1, 0]
n_arr = Palik_arr[::-1, 1]
k_arr = Palik_arr[::-1, 2]

plt.loglog(E_arr_Palik, n_arr, label='n')
plt.loglog(E_arr_Palik, k_arr, label='k')

plt.title('Palik optical measurements for Si')
plt.xlabel('E, eV')
plt.ylabel('value')

plt.legend()
plt.ylim(1e-6, 1e+1)
plt.grid()
plt.show()

#%%
E_arr = np.logspace(0, 3.3, 5000)

N_arr = mf.log_interp1d(E_arr_Palik, n_arr)(E_arr)
K_arr = mf.log_interp1d(E_arr_Palik, k_arr)(E_arr)

plt.loglog(E_arr, N_arr, '.')
plt.loglog(E_arr, K_arr, '.')

#%%
Im_arr = 2 * N_arr * K_arr / ( (N_arr**2 - K_arr**2)**2 + (2*N_arr*K_arr)**2 )

plt.loglog(E_arr, Im_arr, label='Im[$1/\epsilon$]')

#plt.legend()
#plt.xlabel('E, eV')
#plt.ylabel('Im[1/eps]')
#plt.grid()
#plt.show()

#%%
eps = (N_arr**2 - K_arr**2) + (2*N_arr*K_arr)*1j

Im_arr_2 = np.imag(-1/eps)

plt.loglog(E_arr, Im_arr_2, 'r.', label='Im[$1/\epsilon$] 2')

plt.legend()
plt.xlabel('E, eV')
plt.ylabel('Im[1/eps]')
plt.grid()
plt.show()

#%%
def L(x):
    f = (1-x)*np.log(4/x) - 7/4*x + x**(3/2) - 33/32*x**2
    return f

def S(x):
    f = np.log(1.166/x) - 3/4*x - x/4*np.log(4/x) + 1/2*x**(3/2) - x**2/16*np.log(4/x) - 31/48*x**2
    return f

#%%
IMFP_inv_arr = np.zeros(len(E_arr))
dEds_arr = np.zeros(len(E_arr))

diff_IMFP_inv_arr = np.zeros((len(E_arr), len(E_arr)))

for i in range(len(E_arr)):
    
    now_E = E_arr[i]
    
    inds = np.where(E_arr <= now_E/2)[0]
    
    y_IMFP = Im_arr[inds] * L(E_arr[inds]/E_arr[i])
    y_dEds = Im_arr[inds] * S(E_arr[inds]/E_arr[i]) * E_arr[inds]*mc.eV
    
    IMFP_inv_arr[i] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 * E_arr[i]*mc.eV) *\
        np.trapz(y_IMFP, x=E_arr[inds]*mc.eV)
        
    dEds_arr[i] = mc.k_el * mc.m * mc.e**2 / (np.pi * mc.hbar**2 * E_arr[i]*mc.eV) *\
        np.trapz(y_dEds, x=E_arr[inds]*mc.eV)
    
    diff_IMFP_inv_arr[i, inds] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 *\
        E_arr[i]*mc.eV) * Im_arr[inds] * L(E_arr[inds]/E_arr[i])

#%%
#plt.loglog(E_arr, 1/IMFP_inv_arr * 1e+2, label='My')
plt.loglog(E_arr, dEds_arr / mc.eV / 1e+2, label='My')

#plt.xlim(10, 10000)
#plt.ylim(1, 1000)

plt.xlabel('E, eV')
plt.ylabel('IMFP, $\AA$')
plt.legend()
plt.grid()
plt.show()

#%%
U_SI_PALIK = IMFP_inv_arr * 1e+2
SP_SI_PALIK = dEds_arr / mc.eV / 1e+2

#%%
sun_hui = np.loadtxt('curves/sun hui v chai.txt')

plt.loglog(sun_hui[:, 0], sun_hui[:, 1])

#%%

