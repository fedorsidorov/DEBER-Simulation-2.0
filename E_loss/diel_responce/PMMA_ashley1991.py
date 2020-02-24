#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce'
        ))


#%%
#EE_eV = np.logspace(0, 4.4, 1000)
EE_eV = np.logspace(-1, 4.4, 10000)

EE = EE_eV * mc.eV
qq = np.sqrt(2*mc.m*EE)

a0 = 5.29e-11 ## m

h2si = mc.k_el * mc.m * mc.e**2 / mc.hbar**2

#WW_ext = np.logspace(-1, 4.5, 5000) * mc.eV

## En, Gn, An from dapor2015.pdf
params = [
         [19.46, 8.770, 100.0],
         [25.84, 14.75, 286.5],
         [300.0, 140.0, 80.0],
         [550.0, 300.0, 55.0],
         ]


#%%
OLF_1d = np.zeros(len(EE))

for arr in params:
    
    E, G, A, = arr
    OLF_1d += A*G*EE_eV / ((E**2 - EE_eV**2)**2 + (G*EE_eV)**2)


#%%
plt.loglog(EE_eV, OLF_1d, 'ro', label='OLF, q = 0')

plt.xlabel('E, eV')
plt.ylabel('Im[-1/$\epsilon(\omega, 0)$]')

plt.xlim(1, 1e+4)
plt.ylim(1e-9, 1e+1)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('oscillators_Dapor.png', dpi=300)


#%%
tau_exc = np.zeros((len(EE), len(EE)))


for i in range(len(EE)):
    
    mu.pbar(i, len(EE))
    
    E = EE[i]

    for j in range(len(EE)):
        
        w = EE[j]
        
        w_min = 0
        
        y = w/E
        
        if w <= E/2:
            w_min = 0
        elif w <= 3/4*E:
            w_min = E*(2*y - 1)
        else:
            continue
        
        w_max = 2*E*(y - 1 + np.sqrt(1 - y))
        
        
        
        theta = np.heaviside(EE - w_min, 1) * np.heaviside(w_max - EE, 1)
        
        wp = EE
        
        F = (w*(w - wp))**(-1) + ((E + wp - w)*(E - w))**(-1) +\
            (w*(w - wp)*(E + wp - w)*(E-w))**(-1/2)
        
        tau_exc[i, j] = h2si * 1/(2*np.pi*E) * np.trapz(OLF_1d * wp * F * theta, x=EE)


#%%
IMFP_solid = np.loadtxt('curves/IMFP_solid.txt')
IMFP_dashed = np.loadtxt('curves/IMFP_dashed.txt')

plt.loglog(IMFP_solid[:, 0], IMFP_solid[:, 1], label='Dapor_solid')
plt.loglog(IMFP_dashed[:, 0], IMFP_dashed[:, 1], label='Dapor_dashed')

plt.loglog(EE, 1/U * 1e+8, label='My')

#plt.xlim(10, 10000)
#plt.ylim(1, 1000)

plt.xlabel('E, eV')
plt.ylabel('IMFP, $\AA$')
plt.legend()
plt.grid()
plt.show()


#%%
dEds_solid = np.loadtxt('curves/dEds_solid.txt')
dEds_dashed = np.loadtxt('curves/dEds_dashed.txt')
dEds_dotted = np.loadtxt('curves/dEds_dotted.txt')

SP_TAHIR = np.loadtxt('curves/SP_Tahir.txt')

plt.semilogx(dEds_solid[:, 0], dEds_solid[:, 1], label='Dapor_solid')
plt.semilogx(dEds_dashed[:, 0], dEds_dashed[:, 1], label='Dapor_dashed')
plt.semilogx(dEds_dotted[:, 0], dEds_dotted[:, 1], label='Dapor_dotted')

plt.semilogx(SP_TAHIR[:, 0], SP_TAHIR[:, 1], 'ro', label='Tahir paper')

plt.semilogx(EE, SP / 1e+8, label='My')
plt.xlim(10, 10000)
plt.ylim(0, 4)

plt.title('PMMA stopping power')

plt.xlabel('E, eV')
plt.ylabel('SP, eV/$\AA$')
plt.legend()
plt.grid()
plt.show()

#plt.savefig('PMMA_SP_Dapor_Tahir.png', dpi=300)


#%%
## E = 200 eV
#ind = 522
#ind = 682
ind = 750

plt.loglog(EE, U_DIFF[ind, :], label='Dapor')
plt.loglog(EE, U_DIFF_A[ind, :], label='Ashley')

plt.grid()
plt.legend()
plt.show()


#%%
IMFP_inv_arr_test = np.zeros(len(EE))
IMFP_inv_arr_test_A = np.zeros(len(EE))

for i in range(len(EE)):
    
    E = EE[i]
    
    inds = np.where(EE <= E/2)[0]
    
    y_IMFP = IM[inds] * L(EE[inds]/EE[i])
    y_dEds = IM[inds] * S(EE[inds]/EE[i]) * EE[inds]*mc.eV
    
    IMFP_inv_arr_test[i] = np.trapz(U_DIFF[i, :], x=EE)
    IMFP_inv_arr_test_A[i] = np.trapz(U_DIFF_A[i, :], x=EE)


plt.loglog(EE, IMFP_inv_arr_test * 1e+8, label='My')
plt.loglog(EE, IMFP_inv_arr_test_A * 1e+8, label='My A')


#%% Integrals - from Ashley!!!
U_INT_A = np.zeros((len(EE), len(EE)))

for i in range(len(EE)):
    
    mu.pbar(i, len(EE))
    
    integral = np.trapz(U_DIFF_A[i, :], x=EE)
    
    if integral == 0:
        continue
    
    for j in range(1, len(EE)):
        
        U_INT_A[i, j] = np.trapz(U_DIFF_A[i, :j], x=EE[:j]) / integral


#%%
O_core_diff = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'PMMA_O_1S_diff_U.npy'
        ))

C_core_diff = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'Gryzinski', 'PMMA', 'PMMA_C_1S_diff_U.npy'
        ))

val_diff = U_DIFF - O_core_diff - C_core_diff
val_diff_A = U_DIFF_A - O_core_diff - C_core_diff


#%%
U_val_INT_A = np.zeros((len(EE), len(EE)))

for i in range(len(EE)):
    
    mu.pbar(i, len(EE))
    
    integral = np.trapz(val_diff_A[i, :], x=EE)
    
    if integral == 0:
        continue
    
    for j in range(1, len(EE)):
        
        U_val_INT_A[i, j] = np.trapz(val_diff_A[i, :j], x=EE[:j]) / integral


#%%
IMFP_inv_val = np.zeros(len(EE))
IMFP_inv_val_A = np.zeros(len(EE))

U_loaded = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Dapor', 'PMMA_val_tot_U_D+G.npy'
        ))


for i in range(len(EE)):
    
    mu.pbar(i, len(EE))
    
    E = EE[i]
    
    inds = np.where(EE <= E/2)[0]
    
    IMFP_inv_val[i] = np.trapz(val_diff[i, inds], x=EE[inds])
    IMFP_inv_val_A[i] = np.trapz(val_diff_A[i, :], x=EE)


plt.loglog(EE, IMFP_inv_val, label='Dapor')
plt.loglog(EE, IMFP_inv_val_A, label='Ashley')
plt.loglog(EE, U_loaded, '--', label='loaded')

plt.legend()

