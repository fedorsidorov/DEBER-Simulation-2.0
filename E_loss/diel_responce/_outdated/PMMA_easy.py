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
#EE_eV = np.linspace(0.01, 1e+4, 1000)

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


#%% Na easichah
def S(x):
    f = (1-x)*np.log(4/x) - 7/4*x + x**(3/2) - 33/32*x**2
    return f


def G(x):
    f = np.log(1.166/x) - 3/4*x - x/4*np.log(4/x) + 1/2*x**(3/2) - x**2/16*np.log(4/x) - 31/48*x**2
    return f


#%% Dapor
U_D = np.zeros(len(EE))
SP_D = np.zeros(len(EE))
U_D_test = np.zeros(len(EE))
SP_D_test = np.zeros(len(EE))
U_DIFF_D = np.zeros((len(EE), len(EE)))


for i in range(len(EE)):
    
    mu.pbar(i, len(EE))
    
    E = EE[i]
    
    mult = np.heaviside(E/2 - EE, 1)
    
    y_U = OLF_1d * S(EE/E) * mult
    y_SP = OLF_1d * G(EE/E) * EE * mult
    
    U_D[i] = h2si * 1/(2*np.pi*E) * np.trapz(y_U, x=EE)
    
    SP_D[i] = h2si * 1/(np.pi*E) * np.trapz(y_SP, x=EE)
    
    U_DIFF_D[i, :] = h2si * 1/(2*np.pi*E) * OLF_1d * G(EE/E)
    
    U_D_test[i] = np.trapz(U_DIFF_D[i, :] * mult, x=EE)
    SP_D_test[i] = np.trapz(U_DIFF_D[i, :] * EE * mult, x=EE)


#%% SP
plt.semilogx(EE_eV, SP_D / mc.eV / 1e+10, 'o', label='my')

SP_D_solid = np.loadtxt('curves/dEds_solid.txt')
plt.semilogx(SP_D_solid[:, 0], SP_D_solid[:, 1], label='Dapor_solid')

plt.semilogx(EE_eV, SP_D_test * 2 / mc.eV / 1e+10, label='test')

plt.xlim(1, 1e+4)
plt.ylim(0, 4)

plt.legend()
plt.grid()



#%%
plt.semilogx(EE_eV, U_D / 1e+10, 'o', label='my')

U_D_solid = np.loadtxt('curves/IMFP_solid.txt')
plt.semilogx(U_D_solid[:, 0], 1/U_D_solid[:, 1], label='Dapor_solid')

plt.semilogx(EE_eV, U_D_test / 1e+10, label='test')

plt.xlim(1, 1e+4)
plt.ylim(0, 1.2e-1)

plt.legend()
plt.grid()


#%% Ashley 1990
U_DIFF_A = np.zeros((len(EE), len(EE)))


for i in range(len(EE)):
    
    mu.pbar(i, len(EE))
    
    now_E = EE[i]

    for j in range(len(EE)):
        
        now_w = EE[j]
        
        w_min = 0
        
        if now_w <= now_E/2:
            w_min = 0
        elif now_w <= 3*now_E/4:
            w_min = 2*now_w - now_E
        else:
            continue
        
        w_max = 2*np.sqrt(now_E - now_w)*(np.sqrt(now_E) - np.sqrt(now_E - now_w))
        
        inds = np.where(np.logical_and(EE >= w_min, EE <= w_max))[0]
        
        X = EE[inds]*mc.eV
        
        E = now_E * mc.eV
        w = now_w * mc.eV
        wp = EE[inds] * mc.eV
        
        ## From Chan thesis - W/O exchange correction
#        F = (now_w*mc.eV * (now_w - E_arr[inds])*mc.eV)**(-1)
        
        ## From ashley1990.pdf
        F = (w*(w - wp))**(-1) +\
            ((E + wp - w)*(E - w))**(-1) +\
            (w*(w - wp)*(E + wp - w)*(E-w))**(-1/2)
        
        Y = IM[inds] * EE[inds]*mc.eV * F
        
        U_DIFF_A[i, j] = mc.k_el * mc.m * mc.e**2 /\
            (2 * np.pi * mc.hbar**2 * now_E*mc.eV) * np.trapz(Y, x=X) / 1e+2 * mc.eV


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

