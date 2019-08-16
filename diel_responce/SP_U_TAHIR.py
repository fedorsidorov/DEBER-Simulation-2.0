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

os.chdir(mv.sim_path_MAC + 'diel_responce')

#%%
E = np.logspace(0, 4.4, 1000)
IM = np.zeros(len(E))

## hw0i, Ai, hgi from tahir2015.pdf
ai = 0.05
Eg = 5.0

params = [
        [19.7, 129.22, 10.5],
        [23.8, 60.45, 7.2],
        [30.0, 159.6, 13.0]
        ]

for arr in params:
    
    ## k == 0 ??
    
    hw0i, Ai, hgi, = arr
    
    inds = np.where(E >= Eg)
    
    IM[inds] += Ai*hgi*E[inds] / ((hw0i**2 - E[inds]**2)**2 + (hgi*E[inds])**2)


plt.loglog(E, IM, 'r.', label='oscillators')

plt.title('Tahir Im[-1/eps]')
plt.xlabel('E, eV')
plt.ylabel('Im[-1/eps]')

plt.ylim(1e-9, 1e+1)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('oscillators_Tahir.png', dpi=300)

#%%
#EE = np.logspace(0, 4.4, 1000)
#
#IM = np.zeros(len(EE))
#
### En, Gn, An from dapor2015.pdf
#params = [
#        [19.46, 8.770, 100.0],
#        [25.84, 14.75, 286.5],
#        [300.0, 140.0, 80.0],
#        [550.0, 300.0, 55.0],
#        ]
#
#for arr in params:
#    
#    E, G, A, = arr
#    IM += A*G*EE / ((E**2 - EE**2)**2 + (G*EE)**2)

#%%
def v(A):
    S = np.sqrt(1 - 2*A)
    V = 2*S/( (1+A)*(1+A+S) ) + np.log( ( (1-A**2)*(1+A) ) / ( (1-A-S)*(1+A+S)**2 ) )
    return V

def w(A):
    S = np.sqrt(1 - 2*A)
    W = (3*A**2 + 3*A + 1)/(1+A)**2 * np.log((1+A-S)/(1+A)) + np.log((1-A)/(1-A-S)) +\
        (2*A**2+A)/(1+A)**2 * np.log((1+A)/(1+A+S)) + 2*A*S/((1+A)**2 * (1+A+S))
    return W

#def g(A):
#    G = np.zeros(len(A))
#    for i in range(len(A)):
#        a = A[i]
#        s = np.sqrt(1 - 2*a)
#        u1 = (1+a-s)/2
#        u2 = (1+a)/2
#        u = np.logspace(u1, u2, 1000)
#        G[i] = np.log((1-a**2)/((1-a-s)*(1+a+s))) + 1/a*np.log(((1+a)*(1-a+s))/((1-a)*(1+a+s))) +\
#        np.trapz(np.sqrt(u/((1-u)*(u-a)*(1+a-u))), x=u)
#    return G
        
#%%
hw = np.logspace(0, 4.4, 1000)

E = np.logspace(0, 4.4, 1000)

U = np.zeros(len(hw))
SP = np.zeros(len(hw))
SP_0 = np.zeros(len(hw))

U_DIFF = np.zeros((len(hw), len(hw)))

for i in range(len(hw)):
    
    now_E = hw[i]
    
    inds = np.where(hw <= now_E/2)[0]
    
    y_U = IM[inds] * w(hw[inds]/now_E)
    y_SP = IM[inds] * v(hw[inds]/now_E) * hw[inds]*mc.eV
#    y_SP_0 = IM[inds] * g(hw[inds]/now_E) * hw[inds]*mc.eV
    
    U[i] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 * now_E*mc.eV) *\
        np.trapz(y_U, x=hw[inds]*mc.eV) * 1e-2
        
    SP[i] = mc.k_el * mc.m * mc.e**2 / (np.pi * mc.hbar**2 * now_E*mc.eV) *\
        np.trapz(y_SP, x=hw[inds]*mc.eV) / mc.eV * 1e-2
    
#    SP_0[i] = mc.k_el * mc.m * mc.e**2 / (np.pi * mc.hbar**2 * now_E*mc.eV) *\
#        np.trapz(y_SP_0, x=E[inds]*mc.eV) / mc.eV * 1e-2
    
    U_DIFF[i, inds] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 * now_E*mc.eV) *\
        IM[inds] * w(hw[inds]/now_E) * 1e-2

#%%
U_DAPOR = np.load('Dapor/U_PMMA_DAPOR.npy')
U_TAHIR = np.loadtxt('curves/U_Tahir.txt')

plt.loglog(E, 1/U_DAPOR, label='Dapor')
plt.loglog(U_TAHIR[:, 0], U_TAHIR[:, 1] * 1e-8, label='Tahir paper')
plt.loglog(E, 1/U, label='Tahir')

#plt.title('PMMA inverse IMFP')
plt.title('PMMA IMFP')
plt.xlabel('E, eV')
#plt.ylabel('$\mu$, cm$^{-1}$')
plt.ylabel('IMFP, cm')

#plt.ylim(1e+3, 1e+8)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('U_Dapor_VS_Tahir.png', dpi=300)

#%%
SP_DAPOR = np.load('Dapor/SP_PMMA_DAPOR.npy')
SP_TAHIR = np.loadtxt('curves/SP_Tahir.txt')

plt.semilogx(E, SP_DAPOR, label='Dapor')
plt.semilogx(SP_TAHIR[:, 0], SP_TAHIR[:, 1] * 1e+8, label='Tahir paper')
plt.semilogx(E, SP, label='Tahir')
#plt.semilogx(E, SP_0, label='Tahir_0')

plt.title('PMMA stopping power')
plt.xlabel('E, eV')
plt.ylabel('SP, eV/cm')

plt.legend()
plt.grid()
plt.show()

#%%
for i in range(0, len(E_arr), 1000):
    
    plt.loglog(E_arr, diff_IMFP_inv_arr[i, :])

#%%
IMFP_inv_arr_test = np.zeros(len(E_arr))

for i in range(len(E_arr)):
    
    now_E = E_arr[i]
    
    inds = np.where(E_arr <= now_E/2)[0]
    
    y_IMFP = Im_arr[inds] * L(E_arr[inds]/E_arr[i])
    y_dEds = Im_arr[inds] * S(E_arr[inds]/E_arr[i]) * E_arr[inds]*mc.eV
    
    IMFP_inv_arr_test[i] = np.trapz(diff_IMFP_inv_arr[i, :], x=E_arr*mc.eV)

#%%
plt.loglog(E_arr, 1/IMFP_inv_arr_test * 1e+10, label='My')

#%%
E_DAPOR = E_arr
U_DAPOR = IMFP_inv_arr * 1e-2
SP_DAPOR = dEds_arr / mc.eV / 1e+2
