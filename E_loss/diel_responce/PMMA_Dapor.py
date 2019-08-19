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
EE = np.logspace(0, 4.4, 1000)

IM = np.zeros(len(EE))

## En, Gn, An from dapor2015.pdf
params = [
        [19.46, 8.770, 100.0],
        [25.84, 14.75, 286.5],
        [300.0, 140.0, 80.0],
        [550.0, 300.0, 55.0],
        ]

for arr in params:
    
    E, G, A, = arr
    IM += A*G*EE / ((E**2 - EE**2)**2 + (G*EE)**2)

plt.loglog(EE, IM, 'r.', label='oscillators')

plt.title('Dapor Im[-1/eps]')
plt.xlabel('E, eV')
plt.ylabel('Im[-1/eps]')

plt.ylim(1e-9, 1e+1)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('oscillators_Dapor.png', dpi=300)

#%%
def L(x):
    f = (1-x)*np.log(4/x) - 7/4*x + x**(3/2) - 33/32*x**2
    return f

def S(x):
    f = np.log(1.166/x) - 3/4*x - x/4*np.log(4/x) + 1/2*x**(3/2) - x**2/16*np.log(4/x) - 31/48*x**2
    return f

#%%
U = np.zeros(len(EE))
SP = np.zeros(len(EE))

U_DIFF = np.zeros((len(EE), len(EE)))

for i in range(len(EE)):
    
    E = EE[i]
    
    inds = np.where(EE <= E/2)[0]
    
    y_U = IM[inds] * L(EE[inds]/E)
    y_SP = IM[inds] * S(EE[inds]/E) * EE[inds]
    
    U[i] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 * E) *\
        np.trapz(y_U, x=EE[inds]) * 1e-2 ## cm^-1
        
    SP[i] = mc.k_el * mc.m * mc.e**2 / (np.pi * mc.hbar**2 * E) *\
        np.trapz(y_SP, x=EE[inds]) * 1e-2 ## eV/cm
    
    U_DIFF[i, inds] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 *\
        EE[i]) * IM[inds] * L(EE[inds]/E) * 1e-2 ## eV^-1 cm^-1
    

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

plt.savefig('PMMA_SP_Dapor_Tahir.png', dpi=300)

#%%
IMFP_inv_arr_test = np.zeros(len(EE))

for i in range(len(EE)):
    
    E = EE[i]
    
    inds = np.where(EE <= E/2)[0]
    
    y_IMFP = IM[inds] * L(EE[inds]/EE[i])
    y_dEds = IM[inds] * S(EE[inds]/EE[i]) * EE[inds]*mc.eV
    
    IMFP_inv_arr_test[i] = np.trapz(U_DIFF[i, :], x=EE)


plt.loglog(EE, 1/IMFP_inv_arr_test * 1e+8, label='My 2')

#%% Integrals
U_INT = np.zeros((len(EE), len(EE)))

for i in range(len(EE)):
    
    integral = np.trapz(U_DIFF[i, :], x=EE)
    
    if integral == 0:
        continue
    
    for j in range(1, len(EE)):
        
        U_INT[i, j] = np.trapz(U_DIFF[i, :j], x=EE[:j]) / integral
