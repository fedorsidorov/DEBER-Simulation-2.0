#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu

from itertools import product

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
OLF = np.zeros((len(EE), len(EE)))
OLF_1d = np.zeros(len(EE))


for i in range(len(EE_eV)):
    
    mu.pbar(i, len(EE_eV))
    
    for j in range(len(qq)):

        for arr in params:
        
            En, Gn, An, = arr
            
            Enq = En + qq[j]**2 / (2*mc.m) / mc.eV
            
            OLF[i, j] += An * Gn * EE_eV[i] / ((Enq**2 - EE_eV[i]**2)**2 + (Gn * EE_eV[i])**2)


for arr in params:
    
    E, G, A, = arr
    OLF_1d += A*G*EE_eV / ((E**2 - EE_eV**2)**2 + (G*EE_eV)**2)


#%%
plt.loglog(EE_eV, OLF[:, 400], label='OLF, q = 1')
plt.loglog(EE_eV, OLF_1d, '--', label='OLF, q = 0')

plt.xlabel('E, eV')
plt.ylabel('Im[-1/$\epsilon(\omega, 0)$]')

plt.xlim(1, 1e+4)
plt.ylim(1e-9, 1e+1)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('oscillators_Dapor.png', dpi=300)


#%% Dapor
TT = EE
tau = np.zeros((len(EE), len(EE)))
tau_h = np.zeros((len(EE), len(EE)))


for i in range(len(TT)):
    
    T = TT[i]
    
    mu.pbar(i, len(EE))
    
    for j in range(len(EE)):
        
        E = EE[j]
        
        if T - E < 0:
            continue
        
        qp = np.sqrt(2*mc.m)*(np.sqrt(T) + np.sqrt(T - E))
        qm = np.sqrt(2*mc.m)*(np.sqrt(T) - np.sqrt(T - E))
        
        inds = np.where(np.logical_and(qq >= qm, qq <= qp))[0]
        
        Y = 1/qq[inds] * OLF[j, inds]
        tau[i, j] = h2si * 1 / (np.pi * EE[i]) * np.trapz(Y, x=qq[inds])
        
        Y_h = 1/qq * OLF[j, :] * np.heaviside(qq - qm, 1) * np.heaviside(qp - qq, 1)
        tau_h[i, j] = h2si * 1 / (np.pi * EE[i]) * np.trapz(Y_h, x=qq)
        

#%%
plt.loglog(EE_eV, tau[8000], '.')

#plt.xlim(1, 1e+4)


#%%
SS = np.zeros(len(EE))
uu = np.zeros(len(EE))


for i in range(len(EE)):
    
#    inds = np.where(EE <= EE[i]/2)
    
#    S[i] = np.trapz(tau[i, inds] * EE[inds], x=EE[inds])
    
    SS[i] = np.trapz(tau[i] * EE * \
         np.heaviside(EE[i]/2 - EE, 1), x=EE)
    
    uu[i] = np.trapz(tau[i], x=EE)


#%% Na easichah
def L(x):
    f = (1-x)*np.log(4/x) - 7/4*x + x**(3/2) - 33/32*x**2
    return f


def S(x):
    f = np.log(1.166/x) - 3/4*x - x/4*np.log(4/x) + 1/2*x**(3/2) - x**2/16*np.log(4/x) - 31/48*x**2
    return f


#%% Dapor
U_D = np.zeros(len(EE))
SP_D = np.zeros(len(EE))

U_DIFF_D = np.zeros((len(EE), len(EE)))

for i in range(len(EE)):
    
    mu.pbar(i, len(EE))
    
    E = EE[i]
    
    inds = np.where(EE <= E/2)[0]
    
    y_U = OLF_1d[inds] * L(EE[inds]/E)
    y_SP = OLF_1d[inds] * S(EE[inds]/E) * EE[inds]
    
    U_D[i] = h2si * 1/(2*np.pi*E) * np.trapz(y_U, x=EE[inds])
        
    SP_D[i] = h2si * 1/(np.pi*E) * np.trapz(y_SP, x=EE[inds])
    
    U_DIFF_D[i, inds] = h2si * 1/(2*np.pi*EE[i]) * OLF_1d[inds] * L(EE[inds]/E)


#%%
plt.semilogx(EE_eV, SS / mc.eV / 1e+10, label='my')
plt.semilogx(EE_eV, SP_D / mc.eV / 1e+10, label='my easy')

S_Dapor = np.loadtxt('curves/S_dapor2015.txt')
plt.semilogx(S_Dapor[:, 0], S_Dapor[:, 1], label='dapor2015.pdf')

S_Ciappa = np.loadtxt('curves/dEds_solid.txt')
plt.semilogx(S_Ciappa[:, 0], S_Ciappa[:, 1], label='ciappa2010.pdf')

plt.xlim(1, 1e+4)
plt.ylim(0, 4)

#plt.plot(np.log10(EE_eV), S / mc.eV / 1e+10, label='my')
#S_Dapor = np.loadtxt('curves/S_dapor2015_log.txt')
#plt.plot(S_Dapor[:, 0], S_Dapor[:, 1], label='Dapor')
#plt.xlim(1, 4)

plt.legend()
plt.grid()

#plt.savefig('S 0.2, 4.4, 1000.png', dpi=300)


#%%
plt.semilogx(EE_eV, 1/uu * 1e+10, label='my')
plt.semilogx(EE_eV, 1/U_D * 1e+10, label='my easy')

l_Dapor = np.loadtxt('curves/l_dapor2015.txt')
plt.semilogx(l_Dapor[:, 0], l_Dapor[:, 1], label='dapor2015.pdf')

l_Ciappa = np.loadtxt('curves/IMFP_solid.txt')
plt.semilogx(l_Ciappa[:, 0], l_Ciappa[:, 1], label='ciappa2010.pdf')

plt.xlim(20, 1.1e+4)
plt.ylim(0, 250)

plt.legend()

plt.grid()


