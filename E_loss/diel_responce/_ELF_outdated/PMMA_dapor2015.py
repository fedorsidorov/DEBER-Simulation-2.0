#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu

#from itertools import product

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce'
        ))


#%%
#EE_eV = np.logspace(0, 4.4, 1000)
#EE_eV = np.logspace(-1, 4.4, 2000)
#EE_eV = np.linspace(0.01, 1e+4, 1000)

#EE = EE_eV * mc.eV
#qq = np.sqrt(2*mc.m*EE)

EE_eV = mc.EE

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
#OLF = np.zeros((len(EE), len(EE)))
OLF_1d = np.zeros(len(mc.EE))
#
#
#for i in range(len(EE_eV)):
#    
#    mu.pbar(i, len(EE_eV))
#    
#    for j in range(len(qq)):
#
#        for arr in params:
#        
#            En, Gn, An, = arr
#            
#            Enq = En + qq[j]**2 / (2*mc.m) / mc.eV
#            
#            OLF[i, j] += An * Gn * EE_eV[i] / ((Enq**2 - EE_eV[i]**2)**2 + (Gn * EE_eV[i])**2)


for arr in params:
    
    E, G, A, = arr
    OLF_1d += A*G*EE_eV / ((E**2 - EE_eV**2)**2 + (G*EE_eV)**2)


#%%
#plt.loglog(EE_eV, OLF[:, 0], label='OLF, q = 1')
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


#%%
tau_saved = np.load('PMMA_dapor2015/PMMA_tau_dapor2015.npy')

plt.loglog(EE_eV, tau[1500], 'o', label='my')
plt.loglog(EE_eV, tau_saved[1500], label='saved')

#plt.xlim(1, 1e+4)


#%%
S = np.zeros(len(EE))
u = np.zeros(len(EE))


for i in range(len(EE)):
    
    inds = np.where(EE <= EE[i]/2)
    
    S[i] = np.trapz(tau[i, inds] * EE[inds], x=EE[inds])
    u[i] = np.trapz(tau[i, inds], x=EE[inds])


#%%
plt.loglog(EE_eV, S / mc.eV / 1e+10, label='my')

S_Dapor = np.loadtxt('curves/S_dapor2015.txt')
plt.loglog(S_Dapor[:, 0], S_Dapor[:, 1], label='dapor2015.pdf')

S_Ciappa = np.loadtxt('curves/dEds_solid.txt')
plt.semilogx(S_Ciappa[:, 0], S_Ciappa[:, 1], label='ciappa2010.pdf')

EE_eV_Ashley = np.load('ashley1988/EE_eV.npy')

S_Ashley = np.load('ashley1988/SP.npy')
plt.semilogx(EE_eV_Ashley, S_Ashley / mc.eV / 1e+10, label='ashley1988')

S_exc_Ashley = np.load('ashley1988/SP_exc.npy')
plt.semilogx(EE_eV_Ashley, S_exc_Ashley / mc.eV / 1e+10, label='ashley1988_exc')

plt.xlim(1, 1e+4)
plt.ylim(0, 4)

plt.legend()
plt.grid()

#plt.savefig('S 0.2, 4.4, 1000.png', dpi=300)


#%%
plt.semilogx(EE_eV, 1/u * 1e+10, label='my')

l_Dapor = np.loadtxt('curves/l_dapor2015.txt')
plt.semilogx(l_Dapor[:, 0], l_Dapor[:, 1], label='dapor2015.pdf')

l_Ciappa = np.loadtxt('curves/IMFP_solid.txt')
plt.semilogx(l_Ciappa[:, 0], l_Ciappa[:, 1], label='ciappa2010.pdf')

EE_eV_Ashley = np.load('ashley1988/EE_eV.npy')

u_Ashley = np.load('ashley1988/u.npy')
plt.semilogx(EE_eV_Ashley, 1 / u_Ashley * 1e+10, label='ashley1988')

u_exc_Ashley = np.load('ashley1988/u_exc.npy')
plt.semilogx(EE_eV_Ashley, 1 / u_exc_Ashley * 1e+10, label='ashley1988_exc')

plt.xlim(20, 1.1e+4)
plt.ylim(0, 250)

plt.legend()
plt.grid()


#%%
tau_int = mu.diff2int(tau, EE, EE)


#%%
plt.loglog(EE, tau_int[1500, :])


#%% Interpolate to common EE
u_f = mu.log_interp1d(EE_eV, u)(mc.EE)
S_f = mu.log_interp1d(EE_eV, S)(mc.EE)

#plt.loglog(mc.EE, u_f)
#plt.loglog(mc.EE, S_f*2e+17)


#%%
tau_pre = np.zeros((len(EE), len(mc.EE)))
tau_int_pre = np.zeros((len(EE), len(mc.EE)))


for i in range(len(EE)):
    
    mu.pbar(i, len(EE))
    
    tau_pre[i, :] = mu.log_interp1d(EE_eV, tau[i, :])(mc.EE)
    tau_int_pre[i, :] = mu.log_interp1d(EE_eV, tau_int[i, :])(mc.EE)


#%%
tau_f = np.zeros((len(mc.EE), len(mc.EE)))
tau_int_f = np.zeros((len(mc.EE), len(mc.EE)))


for i in range(len(mc.EE)):
    
    mu.pbar(i, len(mc.EE))
    
    tau_f[:, i] = mu.log_interp1d(EE_eV, tau_pre[:, i])(mc.EE)
    tau_int_f[:, i] = mu.log_interp1d(EE_eV, tau_int[:, i])(mc.EE)


#%%
plt.loglog(mc.EE, tau_int_f[500, :])

