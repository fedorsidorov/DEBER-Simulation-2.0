#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu
from scipy import integrate

#from itertools import product

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce'
        ))


#%%
#EE_eV = np.logspace(0, 4.4, 1000)
EE_eV = np.logspace(0, 4.4, 500)
#EE_eV = np.linspace(0.01, 1e+4, 1000)

EE = EE_eV * mc.eV
qq = np.sqrt(2*mc.m*EE)

qq_eV = np.sqrt(2*mc.m*EE_eV)

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
ELF = np.zeros((len(EE), len(EE)))
OLF = np.zeros(len(EE))


for i in range(len(EE_eV)):
    
    mu.pbar(i, len(EE_eV))
    
    for j in range(len(qq)):

        for arr in params:
            En, Gn, An, = arr
            Enq = En + qq_eV[j]**2 / (2*mc.m)
            ELF[i, j] += An * Gn * EE_eV[i] / ((Enq**2 - EE_eV[i]**2)**2 + (Gn * EE_eV[i])**2)


for arr in params:
    E, G, A, = arr
    OLF += A*G*EE_eV / ((E**2 - EE_eV**2)**2 + (G*EE_eV)**2)


#%%
plt.loglog(EE_eV, ELF[:, 0], label='OLF start')
plt.loglog(EE_eV, OLF, '--', label='OLF, q = 0')

plt.xlabel('E, eV')
plt.ylabel('Im[-1/$\epsilon(\omega, 0)$]')

plt.xlim(1, 1e+4)
plt.ylim(1e-9, 1e+1)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('oscillators_Dapor.png', dpi=300)


#%% Dapor
TT = EE_eV
tau = np.zeros((len(EE), len(EE)))


for i in range(len(TT)):
    
    T = TT[i]
    
    mu.pbar(i, len(EE))
    
    for j in range(len(EE)):
        
        E = EE_eV[j]
        
        if T - E < 0:
            continue
        
        qp = np.sqrt(2*mc.m)*(np.sqrt(T) + np.sqrt(T - E))
        qm = np.sqrt(2*mc.m)*(np.sqrt(T) - np.sqrt(T - E))
        
        inds = np.where(np.logical_and(qq_eV >= qm, qq_eV <= qp))[0]
        
        Y = 1/qq_eV[inds] * ELF[j, inds]
        tau[i, j] = h2si * 1 / (np.pi * EE_eV[i]) * np.trapz(Y, x=qq_eV[inds])


#%%
S = np.zeros(len(EE))
u = np.zeros(len(EE))


for i in range(len(EE)):
    
    inds = np.where(EE_eV <= EE_eV[i]/2)
    
    S[i] = np.trapz(tau[i, inds] * EE_eV[inds], x=EE_eV[inds])
    u[i] = np.trapz(tau[i, inds], x=EE_eV[inds])
    

#%%
def get_oscillator(E_eV, A, E, w, q_eV):
    Eq = E + q_eV**2 / (2*mc.m)
    return A*w*E_eV / ((E_eV**2 - Eq**2)**2 + (w*E_eV)**2)


def get_ELF(E_eV, q_eV):
    
    ELF = 0
    
    for arr in params:
        E, w, A = arr
        ELF += get_oscillator(E_eV, A, E, w, q_eV)
    
    return ELF


def get_tau(E_eV, hw_eV):
    
    if hw_eV > E_eV:
        return 0
    
    def get_ELF_q(q_eV):
        return get_ELF(hw_eV, q_eV) / q_eV
    
    qp = np.sqrt(2*mc.m)*(np.sqrt(E_eV) + np.sqrt(E_eV - hw_eV))
    qm = np.sqrt(2*mc.m)*(np.sqrt(E_eV) - np.sqrt(E_eV - hw_eV))
    
    return h2si * 1/(np.pi*E_eV) * integrate.quad(get_ELF_q, qm, qp)[0]


def get_S(E_eV):
    
    def get_tau_hw_S(hw_eV):
        return get_tau(E_eV, hw_eV) * hw_eV
    
    return integrate.quad(get_tau_hw_S, 0, E_eV/2)[0]


def get_u(E_eV):
    
    def get_tau_u(hw_eV):
        return get_tau(E_eV, hw_eV)
    
    return integrate.quad(get_tau_u, 0, E_eV/2)[0]


#%%
tau_quad = np.zeros((len(EE_eV), len(EE_eV)))

for i in range(len(EE_eV)):
    
    mu.pbar(i, len(EE_eV))

    for j in range(len(EE_eV)):
        
        tau_quad[i, j] = get_tau(EE_eV[i], EE_eV[j])


#%%
ind = 630

plt.loglog(EE_eV, tau[ind, :])
plt.loglog(EE_eV, tau_quad[ind, :])


#%%
#EE_eV = np.logspace(0, 4.4, 100)

S = np.zeros(len(EE_eV))
u = np.zeros(len(EE_eV))


for i, E_eV in enumerate(EE_eV):
    
    mu.pbar(i, len(EE_eV))
    
    S[i] = get_S(E_eV)
    u[i] = get_u(E_eV)


#%%
#plt.loglog(EE_eV, S / mc.eV / 1e+10, label='my')
plt.semilogx(EE_eV, S / 1e+10, label='my')

S_Dapor = np.loadtxt('curves/S_dapor2015.txt')
plt.semilogx(S_Dapor[:, 0], S_Dapor[:, 1], label='dapor2015.pdf')

S_Ciappa = np.loadtxt('curves/dEds_solid.txt')
plt.semilogx(S_Ciappa[:, 0], S_Ciappa[:, 1], label='ciappa2010.pdf')

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

