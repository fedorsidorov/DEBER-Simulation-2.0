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
EE_eV = np.logspace(0, 4.4, 1000)
#EE_eV = np.logspace(-1, 4.4, 2000)
#EE_eV = np.linspace(0.01, 1e+4, 1000)

EE = EE_eV * mc.eV
qq = np.sqrt(2*mc.m*EE)

a0 = 5.29e-11 ## m

h2si = mc.k_el * mc.m * mc.e**2 / mc.hbar**2

#WW_ext = np.logspace(-1, 4.5, 5000) * mc.eV

## A, E, G
params = [
         [128,  16.127,  3.5],
         [126,  17.326,  2.5],
         [ 10,  20.875,    5],
         [800,  99.699,  120],
         [ 70, 146.660,   50],
         [200,    1777, 1000]
         ]


#%%
def func(EE_eV, A, E, w):
    return A*w*EE_eV / ((EE_eV**2 - E**2)**2 + (w*EE_eV)**2)

def func_h(EE_eV, A, E, w):
    return A*w*EE_eV * np.heaviside(EE_eV - E, 1) / ((EE_eV**2 - E**2)**2 + (w*EE_eV)**2)


#%%
OLF = np.zeros((len(EE), len(EE)))
OLF_1d = np.zeros(len(EE))


for i in range(len(EE_eV)):
    
    mu.pbar(i, len(EE_eV))
    
    for j in range(len(qq)):
        
        for arr in params[:3]:
        
            An, En, Gn, = arr
            Enq = En + qq[j]**2 / (2*mc.m) / mc.eV
            OLF[i, j] += An * Gn * EE_eV[i] / ((Enq**2 - EE_eV[i]**2)**2 + (Gn * EE_eV[i])**2)
        
        
        for arr in params[3:]:
            
            An, En, Gn, = arr
            Enq = En + qq[j]**2 / (2*mc.m) / mc.eV
            
            if EE_eV[i] >= En:
                OLF[i, j] += An * Gn * EE_eV[i] / ((Enq**2 - EE_eV[i]**2)**2 + (Gn * EE_eV[i])**2)


for arr in params[:3]:
        
    An, En, Gn, = arr
    OLF_1d += An * Gn * EE_eV / ((En**2 - EE_eV**2)**2 + (Gn * EE_eV)**2)


for arr in params[3:]:
        
    An, En, Gn, = arr
    OLF_1d += An * Gn * EE_eV * np.heaviside(EE_eV - En, 1) /\
        ((En**2 - EE_eV**2)**2 + (Gn * EE_eV)**2)


#%%
plt.imshow(np.log(OLF))


#%%
Palik_arr = np.loadtxt('curves/Palik_Si_E_n_k.txt')[:-3, :]

E_arr_Palik = Palik_arr[::-1, 0]
n_arr = Palik_arr[::-1, 1]
k_arr = Palik_arr[::-1, 2]

E_arr = np.logspace(0, 3.3, 5000)

N_arr = mu.log_interp1d(E_arr_Palik, n_arr)(E_arr)
K_arr = mu.log_interp1d(E_arr_Palik, k_arr)(E_arr)

Im_arr = 2 * N_arr * K_arr / ( (N_arr**2 - K_arr**2)**2 + (2*N_arr*K_arr)**2 )


#%% Add points to Im
x1 = E_arr[-40]
y1 = Im_arr[-40]

x2 = E_arr[-1]
y2 = Im_arr[-1]

x3 = EE_eV[-1]
y3 = y2 * np.exp( np.log(y2/y1) * np.log(x3/x2) / np.log(x2/x1) )

E_arr_new = np.append(E_arr, [x3])
Im_arr_new = np.append(Im_arr, [y3])

OLF_1d_Palik = mu.log_interp1d(E_arr_new, Im_arr_new)(EE_eV)

plt.loglog(EE_eV, OLF_1d, label='OLF, q = 0')
plt.loglog(EE_eV, OLF[:, 0], label='OLF, q = 0.1')
plt.loglog(EE_eV, OLF_1d_Palik, label='OLF, Palik')

plt.xlabel('E, eV')
plt.ylabel('Im[-1/$\epsilon(\omega, 0)$]')

plt.xlim(1, 1e+4)
plt.ylim(1e-9, 1e+1)

plt.legend()
plt.grid()
plt.show()


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
S = np.zeros(len(EE))
u = np.zeros(len(EE))


for i in range(len(EE)):
    
    inds = np.where(EE <= EE[i]/2)
    
    S[i] = np.trapz(tau[i, inds] * EE[inds], x=EE[inds])
    u[i] = np.trapz(tau[i, inds], x=EE[inds])


#%%
plt.semilogx(EE_eV, S / mc.eV / 1e+2, label='my') ## IS BETTER
#plt.semilogx(EE_eV, S_exc / mc.eV / 1e+2, label='exchange')

S_Chan = np.loadtxt('curves/Chan_Si_S.txt')
plt.loglog(S_Chan[:, 0], S_Chan[:, 1], label='Chan')

S_MuElec = np.loadtxt('curves/Si_MuElec_S.txt')
plt.loglog(S_MuElec[:, 0], S_MuElec[:, 1] * 1e+7, '--', label='MuElec')

plt.xlim(1, 1e+4)
plt.ylim(1e+5, 1e+9)

plt.legend()
plt.grid()

#plt.savefig('S 0.2, 4.4, 1000.png', dpi=300)


#%%
plt.semilogx(EE_eV, u / 1e+2, label='no exchange') ## IS BETTER
#plt.semilogx(EE_eV, u_exc / 1e+2, label='exchange')

l_Chan = np.loadtxt('curves/Chan_Si_l.txt')
plt.loglog(l_Chan[:, 0], 1 / l_Chan[:, 1], label='Chan')

sigma_MuElec = np.loadtxt('curves/Si_MuElec_sigma.txt')
plt.loglog(sigma_MuElec[:, 0], sigma_MuElec[:, 1] * 1e-18 * mc.n_Si, '--', label='MuElec')

plt.xlim(1, 1e+4)
plt.ylim(1e+5, 1e+8)

plt.legend()
plt.grid()

#plt.savefig('S 0.2, 4.4, 1000.png', dpi=300)


#%%
tau_int = mu.diff2int(tau, EE, EE)


#%% Interpolate to common EE
u_f = mu.log_interp1d(EE_eV, u)(mc.EE)
S_f = mu.log_interp1d(EE_eV, S)(mc.EE)

plt.loglog(mc.EE, u_f)
plt.loglog(mc.EE, S_f*1e+17)


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

