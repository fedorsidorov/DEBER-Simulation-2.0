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
EE_eV = np.logspace(-1, 4.4, 500)
EE = EE_eV * mc.eV

h2si = mc.k_el * mc.m * mc.e**2 / mc.hbar**2

IM = np.zeros(len(EE))

#WW_ext = np.logspace(-1, 4.5, 5000) * mc.eV

## En, Gn, An from dapor2015.pdf
params = [
         [19.46, 8.770, 100.0],
         [25.84, 14.75, 286.5],
         [300.0, 140.0, 80.0],
         [550.0, 300.0, 55.0],
         ]

for arr in params:
    
    E, G, A, = arr
    IM += A*G*EE_eV / ((E**2 - EE_eV**2)**2 + (G*EE_eV)**2)

plt.loglog(EE_eV, IM, 'r.', label='oscillators')

plt.title('Dapor Im[-1/eps]')
plt.xlabel('E, eV')
plt.ylabel('Im[-1/eps]')

plt.xlim(1, 1e+4)
plt.ylim(1e-9, 1e+1)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('oscillators_Dapor.png', dpi=300)


#%%
def F(E, wp, w): ## wp is array!
    
    F = np.zeros(len(wp))
    
    if E - w < 0:
        return F
    
    F = np.zeros(len(wp))
    
    qp = np.sqrt(2 * mc.m) * (np.sqrt(E) + np.sqrt(E - w))
    qm = np.sqrt(2 * mc.m) * (np.sqrt(E) - np.sqrt(E - w))
    
    inds = np.where(np.logical_and(w - qm**2/(2*mc.m) - wp >= 0, wp - w + qp**2/(2*mc.m) >= 0))
    
    if np.size(w) > 1:
        F[inds] = 1 / (w[inds]*(w[inds] - wp[inds]))
        return F
    
    F[inds] = 1 / (w * (w - wp[inds])) 
    
    return F


def F_arr(E, wp, w): ## wp and w are arrays of the same length
    
    F = np.zeros(len(wp))
    
    
    for i in range(len(F)):
        
        if E - w[i] < 0:
            continue
        
        qp = np.sqrt(2 * mc.m) * (np.sqrt(E) + np.sqrt(E - w[i]))
        qm = np.sqrt(2 * mc.m) * (np.sqrt(E) - np.sqrt(E - w[i]))
        
        if np.logical_and(w[i] - qm**2/(2*mc.m) - wp[i] >= 0,\
                wp[i] - w[i] + qp**2/(2*mc.m) >= 0):
            
            F[i] = 1 / (w[i] * (w[i] - wp[i]))
    
    
    return F


#%%
ans = F(EE[800], EE, EE[700])

plt.loglog(EE, ans)


#%% No exchange
tau = np.zeros((len(EE), len(EE)))
u = np.zeros(len(EE))
SP = np.zeros(len(EE))

tau_exc = np.zeros((len(EE), len(EE)))
u_exc = np.zeros(len(EE))
SP_exc = np.zeros(len(EE))


for i in range(len(EE)):
    
    mu.pbar(i, len(EE))

    E = EE[i]

    for j in range(len(EE)):
        
        w = EE[j]
        wp = EE
        
        Y = h2si * 1/(2*np.pi*E) * wp * IM * F(E, wp, w)
        
        A = F(E, wp, w)
        B = F_arr(E, wp, E+wp-w)
        
        Y_exc = h2si * 1/(2*np.pi*E) * wp * IM * (A + B - np.sqrt(A * B))
        
        tau[i, j] = np.trapz(Y, x=EE)
        tau_exc[i, j] = np.trapz(Y_exc, x=EE)
    
    
    inds = np.where(EE <= EE[i]/2)
    
    u[i] = np.trapz(tau[i, inds], x=EE[inds])
    SP[i] = np.trapz(tau[i, inds] * EE[inds], x=EE[inds])
    
    u_exc[i] = np.trapz(tau_exc[i, inds], x=EE[inds])
    SP_exc[i] = np.trapz(tau_exc[i, inds] * EE[inds], x=EE[inds])
    
    

#%%
plt.semilogx(EE_eV, SP / mc.eV / 1e+10, label='no exchange')
plt.semilogx(EE_eV, SP_exc / mc.eV / 1e+10, label='exchange')
#plt.semilogx(EE_eV, u / 1e+10)

plt.xlim(1, 1e+4)
plt.ylim(0, 4)

plt.legend()
plt.grid()

#plt.savefig('S 0.2, 4.4, 1000.png', dpi=300)


#%%
IMFP_solid = np.loadtxt('curves/IMFP_solid.txt')
IMFP_dashed = np.loadtxt('curves/IMFP_dashed.txt')

plt.loglog(IMFP_solid[:, 0], IMFP_solid[:, 1], label='Dapor_solid')
plt.loglog(IMFP_dashed[:, 0], IMFP_dashed[:, 1], label='Dapor_dashed')

plt.xlim(10, 10000)
plt.ylim(1, 1000)

plt.xlabel('E, eV')
plt.ylabel('IMFP, $\AA$')
plt.legend()
plt.grid()
plt.show()


#%%
dEds_solid = np.loadtxt('curves/dEds_solid.txt')
dEds_dashed = np.loadtxt('curves/dEds_dashed.txt')
dEds_dotted = np.loadtxt('curves/dEds_dotted.txt')

#SP_TAHIR = np.loadtxt('curves/SP_Tahir.txt')

plt.semilogx(dEds_solid[:, 0], dEds_solid[:, 1], label='Dapor_solid')
plt.semilogx(dEds_dashed[:, 0], dEds_dashed[:, 1], label='Dapor_dashed')
plt.semilogx(dEds_dotted[:, 0], dEds_dotted[:, 1], label='Dapor_dotted')

#plt.semilogx(EE, SP / 1e+8, label='My')
#plt.xlim(10, 10000)
#plt.ylim(0, 4)

plt.title('PMMA stopping power')

plt.xlabel('E, eV')
plt.ylabel('SP, eV/$\AA$')
plt.legend()
plt.grid()
plt.show()

#plt.savefig('PMMA_SP_Dapor_Tahir.png', dpi=300)

