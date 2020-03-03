#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce'
        ))


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

EE_eV = np.logspace(0, 4.4, 1000)
EE = EE_eV * mc.eV

x3 = EE_eV[-1]
y3 = y2 * np.exp( np.log(y2/y1) * np.log(x3/x2) / np.log(x2/x1) )

E_arr_new = np.append(E_arr, [x3])
Im_arr_new = np.append(Im_arr, [y3])

OLF_1d = mu.log_interp1d(E_arr_new, Im_arr_new)(EE_eV)


#%%
plt.loglog(EE_eV, OLF_1d, 'ro', label='Extended')
plt.loglog(E_arr, Im_arr, '.', label='Original')

plt.title('Si optical E-loss function from Palik data')
plt.xlabel('E, eV')
plt.ylabel('Im[1/eps]')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Si_OELF_Palik.png', dpi=300)


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


#%% No exchange
h2si = mc.k_el * mc.m * mc.e**2 / mc.hbar**2

tau = np.zeros((len(EE), len(EE)))
u = np.zeros(len(EE))
S = np.zeros(len(EE))

tau_exc = np.zeros((len(EE), len(EE)))
u_exc = np.zeros(len(EE))
S_exc = np.zeros(len(EE))


for i in range(len(EE)):
    
    mu.pbar(i, len(EE))

    E = EE[i]

    for j in range(len(EE)):
        
        w = EE[j]
        wp = EE
        
        Y = h2si * 1/(2*np.pi*E) * wp * OLF_1d * F(E, wp, w)
        
        A = F(E, wp, w)
        B = F_arr(E, wp, E+wp-w)
        
        Y_exc = h2si * 1/(2*np.pi*E) * wp * OLF_1d * (A + B - np.sqrt(A * B))
        
        tau[i, j] = np.trapz(Y, x=EE)
        tau_exc[i, j] = np.trapz(Y_exc, x=EE)
    
    
    inds = np.where(EE <= EE[i]/2)
    
    u[i] = np.trapz(tau[i, inds], x=EE[inds])
    S[i] = np.trapz(tau[i, inds] * EE[inds], x=EE[inds])
    
    u_exc[i] = np.trapz(tau_exc[i, inds], x=EE[inds])
    S_exc[i] = np.trapz(tau_exc[i, inds] * EE[inds], x=EE[inds])


#%%
plt.loglog(EE_eV, tau[450, :])


#%%
plt.semilogx(EE_eV, S / mc.eV / 1e+2, label='no exchange') ## IS BETTER
#plt.semilogx(EE_eV, S_exc / mc.eV / 1e+2, label='exchange')

S_Chan = np.loadtxt('curves/Chan_Si_S.txt')
plt.loglog(S_Chan[:, 0], S_Chan[:, 1], label='Chan')

S_MuElec = np.loadtxt('curves/Si_MuElec_S.txt')
plt.loglog(S_MuElec[:, 0], S_MuElec[:, 1] * 1e+7, label='MuElec')

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
plt.loglog(sigma_MuElec[:, 0], sigma_MuElec[:, 1] * 1e-18 * mc.n_Si, label='MuElec')

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



