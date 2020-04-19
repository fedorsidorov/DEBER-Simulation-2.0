#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import my_constants as mc
import my_utilities as mu
from scipy.optimize import curve_fit

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce'
        ))


#%%
#EE_eV = np.logspace(0, 4.4, 1000)
EE_eV = np.logspace(-1, 4.4, 1000)

EE = EE_eV * mc.eV

OLF_file = np.loadtxt('curves/OLF_Akkerman_fit.txt')

EE_file = OLF_file[:, 0]
OLF_file = OLF_file[:, 1]

plt.loglog(EE_file, OLF_file)

#plt.ylim(1e-6, 1e+2)


#%%
def func(EE_eV, A, E, w):
    return A*w*EE_eV / ((EE_eV**2 - E**2)**2 + (w*EE_eV)**2)

def func_h(EE_eV, A, E, w):
    return A*w*EE_eV * np.heaviside(EE_eV - E, 1) / ((EE_eV**2 - E**2)**2 + (w*EE_eV)**2)


def func_n(EE_eV,
           A1, E1, w1,
           A2, E2, w2,
           A3, E3, w3,
           A4, E4, w4,
           A5, E5, w5,
           A6, E6, w6,
           ):
    
    return func(EE_eV, A1, E1, w1) + func(EE_eV, A2, E2, w2) + func(EE_eV, A3, E3, w3) +\
           func_h(EE_eV, A4, E4, w4) + func_h(EE_eV, A5, E5, w5) + func_h(EE_eV, A6, E6, w6)


def func_n_log(EE_eV,
           A1, E1, w1,
           A2, E2, w2,
           A3, E3, w3,
           A4, E4, w4,
           A5, E5, w5,
           A6, E6, w6,
           ):

    return np.log10(func(EE_eV, A1, E1, w1) + func(EE_eV, A2, E2, w2) + func(EE_eV, A3, E3, w3) +\
           func_h(EE_eV, A4, E4, w4) + func_h(EE_eV, A5, E5, w5) + func_h(EE_eV, A6, E6, w6))


params = [
         120,   16,    4,
         120,   17,    4,
          15,   20,   10,
         650,  101,  140,
          60,  152,   90,
         140, 1820, 1100
         ]

#popt, pcov = curve_fit(func_n, np.log10(EE_eV), np.log10(OLF_1d), p0=params,
#                       bounds=([100,  3, 100, 2,  5, 2,  300,  50,  50,  50, 100, 1000],
#                               [150, 10, 150, 2.5, 20, 6, 1000, 200, 200, 150, 300, 3000]))

popt, pcov = curve_fit(func_n, EE_file, OLF_file, p0=params, bounds=(0, np.inf))


plt.loglog(EE_file, OLF_file, 'ro', label='Akkerman')
plt.loglog(EE_eV, func_n(EE_eV, *popt), 'b-', label='fit')

plt.grid()


#%%
plt.loglog(EE_file, OLF_file, 'r-', label='Extended')

f1 = func(EE_eV, 120, 16, 4)
f2 = func(EE_eV, 120, 17, 4)
f3 = func(EE_eV, 15, 20, 10)
f4 = func_h(EE_eV, 650, 101, 140)
f5 = func_h(EE_eV, 60, 152, 90)
f6 = func_h(EE_eV, 140, 1820, 1100)

#plt.loglog(EE_eV, f1, '--', label='1')
#plt.loglog(EE_eV, f2, '--', label='2')
#plt.loglog(EE_eV, f3, '--', label='3')
#plt.loglog(EE_eV, f4, '--', label='4')
#plt.loglog(EE_eV, f5, '--', label='5')
#plt.loglog(EE_eV, f6, '--', label='6')

#plt.loglog(EE_eV, f1+f2+f3, label='fit sum')
#plt.loglog(EE_eV, f1+f4+f5+f6, label='fit sum')
plt.loglog(EE_eV, f1+f2+f3+f4+f5+f6, '-', label='fit sum')

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

