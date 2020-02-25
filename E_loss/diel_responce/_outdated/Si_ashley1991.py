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

x1 = E_arr[-2]
y1 = Im_arr[-2]

x2 = E_arr[-1]
y2 = Im_arr[-1]

EE_eV = mc.EE
EE = EE_eV * mc.eV

x3 = EE_eV[-1]
y3 = y2 * np.exp( np.log(y2/y1) * np.log(x3/x2) / np.log(x2/x1) )

E_arr_new = np.append(E_arr, [x3])
Im_arr_new = np.append(Im_arr, [y3])

OLF_1d = mu.log_interp1d(E_arr_new, Im_arr_new)(EE_eV)


plt.loglog(EE_eV, OLF_1d, 'ro', label='Extended')
plt.loglog(E_arr, Im_arr, label='Original')

plt.title('Si optical E-loss function from Palik data')
plt.xlabel('E, eV')
plt.ylabel('Im[1/eps]')

plt.legend()
plt.grid()
plt.show()
#plt.savefig('Si_OELF_Palik.png', dpi=300)


#%%
h2si = mc.k_el * mc.m * mc.e**2 / mc.hbar**2
tau_exc = np.zeros((len(EE), len(EE)))

S = np.zeros(len(EE))
u = np.zeros(len(EE))


for i in range(len(EE)):
    
    mu.pbar(i, len(EE))
    
    E = EE[i]


    for j in range(len(EE)):
        
        w = EE[j]
        
        w_min = 0
        
        y = w/E
        
        if w <= E/2:
            w_min = 0
        elif w <= 3/4*E:
            w_min = E*(2*y - 1)
        else:
            continue
        
        w_max = 2*E*(y - 1 + np.sqrt(1 - y))
        
        inds = np.where(np.logical_and(EE >= w_min, EE <= w_max))[0]
        
        wp = EE[inds]
        
        F = (w*(w - wp))**(-1) + ((E + wp - w)*(E - w))**(-1) +\
            (w*(w - wp)*(E + wp - w)*(E-w))**(-1/2)
        
        tau_exc[i, j] = h2si * 1/(2*np.pi*E) * np.trapz(OLF_1d[inds] * wp * F, x=EE[inds])
    
    
    S[i] = np.trapz(tau_exc[i] * EE * np.heaviside(EE[i]/2 - EE, 1), x=EE)
    u[i] = np.trapz(tau_exc[i] * np.heaviside(EE[i]/2 - EE, 1), x=EE)


#%%
l_Chan = np.loadtxt('curves/Chan_Si_l.txt')

plt.loglog(EE_eV, u / 1e+2, label='My')
#plt.loglog(l_Chan[:, 0], 1 / l_Chan[:, 1], label='Chan')

sigma_MuElec = np.loadtxt('curves/Si_MuElec_sigma.txt')
plt.loglog(sigma_MuElec[:, 0], sigma_MuElec[:, 1] * 1e-18 * mc.n_Si, label='MuElec')

plt.xlim(1, 1e+4)
plt.ylim(1e+4, 1e+8)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Palik_U_Si.png', dpi=300)


#%%
plt.loglog(EE_eV, S / mc.eV / 1e+2, label='my')

S_Chan = np.loadtxt('curves/Chan_Si_S.txt')
#plt.loglog(S_Chan[:, 0], S_Chan[:, 1], label='Chan')

S_MuElec = np.loadtxt('curves/Si_MuElec_S.txt')
plt.loglog(S_MuElec[:, 0], S_MuElec[:, 1] * 1e+7, label='MuElec')

plt.xlim(1, 1e+4)
plt.ylim(1e+6, 1e+9)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Palik_U_Si.png', dpi=300)

