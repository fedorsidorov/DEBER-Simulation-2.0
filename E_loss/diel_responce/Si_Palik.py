#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import my_constants as mc
import my_utilities as mu
import E_loss_functions as elf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
elf = importlib.reload(elf)

os.chdir(mc.sim_path_MAC + 'E_loss/diel_responce')


#%%
Palik_arr = np.loadtxt('curves/Palik_Si_E_n_k.txt')[:-3, :]

E_arr_Palik = Palik_arr[::-1, 0]
n_arr = Palik_arr[::-1, 1]
k_arr = Palik_arr[::-1, 2]

plt.loglog(E_arr_Palik, n_arr, label='n')
plt.loglog(E_arr_Palik, k_arr, label='k')

plt.title('Palik optical measurements for Si')
plt.xlabel('E, eV')
plt.ylabel('value')

plt.legend()
plt.ylim(1e-6, 1e+1)
plt.grid()
plt.show()

#plt.savefig('Si_n_k_Palik.png', dpi=300)


#%%
E_arr = np.logspace(0, 3.3, 5000)

N_arr = mu.log_interp1d(E_arr_Palik, n_arr)(E_arr)
K_arr = mu.log_interp1d(E_arr_Palik, k_arr)(E_arr)

#plt.loglog(E_arr, N_arr, '.')
#plt.loglog(E_arr, K_arr, '.')


#%%
Im_arr = 2 * N_arr * K_arr / ( (N_arr**2 - K_arr**2)**2 + (2*N_arr*K_arr)**2 )


#%% Add points to Im
x1 = E_arr[-2]
y1 = Im_arr[-2]

x2 = E_arr[-1]
y2 = Im_arr[-1]

EE = mc.EE

x3 = EE[-1]
y3 = y2 * np.exp( np.log(y2/y1) * np.log(x3/x2) / np.log(x2/x1) )

E_arr_new = np.append(E_arr, [x3])
Im_arr_new = np.append(Im_arr, [y3])

IM = mu.log_interp1d(E_arr_new, Im_arr_new)(EE)


#%%
plt.loglog(EE, IM, 'ro', label='Extended')
plt.loglog(E_arr, Im_arr, label='Original')

plt.title('Si optical E-loss function from Palik data')
plt.xlabel('E, eV')
plt.ylabel('Im[1/eps]')

plt.legend()
plt.grid()
plt.show()
#plt.savefig('Si_OELF_Palik.png', dpi=300)


#%%
Si_U = np.zeros(len(EE))
Si_SP = np.zeros(len(EE))

Si_DIFF_U = np.zeros((len(EE), len(EE)))

for i in range(len(EE)):
    
    now_E = EE[i]
    
    inds = np.where(EE <= now_E/2)[0]
    
    y_U = IM[inds] * elf.Ashley_L(EE[inds]/now_E)
    y_SP = IM[inds] * elf.Ashley_S(EE[inds]/now_E) * EE[inds]*mc.eV
    
    Si_U[i] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 * now_E*mc.eV) *\
        np.trapz(y_U, x=EE[inds]*mc.eV) / 1e+2 ## cm^-1
        
    Si_SP[i] = mc.k_el * mc.m * mc.e**2 / (np.pi * mc.hbar**2 * now_E*mc.eV) *\
        np.trapz(y_SP, x=EE[inds]*mc.eV) / mc.eV / 1e+2 ## eV / cm
    
    Si_DIFF_U[i, inds] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 *\
        now_E*mc.eV) * Im_arr[inds] * elf.Ashley_L(EE[inds]/now_E) / 1e+2 * mc.eV ## cm^-1 / eV


#%% Ashley 1990
U_DIFF_A = np.zeros((len(EE), len(EE)))

for i in range(len(EE)):
    
    print(i)
    
    now_E = EE[i]

    for j in range(len(EE)):
        
        now_w = EE[j]
        
        w_min = 0
        
        if now_w <= now_E/2:
            w_min = 0
        elif now_w <= 3*now_E/4:
            w_min = 2*now_w - now_E
        else:
            continue
        
        w_max = 2*np.sqrt(now_E - now_w)*(np.sqrt(now_E) - np.sqrt(now_E - now_w))
        
        inds = np.where(np.logical_and(EE >= w_min, EE <= w_max))[0]
        
        X = EE[inds]*mc.eV
        
        E = now_E * mc.eV
        w = now_w * mc.eV
        wp = EE[inds] * mc.eV
        
        ## From Chan thesis - W/O exchange correction
#        F = (now_w*mc.eV * (now_w - E_arr[inds])*mc.eV)**(-1)
        
        ## From ashley1990.pdf
        F = (w*(w - wp))**(-1) +\
            ((E + wp - w)*(E - w))**(-1) +\
            (w*(w - wp)*(E + wp - w)*(E-w))**(-1/2)
        
        Y = IM[inds] * EE[inds]*mc.eV * F
        
        U_DIFF_A[i, j] = mc.k_el * mc.m * mc.e**2 /\
            (2 * np.pi * mc.hbar**2 * now_E*mc.eV) * np.trapz(Y, x=X) / 1e+2 * mc.eV


#%%
ind = 681 ## E = 1 keV
#ind = 817 ## E = 4 keV
plt.loglog(EE, Si_DIFF_U[ind, :])


#%%
CHAN = np.loadtxt('diel_responce/curves/Chan_Si_IMFP.txt')

plt.loglog(CHAN[:, 0], 1/CHAN[:, 1], label='Chan')
plt.loglog(EE, Si_U, label='My')

plt.title('$\mu$ for Si')
plt.xlabel('E, eV')
plt.ylabel('$\mu$, cm$^{-1}$')

plt.xlim(1, 1e+4)
plt.ylim(1e+4, 1e+8)

plt.legend()
plt.grid()
plt.show()
#plt.savefig('Palik_U_Si.png', dpi=300)


#%%
Si_U_Gryzinski_SS = np.load('Gryzinski/Si_TOTAL_U_SS.npy')
Si_U_Gryzinski_VAL = np.load('Gryzinski/Si_TOTAL_U_VAL.npy')

plt.loglog(mc.EE, Si_U_Gryzinski_SS, label='Gryzinski SS')
plt.loglog(mc.EE, Si_U_Gryzinski_VAL, label='Gryzinski mean E$_{val}$')

plt.loglog(EE, Si_U, label='Extended Palik')

plt.title('$\mu$ for Si')
plt.xlabel('E, eV')
plt.ylabel('$\mu$, cm$^{-1}$')

plt.xlim(1, 1e+4)
plt.ylim(1e+4, 1e+9)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Gryz_Palik_U_Si.png', dpi=300)


#%%
Si_INT_U = elf.diff2int(Si_DIFF_U)

np.save('diel_responce/Palik/Si_U_Palik.npy', Si_U)
np.save('diel_responce/Palik/Si_SP_Palik.npy', Si_SP)
np.save('diel_responce/Palik/Si_diff_U_Palik.npy', Si_DIFF_U)
np.save('diel_responce/Palik/Si_int_U_Palik.npy', Si_INT_U)


#%%
U_INT_A = np.zeros((len(EE), len(EE)))

for i in range(len(EE)):
    
    integral = np.trapz(U_DIFF_A[i, :], x=EE)
    
    if integral == 0:
        continue
    
    for j in range(1, len(EE)):
        
        U_INT_A[i, j] = np.trapz(U_DIFF_A[i, :j], x=EE[:j]) / integral

