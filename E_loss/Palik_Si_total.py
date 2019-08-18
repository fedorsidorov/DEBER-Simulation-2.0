#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import my_functions as mf
import my_variables as mv
import my_constants as mc
import E_loss_functions as elf

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mc = importlib.reload(mc)
elf = importlib.reload(elf)

os.chdir(mv.sim_path_MAC + 'E_loss')


#%%
Palik_arr = np.loadtxt('diel_responce/curves/Palik_Si_E_n_k.txt')[:-3, :]

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


#%%
E_arr = np.logspace(0, 3.3, 5000)

N_arr = mf.log_interp1d(E_arr_Palik, n_arr)(E_arr)
K_arr = mf.log_interp1d(E_arr_Palik, k_arr)(E_arr)

plt.loglog(E_arr, N_arr, '.')
plt.loglog(E_arr, K_arr, '.')


#%%
Im_arr = 2 * N_arr * K_arr / ( (N_arr**2 - K_arr**2)**2 + (2*N_arr*K_arr)**2 )


#%% Add points to Im
x1 = E_arr[-2]
y1 = Im_arr[-2]

x2 = E_arr[-1]
y2 = Im_arr[-1]

EE = mv.EE

x3 = EE[-1]
y3 = y2 * np.exp( np.log(y2/y1) * np.log(x3/x2) / np.log(x2/x1) )

E_arr_new = np.append(E_arr, [x3])
Im_arr_new = np.append(Im_arr, [y3])

IM = mf.log_interp1d(E_arr_new, Im_arr_new)(EE)


#%%
plt.loglog(EE, IM, 'ro', label='Extended')
plt.loglog(E_arr, Im_arr, label='Original')

plt.title('Palik OELF')
plt.xlabel('E, eV')
plt.ylabel('Im[1/eps]')

plt.legend()
plt.grid()
plt.show()
#plt.savefig('Palik_OELF.png', dpi=300)


#%%
#Si_U = np.zeros(len(E_arr))
#Si_SP = np.zeros(len(E_arr))
#
#Si_DIFF_U = np.zeros((len(E_arr), len(E_arr)))
#
#for i in range(len(E_arr)):
#    
#    now_E = E_arr[i]
#    
#    inds = np.where(E_arr <= now_E/2)[0]
#    
#    y_U = Im_arr[inds] * elf.Ashley_L(E_arr[inds]/E_arr[i])
#    y_SP = Im_arr[inds] * elf.Ashley_S(E_arr[inds]/E_arr[i]) * E_arr[inds]*mc.eV
#    
#    Si_U[i] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 * E_arr[i]*mc.eV) *\
#        np.trapz(y_U, x=E_arr[inds]*mc.eV) / 1e+2 ## cm^-1
#        
#    Si_SP[i] = mc.k_el * mc.m * mc.e**2 / (np.pi * mc.hbar**2 * E_arr[i]*mc.eV) *\
#        np.trapz(y_SP, x=E_arr[inds]*mc.eV) / mc.eV / 1e+2 ## eV / cm
#    
#    Si_DIFF_U[i, inds] = mc.k_el * mc.m * mc.e**2 / (2 * np.pi * mc.hbar**2 *\
#        E_arr[i]*mc.eV) * Im_arr[inds] * elf.Ashley_L(E_arr[inds]/E_arr[i]) /\
#        1e+2 * mc.eV ## cm^-1 / eV


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
        now_E*mc.eV) * Im_arr[inds] * elf.Ashley_L(EE[inds]/now_E) /\
        1e+2 * mc.eV ## cm^-1 / eV


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

plt.loglog(mv.EE, Si_U_Gryzinski_SS, label='Gryzinski SS')
plt.loglog(mv.EE, Si_U_Gryzinski_VAL, label='Gryzinski mean E$_{val}$')

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
Si_INT_U = elf.diff2int(Si_DIFF_U, EE=mv.EE)

np.save('Palik/Si_U_TOTAL.npy', Si_U)
np.save('Palik/Si_SP_TOTAL.npy', Si_SP)
np.save('Palik/Si_DIFF_U_TOTAL.npy', Si_DIFF_U)
np.save('Palik/Si_INT_U_TOTAL.npy', Si_INT_U)

