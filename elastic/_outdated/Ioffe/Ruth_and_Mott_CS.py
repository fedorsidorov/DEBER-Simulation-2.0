#%%
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

import my_utilities as mu
import my_constants as mc

import scipy.signal

mu = importlib.reload(mu)
mc = importlib.reload(mc)

os.chdir(mc.sim_path_MAC + 'elastic')


#%% Rutherford CS
def get_Ruth_diff_CS(Z, E, theta):
    
    alpha = mc.k_el**2 * (mc.m * mc.e**4 * np.pi**2 * Z**(2/3)) / (mc.h**2 * E * mc.eV)
    
    diff_cs = mc.k_el**2 * Z**2 * mc.e**4/ (4 * (E * mc.eV)**2) /\
        np.power(1 - np.cos(theta) + alpha, 2)
    
    return diff_cs


def get_Ruth_CS(Z, E):
    
    alpha = mc.k_el**2 * (mc.m * mc.e**4 * np.pi**2 * Z**(2/3)) / (mc.h**2 * E * mc.eV)
        
    CS = np.pi * mc.k_el**2 * Z**2 * mc.e**4/ ((E * mc.eV)**2) / (alpha * (2 + alpha))
    
    return CS


#%% Mott CS
def get_Ioffe_data(el):
    
    text_file = open('Ioffe_DATA/' + el + '.txt', 'r')
    lines = text_file.readlines()
    
    theta_corr = list(map(lambda s: '0'+s, lines[12].split()[2:]))
    
    theta_arr_deg_final = np.array(list(map(float, theta_corr)))
    
    E_arr_list = []
    diff_cs_list = []
    total_cs_list = []
    
    for line in lines[14:]:
        
        cs_line = line[1:]
        
        now_diff_cs, now_total_cs = cs_line.split(sep='|')
        
        now_e_diff_cs_corr = list(map(lambda s: '0'+s, now_diff_cs.split()))
        now_total_cs_corr = float('0' + now_total_cs.strip())
        
        E_arr_list.append(now_e_diff_cs_corr[0])
        diff_cs_list.append(now_e_diff_cs_corr[1:])
        total_cs_list.append(now_total_cs_corr)
    
    
    diff_cs = np.zeros((len(diff_cs_list), len(diff_cs_list[0])))
    
    for i in range(len(diff_cs_list)):
        
        diff_cs[i, :] = np.array(list(map(float, diff_cs_list[i])))
        
    
    total_cs = np.array(list(map(float, total_cs_list)))
    
    E_arr = np.array(list(map(float, E_arr_list)))
    
    E_arr_final = E_arr[::-1]
    diff_cs_final = diff_cs[::-1]
    total_cs_final = total_cs[::-1]

    return E_arr_final, theta_arr_deg_final, diff_cs_final, total_cs_final


#%% Si
E_arr, theta_arr, diff_cs_Mott, total_cs_Mott = get_Ioffe_data('Si')

total_cs_Ruth = get_Ruth_CS(14, mc.EE)

plt.loglog(E_arr, total_cs_Mott, label='Mott')
plt.loglog(mc.EE, total_cs_Ruth * 1e+4, label='Rutherford')

plt.xlabel('E, eV')
plt.ylabel('$\sigma_{el}$, cm$^2$')

plt.legend()
plt.grid()
plt.show()


#%% Ar, Z = 18, E = 1000 eV
Z = 18
E = 1000
E_pos = 19

E_arr, theta_arr, diff_cs_Mott, total_cs_Mott = get_Ioffe_data(Z)

diff_cs_Ruth = get_Ruth_diff_CS(Z, E, np.deg2rad(theta_arr))

plt.semilogy(theta_arr, diff_cs_Mott[E_pos] * 1e+16, label='Mott cross section')
plt.semilogy(theta_arr, diff_cs_Ruth * 1e+20, label='Rutherford cross section')


Ar_cs_exp = np.loadtxt('Ar_1000eV/Iga.txt')
plt.semilogy(Ar_cs_exp[:, 0], Ar_cs_exp[:, 1], 'ro', label='exp')

#plt.legend()
plt.ylim(1e-3, 1e+2)

plt.xlabel(r'$\theta$, градусов')
plt.ylabel(r'$\frac{d\sigma}{d\Omega}, \AA/ср$')

plt.grid()
plt.show()

#plt.savefig('diff_CS_1.png', dpi=300)

#%% Hg, Z = 80, E = 1000 eV
Z = 80
E = 300
E_pos = 14

E_arr, theta_arr, diff_cs_Mott, total_cs_Mott = get_Ioffe_data(Z)

diff_cs_Ruth = get_Ruth_diff_CS(Z, E, np.deg2rad(theta_arr))

#plt.semilogy(theta_arr, diff_cs_Mott[E_pos] * 1e+16, label='Mott cross section')
#plt.semilogy(theta_arr, diff_cs_Ruth * 1e+20, label='Rutherford cross section')


#Ar_cs_exp = np.loadtxt('Hg_300eV/Bromberg.txt')
#plt.semilogy(Ar_cs_exp[:, 0], Ar_cs_exp[:, 1], 'ro', label='exp')

Ar_cs_exp = np.loadtxt('Hg_300eV/Holtkamp.txt')
plt.semilogy(Ar_cs_exp[:, 0], Ar_cs_exp[:, 1], 'ro', label='exp')

#plt.legend()
plt.ylim(1e-3, 1e+2)

#plt.xlabel(r'$\theta$ (deg)')
#plt.ylabel(r'$\frac{d\sigma}{d\Omega}, ( \frac{\AA}{sr} )$')

plt.xlabel(r'$\theta$, градусов')
plt.ylabel(r'$\frac{d\sigma}{d\Omega}, \AA/ср$')

plt.grid()
plt.show()


theta_arr_interp = np.linspace(1, 180, 100)

diff_cs_Mott_interp = mu.log_interp1d(theta_arr, diff_cs_Mott[E_pos])(theta_arr_interp)

arr_filt = scipy.signal.medfilt(diff_cs_Mott_interp, kernel_size=3)

#plt.semilogy(theta_arr_interp, diff_cs_Mott_interp * 1e+16, label='Mott cross section')
plt.semilogy(theta_arr_interp, arr_filt * 1e+16, label='Mott cross section')

plt.semilogy(theta_arr, diff_cs_Ruth * 1e+20, label='Rutherford cross section')

#plt.savefig('diff_CS_2.png', dpi=300)





