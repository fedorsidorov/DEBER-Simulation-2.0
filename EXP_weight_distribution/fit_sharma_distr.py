#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

mf = importlib.reload(mf)
mv = importlib.reload(mv)
os.chdir(mv.sim_path_MAC + 'L_distribution_simulation')

#%%
def Flory_Schulz(k, N, p):
    return N * np.power(1-p, 2) * k * np.power(p, k-1)

def disproportion(k, N, p):
    return N * np.power(p, k-1) * (1-p)

def Flory_Schulz_mod(k, N, a):
    return N * np.power(a, 2) * k * np.power((1-a), (k-1))

def get_Mn(x, y):
    return np.sum(y * x) / np.sum(y)

def get_Mw(x, y):
    return np.sum(y * np.power(x, 2)) / np.sum(y * x)
    
#%%
peak = np.loadtxt('curves/sharma_peak_B.dat')

start_ind = 20
mma_mass = 100

x_Sharma = peak[start_ind:, 0]
y_Sharma = peak[start_ind:, 1]

x_Mn_Mw = np.ones(len(x_Sharma))
y_Mn_Mw = np.linspace(0, np.max(y_Sharma), len(x_Sharma))

#%%
Mn_Sharma = get_Mn(x_Sharma, y_Sharma)
Mw_Sharma = get_Mw(x_Sharma, y_Sharma)

plt.plot(x_Sharma, y_Sharma, 'ro', label='Sharma')
plt.plot(x_Mn_Mw * Mn_Sharma, y_Mn_Mw, label='Mn_Sharma')
plt.plot(x_Mn_Mw * Mw_Sharma, y_Mn_Mw, label='Mw_Sharma')

#%% Flory-Schulz
popt, pcov = curve_fit(Flory_Schulz, x_Sharma, y_Sharma, p0=[1000, 0.9999])

x_FS = np.arange(0, 10000 * mma_mass)
y_FS = Flory_Schulz(x_FS, *popt)

Mn_FS = get_Mn(x_FS, y_FS)
Mw_FS = get_Mw(x_FS, y_FS)

plt.plot(x_FS, y_FS)
plt.plot(x_Mn_Mw * Mn_FS, y_Mn_Mw, label='Mn_FS')
plt.plot(x_Mn_Mw * Mw_FS, y_Mn_Mw, label='Mw_FS')

#%% Disproportion
popt_d, pcov_d = curve_fit(disproportion, x_Sharma, y_Sharma, p0=[100000, 0.9999])

x_d = np.arange(0, 10000 * mma_mass)
y_d = disproportion(x_d, *popt_d)

Mn_d = get_Mn(x_d, y_d)
Mw_d = get_Mw(x_d, y_d)

plt.plot(x_d, y_d)
plt.plot(x_Mn_Mw * Mn_FS, y_Mn_Mw, label='Mn_d')
plt.plot(x_Mn_Mw * Mw_FS, y_Mn_Mw, label='Mw_d')

#%% Polynom
#pol_coeff = np.polyfit(x_Sharma, y_Sharma, deg=10)
#y_pol = np.polyval(pol_coeff, x_Sharma)

#Mn_pol = get_Mn(x_Sharma, y_pol)
#Mw_pol = get_Mw(x_Sharma, y_pol)

#plt.plot(x_Sharma, y_pol)
#plt.plot(x_Mn_Mw * Mn_pol, y_Mn_Mw, label='Mn_pol')
#plt.plot(x_Mn_Mw * Mw_pol, y_Mn_Mw, label='Mw_pol')

#%%
plt.xlabel('chain mass')
plt.ylabel('arbitrary units')
plt.title('Chain mass distribution')
plt.legend()
plt.grid()
plt.gca().get_xaxis().get_major_formatter().set_powerlimits((0, 0))
plt.show()
#