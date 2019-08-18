#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import my_constants as mc
import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mc = importlib.reload(mc)

os.chdir(mv.sim_path_MAC + 'Ionization')

#%% Binding energies and occupancies
##              1s 2s 2p 3s 3p
binding_H    = [13.6]
occupancy_H  = [1]
binding_C    = [296, 16.59, 11.26]
occupancy_C  = [2, 2, 2]
binding_O    = [538, 28.48, 13.62]
occupancy_O  = [2, 2, 4]
binding_Si   = [1844, 154, 104, 13.46, 8.15]
occupancy_Si = [2, 2, 6, 2, 2]

#%%
def get_Vriens_ds_ddE(E, Ui, W):
    
    R = 13.6 ## eV
    Rn = R / E
    Uin = Ui / E
    
    inds = np.where(np.logical_and(W>=Ui, W <= (E+Ui)/2))
    eps = W[inds] / E
    
    PHI = np.cos( -np.sqrt((Rn/(1 + Uin))) * np.log(Uin) )
    
    diff_cs = np.pi * mc.k_el**2 * mc.e**4 / ((E*mc.eV)**3*(1 + 2*Uin)) *\
        ( (1/eps**2 + 4*Uin/(3*eps**3)) + (1/(1 + Uin - eps)**2 +\
           4*Uin/(3*(1 + Uin - eps)**2)) - PHI / (eps*(1 + Uin - eps)))
    
    vriens_ds_ddE = np.zeros(len(W))
    vriens_ds_ddE[inds] = diff_cs ## m^2 / J
    
    return vriens_ds_ddE * (100)**2 * mc.eV  ## cm^2 / eV


def get_Gryzinski_ds_ddE(E, Ui, W):

    inds = np.where(np.logical_and(W>=Ui, W <= (E+Ui)/2))
    dE = W[inds]
    
    diff_cs = np.pi * mc.k_el**2 * mc.e**4 / (dE*mc.eV)**3 * Ui/E *\
        np.power(E / (E + Ui), 3/2) * np.power((1 - dE/E), Ui/(Ui+dE)) *\
        (dE/Ui * (1 - Ui/E) + 4/3 * np.log(2.7 + np.sqrt((E - dE)/Ui)))
    
    gryz_ds_ddE = np.zeros(len(W))
    gryz_ds_ddE[inds] = diff_cs ## m^2 / J
    
    return gryz_ds_ddE * (100)**2 * mc.eV ## cm^2 / eV


def get_total_Gryzinsky_cs(E_arr, Ui):
    
    total_Gryzinsky_cs = np.zeros(len(E_arr))
    
    dE_arr = np.logspace(0, 4.4, 1000)
    
    for i in range(len(E_arr)):
        
        diff_cs = get_Gryzinski_ds_ddE(E_arr[i], Ui, dE_arr)
        total_Gryzinsky_cs[i] = np.trapz(diff_cs, x=dE_arr)
        
    return total_Gryzinsky_cs


def get_total_Vriens_cs_int(E_arr, Ui):
    
    total_Vriens_cs = np.zeros(len(E_arr))
    
    dE_arr = np.logspace(0, 4.4, 1000)
    
    for i in range(len(E_arr)):
        
        diff_cs = get_Vriens_ds_ddE(E_arr[i], Ui, dE_arr)
        total_Vriens_cs[i] = np.trapz(diff_cs, x=dE_arr)
        
    return total_Vriens_cs


def get_total_Vriens_cs(E_arr, Ui):
    
    R = 13.6 ## eV
    Rn = R / E
    Uin = Ui / E
    
    PHI = np.cos( -np.sqrt((Rn/(1 + Uin))) * np.log(Uin) )
    
    total_Vriens_cs = np.pi * mc.k_el**2 * mc.e**4 / ((E_arr*mc.eV)**2 * (1 + 2*Uin)) *\
        (5/(3*Uin) - 1 - 2/3*Uin + PHI*np.log(Uin)/(1 + Uin))
    
    return total_Vriens_cs * (100)**2 ## cm^2


#%% compare diff CS
#E = np.logspace(0, 4.4, 1000)
E = 400
DE = np.logspace(0, 4.4, 1000)

diff_V = get_Vriens_ds_ddE(E, binding_H[0], DE)
diff_G = get_Gryzinski_ds_ddE(E, binding_H[0], DE)

plt.loglog(DE, diff_V, label='Vriens')
plt.loglog(DE, diff_G, label='Gryzinski')

plt.title('$d\sigma_{ion}$/$d$$\Delta$E for H')
plt.xlabel('E, eV')
plt.ylabel('$d\sigma_{ion}$/$d$$\Delta$E, cm$^{2}$/eV')
plt.legend()
plt.grid()
plt.show()

#%% compare total CS for H
E = np.logspace(0, 4.4, 1000)

CS_H_1S_G = get_total_Gryzinsky_cs(E, binding_H[0])*occupancy_H[0]
CS_H_1S_V = get_total_Vriens_cs(E, binding_H[0])*occupancy_H[0]

CS_H_V = CS_H_1S_V
CS_H_G = CS_H_1S_G

plt.loglog(E, CS_H_1S_G, label='Gryzinski')
plt.loglog(E, CS_H_1S_V, label='Vriens')

plt.title('$\sigma_{ion}(E)$ for H')
plt.xlabel('E, eV')
plt.ylabel('$\sigma_{ion}(E)$ for H, cm$^{2}$')
plt.legend()
plt.grid()
plt.show()

#%% compare total CS for C
E = np.logspace(0, 4.4, 1000)

CS_C_1S_G = get_total_Gryzinsky_cs(E, binding_C[0])*occupancy_C[0]
CS_C_1S_V = get_total_Vriens_cs(E, binding_C[0])*occupancy_C[0]

CS_C_2S_G = get_total_Gryzinsky_cs(E, binding_C[1])*occupancy_C[1]
CS_C_2S_V = get_total_Vriens_cs(E, binding_C[1])*occupancy_C[1]

CS_C_2P_G = get_total_Gryzinsky_cs(E, binding_C[2])*occupancy_C[2]
CS_C_2P_V = get_total_Vriens_cs(E, binding_C[2])*occupancy_C[2]

plt.loglog(E, CS_C_1S_G + CS_C_2S_G + CS_C_2P_G, label='Gryzinski')
plt.loglog(E, CS_C_1S_V + CS_C_2S_V + CS_C_2P_V, label='Vriens')

plt.title('$\sigma_{ion}(E)$ for C')
plt.xlabel('E, eV')
plt.ylabel('$\sigma_{ion}(E)$ for C, cm$^{2}$')
plt.legend()
plt.grid()
plt.show()

plt.savefig('Gryzinski VS Vriens for C.png', dpi=300)

#%% compare total CS for O
E = np.logspace(0, 4.4, 1000)

CS_O_1S_G = get_total_Gryzinsky_cs(E, binding_O[0])*occupancy_O[0]
CS_O_1S_V = get_total_Vriens_cs(E, binding_O[0])*occupancy_O[0]

CS_O_2S_G = get_total_Gryzinsky_cs(E, binding_O[1])*occupancy_O[1]
CS_O_2S_V = get_total_Vriens_cs(E, binding_O[1])*occupancy_O[1]

CS_O_2P_G = get_total_Gryzinsky_cs(E, binding_O[2])*occupancy_O[2]
CS_O_2P_V = get_total_Vriens_cs(E, binding_O[2])*occupancy_O[2]

plt.loglog(E, CS_O_1S_G + CS_O_2S_G + CS_O_2P_G, label='Gryzinski')
plt.loglog(E, CS_O_1S_V + CS_O_2S_V + CS_O_2P_V, label='Vriens')

plt.title('$\sigma_{ion}(E)$ for O')
plt.xlabel('E, eV')
plt.ylabel('$\sigma_{ion}(E)$ for O, cm$^{2}$')
plt.legend()
plt.grid()
plt.show()

plt.savefig('Gryzinski VS Vriens for O.png', dpi=300)

#%% compare total CS for Si
E = np.logspace(0, 4.4, 1000)

CS_Si_1S_G = get_total_Gryzinsky_cs(E, binding_Si[0])*occupancy_Si[0]
CS_Si_1S_V = get_total_Vriens_cs(E, binding_Si[0])*occupancy_Si[0]

#plt.loglog(E, CS_Si_1S_G)
#plt.loglog(E, CS_Si_1S_V)

CS_Si_2S_G = get_total_Gryzinsky_cs(E, binding_Si[1])*occupancy_Si[1]
CS_Si_2S_V = get_total_Vriens_cs(E, binding_Si[1])*occupancy_Si[1]

#plt.loglog(E, CS_Si_2S_G)
#plt.loglog(E, CS_Si_2S_V)

CS_Si_2P_G = get_total_Gryzinsky_cs(E, binding_Si[2])*occupancy_Si[2]
CS_Si_2P_V = get_total_Vriens_cs(E, binding_Si[2])*occupancy_Si[2]

#plt.loglog(E, CS_Si_2P_G)
#plt.loglog(E, CS_Si_2P_V)

CS_Si_3S_G = get_total_Gryzinsky_cs(E, binding_Si[3])*occupancy_Si[3]
CS_Si_3S_V = get_total_Vriens_cs(E, binding_Si[3])*occupancy_Si[3]

#plt.loglog(E, CS_Si_3S_G)
#plt.loglog(E, CS_Si_3S_V)

CS_Si_3P_G = get_total_Gryzinsky_cs(E, binding_Si[4])*occupancy_Si[4]
CS_Si_3P_V = get_total_Vriens_cs(E, binding_Si[4])*occupancy_Si[4]

#plt.loglog(E, CS_Si_3P_G)
#plt.loglog(E, CS_Si_3P_V)

plt.loglog(E, CS_Si_1S_G + CS_Si_2S_G + CS_Si_2P_G + CS_Si_3S_G + CS_Si_3P_G, label='Gryzinski')
plt.loglog(E, CS_Si_1S_V + CS_Si_2S_V + CS_Si_2P_V + CS_Si_3S_V + CS_Si_3P_V, label='Vriens')

plt.title('$\sigma_{ion}(E)$ for Si')
plt.xlabel('E, eV')
plt.ylabel('$\sigma_{ion}(E)$ for Si, cm$^{2}$')
plt.legend()
plt.grid()
plt.show()

plt.savefig('Gryzinski VS Vriens for Si.png', dpi=300)
