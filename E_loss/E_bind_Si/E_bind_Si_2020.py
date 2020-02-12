#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import matplotlib.pyplot as plt
import E_loss_functions_2020 as elf
import my_utilities as mu

mu = importlib.reload(mu)
mc = importlib.reload(mc)
elf = importlib.reload(elf)

os.chdir(os.path.join(mc.sim_folder, 'E_loss', 'E_bind_Si'))


#%%
EE = mc.EE

SP_TOT = np.load(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Palik', 'Si_SP_Palik.npy'
        ))

#SP_CORE = np.load(os.path.join(mc.sim_folder,
#        'E_loss', 'Gryzinski', 'Si', 'Si_core_SP.npy'
#        ))

SP_CORE_2020 = elf.get_Si_Gryzinski_core_SP(EE)

plt.loglog(EE, SP_TOT, label='total')
#plt.loglog(EE, SP_CORE, label='core')
plt.loglog(EE, SP_CORE_2020, label='core_2020')

plt.title('Si stopping power')
plt.xlabel('E, eV')
plt.ylabel('SP, eV/cm')

plt.xlim(1, 1e+5)
plt.ylim(1e+3, 1e+9)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('PMMA_SP_core_total.png', dpi=300)


#%%
SP_VAL = SP_TOT - SP_CORE

plt.loglog(EE, SP_VAL)

plt.title('Si valence stopping power')
plt.xlabel('E, eV')
plt.ylabel('SP, eV/cm')

plt.ylim(1e+3, 1e+9)

#plt.legend()
plt.grid()
plt.show()

#plt.savefig('Si_val_SP.png', dpi=300)


#%%
E_BIND = np.zeros(len(EE))

EB = np.logspace(-1, 1.4, 10000)
#EB = np.linspace(0.01, 25, 5000)


for i in range(len(EE)):
    
    mu.pbar(i, len(EE))
    
    now_E = EE[i]
    now_SP = elf.get_Gryzinski_SP_single(now_E, EB[0], mc.n_Si, mc.n_val_Si, mc.WW_ext)
    
    if now_SP < SP_VAL[i]:
        print('Not enough SP,', EE[i])
        E_BIND[i] = EB[0]
        continue
    
    pre_eb = 0
    pre_SP = 0
    
    for eb in EB[1:]:
        
        now_SP = elf.get_Gryzinski_SP_single(now_E, eb, mc.n_Si, mc.n_val_Si, mc.WW_ext)
        
        if now_SP < SP_VAL[i] or now_SP == 0:
            E_BIND[i] = pre_eb
            
            break
        
        pre_eb = eb
        pre_SP = now_SP
        
    pre_SP = 0
        

#%%
plt.semilogx(EE, E_BIND, '--', label='adjusted E$_bind$')

plt.title('E$_{bind}$(E) for Si')
plt.xlabel('e, eV')
plt.ylabel('E$_{bind}$(E)')

plt.xlim(1, 1e+4)
plt.ylim(0, 25)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('E_bind_val_Si_2020.png', dpi=300)


#%%
SP_VAL_TEST = np.zeros(len(EE))

for i in range(70, len(EE)):
    
    SP_VAL_TEST[i] = elf.get_Gryzinski_SP_single(EE[i], E_BIND[i],\
               mc.n_Si, mc.n_val_Si, mc.WW)


plt.loglog(EE, SP_VAL, label='Total - Core')
plt.loglog(EE, SP_VAL_TEST, '--', label='E$_{bind}$')

plt.title('Si SP$_{valence}(E)$')
plt.xlabel('E, eV')
plt.ylabel('SP$_{valence}$')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Si_SP_val_2020.png', dpi=300)


#%%
U_TOT = np.load('../diel_responce/Palik/Si_U_Palik.npy')
U_CORE = elf.get_Si_Gryzinski_core_U(EE)

E_BIND = np.load('Si_E_bind.npy')

U_VAL = U_TOT - U_CORE
U_VAL_TEST = np.zeros(len(EE))


for i in range(len(EE)):
    
    if U_VAL[i] == 0:
        continue
    
    U_VAL_TEST[i] = elf.get_Gryzinski_CS(EE[i:i+1], E_BIND[i], mc.WW) *\
        mc.n_Si * mc.n_val_Si

plt.loglog(EE, U_VAL, label='Difference')
plt.loglog(EE, U_VAL_TEST, '--', label='My')

plt.title('$\mu_{valence}(E)$')
plt.xlabel('e, eV')
plt.ylabel('$\mu_{valence}$, cm$^{-1}$')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('U_valence_Si_2020.png', dpi=300)


