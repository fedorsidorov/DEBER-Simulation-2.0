#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import my_constants as mc
import matplotlib.pyplot as plt
import E_loss_functions as elf

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mc = importlib.reload(mc)
elf = importlib.reload(elf)

os.chdir(mv.sim_path_MAC + 'E_loss/E_bind_Si')

#%%
EE = mv.EE

SP_TOT = np.load('../diel_responce/Palik/Si_SP_Palik.npy')
SP_CORE = np.load('../Gryzinski/Si_core_SP.npy')

plt.loglog(EE, SP_TOT, label='core')
plt.loglog(EE, SP_CORE, label='total')

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
now_E = EE[100]

EB = np.logspace(0, 1.5, 2000)

now_SP = np.zeros(len(EB))


for i in range(len(now_SP)):
    
    now_SP[i] = elf.get_Gryzinski_SP_single(now_E, EB[i], mc.n_Si, elf.n_val_Si, mv.WW_ext)


#%%
SP_1 = elf.get_Gryzinski_SP(EE, mv.WW_ext[0], mc.n_Si, elf.n_val_Si, mv.WW_ext)

plt.loglog(EE, SP_VAL, label='valence')
plt.loglog(EE, SP_1, label='valence at lowest Eb')


#%%
plt.loglog(EB, now_SP)


#%%
E_BIND = np.zeros(len(EE))

EB = np.logspace(0, 1.5, 1000)


for i in range(len(EE)):
    
    mf.upd_progress_bar(i, len(EE))
    
    now_E = EE[i]
    
    now_SP = elf.get_Gryzinski_SP_single(now_E, EB[0], mc.n_Si, elf.n_val_Si, mv.WW_ext)
    
    
    if now_SP < SP_VAL[i]:
        print('Not enough SP,', EE[i])
        E_BIND[i] = EB[0]
        continue
    
    
    pre_eb = 0
    pre_SP = 0
    
    for eb in EB[1:]:
        
        now_SP = elf.get_Gryzinski_SP_single(now_E, eb, mc.n_Si, elf.n_val_Si, mv.WW_ext)
        
        if now_SP < SP_VAL[i] or now_SP == 0:
            E_BIND[i] = pre_eb
            
            break
        
        pre_eb = eb
        pre_SP = now_SP
        
    pre_SP = 0
        

#%%
plt.semilogx(EE, E_BIND, label='My')

plt.title('E$_{bind}$(E) for Si')
plt.xlabel('e, eV')
plt.ylabel('E$_{bind}$(E)')

plt.legend()
plt.grid()
plt.show()

plt.savefig('E_bind_val_Si.png', dpi=300)


#%%
SP_VAL_TEST = np.zeros(len(EE))

for i in range(70, len(EE)):
    
    SP_VAL_TEST[i] = elf.get_Gryzinski_SP_single(EE[i], E_BIND[i],\
               mc.n_Si, elf.n_val_Si, mv.WW)


plt.loglog(EE, SP_VAL, label='Total - Core')
plt.loglog(EE, SP_VAL_TEST, '--', label='E$_{bind}$')

plt.title('SP$_{valence}(E)$')
plt.xlabel('E, eV')
plt.ylabel('SP$_{valence}$')

plt.legend()
plt.grid()
plt.show()

plt.savefig('Si_SP_val.png', dpi=300)


#%%
U_TOT = np.load('../diel_responce/Palik/Si_U_Palik.npy')
U_CORE = elf.get_Si_Gryzinski_core_U(EE)

E_BIND = np.load('Si_E_bind.npy')

U_VAL = U_TOT - U_CORE
U_VAL_TEST = np.zeros(len(EE))


for i in range(len(EE)):
    
    if U_VAL[i] == 0:
        continue
    
    U_VAL_TEST[i] = elf.get_Gryzinski_CS(EE[i:i+1], E_BIND[i], mv.WW) *\
        mc.n_Si * elf.n_val_Si

plt.loglog(EE, U_VAL, label='Difference')
plt.loglog(EE, U_VAL_TEST, '--', label='My')

plt.title('$\mu_{valence}(E)$')
plt.xlabel('e, eV')
plt.ylabel('$\mu_{valence}$, cm$^{-1}$')

plt.legend()
plt.grid()
plt.show()

plt.savefig('U_valence_Si.png', dpi=300)


