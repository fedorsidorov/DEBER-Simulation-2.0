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

os.chdir(mv.sim_path_MAC + 'E_loss/E_bind_PMMA')

#%%
EE = mv.EE

SP_TOT = np.load('../diel_responce/Dapor/PMMA_SP_Dapor.npy')
SP_CORE = np.load('../Gryzinski/PMMA_core_SP.npy')

plt.loglog(EE, SP_TOT, label='core')
plt.loglog(EE, SP_CORE, label='total')

plt.title('PMMA stopping power')
plt.xlabel('E, eV')
plt.ylabel('SP, eV/cm')

plt.xlim(1, 1e+5)
plt.ylim(1e+3, 1e+9)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('PMMA_SP_core_Dapor.png', dpi=300)


#%%
SP_VAL = SP_TOT - SP_CORE

plt.loglog(EE, SP_VAL)

plt.title('PMMA valence stopping power')
plt.xlabel('E, eV')
plt.ylabel('SP, eV/cm')

plt.ylim(1e+3, 1e+9)

#plt.legend()
plt.grid()
plt.show()

#plt.savefig('PMMA_val_SP.png', dpi=300)


#%%
E_BIND = np.zeros(len(EE))

#EB = np.logspace(0, 1.5, 5000)
EB = np.linspace(1, 25, 5000)

CC = np.zeros(len(EE))

for i in range(len(EE)):
    
    mf.upd_progress_bar(i, len(EE))
    
    if SP_VAL[i] == 0:
        continue
    
    
    now_E = EE[i]
    
    pre_eb = 0
    pre_SP = 0
    
    for eb in EB:
        
        now_SP = elf.get_Gryzinski_SP_single(now_E, eb, mc.n_PMMA_mon, elf.n_val_PMMA, mv.WW_ext)
        
        if now_SP < SP_VAL[i] or now_SP == 0:
            E_BIND[i] = pre_eb
            break
        
        pre_eb = eb
        pre_SP = now_SP
        
    pre_SP = 0
        

#%%
plt.semilogx(EE, E_BIND)

plt.title('U$_{bind}$(E)')
plt.xlabel('e, eV')
plt.ylabel('U$_{bind}$(E)')

#plt.legend()
plt.grid()
plt.show()

plt.savefig('My U_bind_VAL_PMMA.png', dpi=300)


#%%
SP_VAL_TEST = np.zeros(len(EE))

for i in range(70, len(EE)):
    
    SP_VAL_TEST[i] = elf.get_Gryzinski_SP_single(EE[i], E_BIND[i],\
               mc.n_PMMA_mon, elf.n_val_PMMA)


plt.loglog(EE, SP_VAL, label='Total - Core')
plt.loglog(EE, SP_VAL_TEST, '--', label='E$_{bind}$')

plt.title('SP$_{valence}(E)$')
plt.xlabel('E, eV')
plt.ylabel('SP$_{valence}$')

plt.legend()
plt.grid()
plt.show()

plt.savefig('SP_VAL_PMMA.png', dpi=300)


#%%
U_TOT = np.load('../diel_responce/Dapor/PMMA_U_Dapor.npy')
U_CORE = elf.get_PMMA_Gryzinski_core_U(EE)

E_BIND = np.load('PMMA_E_bind.npy')

U_VAL = U_TOT - U_CORE
U_VAL_TEST = np.zeros(len(EE))


for i in range(len(EE)):
    
    if U_VAL[i] == 0:
        continue
    
    U_VAL_TEST[i] = elf.get_Gryzinski_CS(EE[i:i+1], E_BIND[i]) *\
        mc.n_PMMA_mon * elf.n_val_PMMA

plt.loglog(EE, U_VAL, label='Difference')
plt.loglog(EE, U_VAL_TEST, '--', label='My')

plt.title('$\mu_{valence}(E)$')
plt.xlabel('e, eV')
plt.ylabel('$\mu_{valence}$, cm$^{-1}$')

plt.legend()
plt.grid()
plt.show()

plt.savefig('U_valence_PMMA.png', dpi=300)

