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

os.chdir(mv.sim_path_MAC + 'E_loss')

#%%
EE = mv.EE

SP_TOT = np.load('Palik/Si_SP_TOTAL.npy')
SP_CORE = np.load('Gryzinski/Si_core_SP.npy')

plt.loglog(EE, SP_TOT, label='core Gryzinski')
plt.loglog(EE, SP_CORE, label='total Palik')

plt.title('Si stopping power')
plt.xlabel('E, eV')
plt.ylabel('SP, eV/cm')

plt.xlim(1, 1e+5)
plt.ylim(1e+3, 1e+9)

plt.legend()
plt.grid()
plt.show()

#plt.savefig('Si_SP_core_total.png', dpi=300)


#%%
SP_VAL = SP_TOT - SP_CORE
E_BIND = np.zeros(len(EE))

EB = np.logspace(-1, 1.5, 1000)

#for i in range(48, len(EE)):
for i in range(0, len(EE)):
    
    mf.upd_progress_bar(i, len(EE))
    
    for e in EB[::-1]:
        
        now_SP = elf.get_Gryzinski_SP(EE[i:i+1], e, mc.n_Si, elf.n_val_Si)
        
        if now_SP > SP_VAL[i]:
            E_BIND[i] = e
#            print('success')
            break


#%%
plt.semilogx(EE, E_BIND, label='My')

plt.title('U$_{bind}$(E)')
plt.xlabel('e, eV')
plt.ylabel('U$_{bind}$(E)')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('My U_bind(E).png', dpi=300)


#%%
SP_VAL_TEST = np.zeros(len(EE))

for i in range(len(EE)):
    
    SP_VAL_TEST[i] = elf.get_Gryzinski_SP(EE[i:i+1], E_BIND[i], mc.n_Si, elf.n_val_Si)


plt.loglog(EE, SP_VAL, label='Difference')
plt.loglog(EE, SP_VAL_TEST, '--', label='My')

plt.title('SP$_{valence}(E)$')
plt.xlabel('e, eV')
plt.ylabel('SP$_{valence}$')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('SP_valence.png', dpi=300)

#%%
U_TOT = np.load('../diel_responce/Dapor/U_PMMA_DAPOR.npy')
#U_CORE = np.load('U_PMMA_CORE.npy')
U_CORE = elf.get_PMMA_Gryzinski_core_U(EE)

E_BIND = np.load('E_BIND_PMMA.npy')
CC = np.load('C_PMMA.npy')

U_VAL = U_TOT - U_CORE
U_VAL_TEST = np.zeros(len(EE))

for i in range(len(EE)):
    
    U_VAL_TEST[i] = elf.get_Gryzinsky_CS([EE[i]], E_BIND[i]) *\
        CC[i] * mc.n_PMMA * elf.n_val_PMMA

plt.loglog(EE, U_VAL, label='Difference')
plt.loglog(EE, U_VAL_TEST, '--', label='My')

plt.title('$\mu_{valence}(E)$')
plt.xlabel('e, eV')
plt.ylabel('$\mu_{valence}$, cm$^{-1}$')

plt.legend()
plt.grid()
plt.show()

plt.savefig('U_valence.png', dpi=300)


