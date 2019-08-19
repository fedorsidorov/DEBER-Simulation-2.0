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

SP_TOT = np.load('../diel_responce/Dapor/SP_PMMA_DAPOR.npy')
SP_CORE = np.load('SP_PMMA_CORE.npy')

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

#plt.savefig('PMMA_SP_core_total.png', dpi=300)


#%%
#CA = np.loadtxt('curves/C.txt')
#C = mf.log_interp1d(CA[:, 0], CA[:, 1])(EE)
#
#plt.semilogx(EE, C, label='Aktary')
#
#plt.title('C(E)')
#plt.xlabel('e, eV')
#plt.ylabel('C(E)')
#
#plt.legend()
#plt.grid()
#plt.show()

#plt.savefig('C(E).png', dpi=300)

#%%
#UA = np.loadtxt('curves/U_bind.txt')
#U = mf.log_interp1d(UA[:, 0], UA[:, 1])(EE)
#
#plt.semilogx(EE, U, label='Aktary')
#
#plt.title('U$_{bind}$(E)')
#plt.xlabel('e, eV')
#plt.ylabel('U$_{bind}$(E)')
#
#plt.legend()
#plt.grid()
#plt.show()

#plt.savefig('U_bind(E).png', dpi=300)

#%%
SP_VAL = SP_TOT - SP_CORE
E_BIND = np.zeros(len(EE))
#CC = np.zeros(len(EE))

EB = np.logspace(-3, 1.5, 1000)

for i in range(48, len(EE)):
#for i in range(len(EE)):
    
    mf.upd_progress_bar(i, len(EE))
    
    for e in EB[::-1]:
        
        now_SP = elf.get_Gryzinski_SP(np.array(EE[i:i+1]), e, mc.n_PMMA_mon, elf.n_val_PMMA)
        
        if now_SP > SP_VAL[i]:
            E_BIND[i] = e
#            CC[i] = SP_VAL[i] / now_SP
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
#plt.semilogx(EE, CC, label='My')
#
#plt.title('C(E)')
#plt.xlabel('e, eV')
#plt.ylabel('C(E)')
#
#plt.legend()
#plt.grid()
#plt.show()
#
#plt.savefig('My C(E).png', dpi=300)

#%%
SP_VAL_TEST = np.zeros(len(EE))

for i in range(len(EE)):
    
    SP_VAL_TEST[i] = elf.get_Gryzinski_SP(EE[i:i+1], E_BIND[i], mc.n_PMMA_mon, elf.n_val_PMMA)


plt.loglog(EE, SP_VAL, label='Difference')
plt.loglog(EE, SP_VAL_TEST, '--', label='My')

plt.title('SP$_{valence}(E)$')
plt.xlabel('e, eV')
plt.ylabel('SP$_{valence}$')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('SP_valence_no_C.png', dpi=300)


#%%
U_TOT = np.load('../diel_responce/Dapor/U_PMMA_DAPOR.npy')
#U_CORE = np.load('U_PMMA_CORE.npy')
U_CORE = elf.get_PMMA_Gryzinski_core_U(EE)

E_BIND = np.load('E_BIND_PMMA.npy')
CC = np.load('C_PMMA.npy')

U_VAL = U_TOT - U_CORE
U_VAL_TEST = np.zeros(len(EE))

for i in range(len(EE)):
    
    U_VAL_TEST[i] = elf.get_Gryzinski_CS(EE[i:i+1], E_BIND[i]) *\
        CC[i] * mc.n_PMMA_mon * elf.n_val_PMMA

plt.loglog(EE, U_VAL, label='Difference')
plt.loglog(EE, U_VAL_TEST, '--', label='My')

plt.title('$\mu_{valence}(E)$')
plt.xlabel('e, eV')
plt.ylabel('$\mu_{valence}$, cm$^{-1}$')

plt.legend()
plt.grid()
plt.show()

#plt.savefig('U_valence.png', dpi=300)


