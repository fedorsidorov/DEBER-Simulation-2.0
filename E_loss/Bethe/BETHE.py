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

#%%
def get_Bethe_SP(E, Z, rho, A, J):
    
    K = 0.734 * Z**0.037
    dEds = 785 * rho*Z / (A*E) * np.log(1.166 * (E + K*J) / J) * 1e+8 ## eV/cm
    
    return dEds

#%% PMMA
E = np.logspace(0, 4.4, 1000)

Z_PMMA = mc.Z_PMMA
rho_PMMA = mc.rho_PMMA
A_PMMA = mc.u_PMMA / 15
J_PMMA = 65.6

dEds_PMMA = get_Bethe_SP(E, Z_PMMA, rho_PMMA, A_PMMA, J_PMMA)

plt.loglog(E, dEds_PMMA)

#%% Si
E = np.logspace(0, 4.4, 1000)

Z_Si = mc.Z_Si
rho_Si = mc.rho_Si
A_Si = mc.u_Si
J_Si = 173

dEds_Si = get_Bethe_SP(E, Z_Si, rho_Si, A_Si, J_Si)

plt.loglog(E, dEds_Si)



