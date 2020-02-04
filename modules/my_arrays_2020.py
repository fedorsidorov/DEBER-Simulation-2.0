#%% Import
import numpy as np
import my_constants as mc
import my_utilities as mu

import prepare_PMMA as pp
import prepare_Si as ps

import importlib
mc = importlib.reload(mc)
mu = importlib.reload(mu)
pp = importlib.reload(pp)
ps = importlib.reload(ps)

#import matplotlib.pyplot as plt


#%%
EE = mc.EE
THETA = mc.THETA

#%%
PMMA_processes_U = pp.PMMA_processes_U
PMMA_processes_int_U = pp.PMMA_processes_int_U
PMMA_E_bind = pp.PMMA_E_bind


#%%
Si_processes_U = ps.Si_processes_U
Si_processes_int_U = ps.Si_processes_int_U
Si_E_bind = ps.Si_E_bind


#%%
processes_U = [PMMA_processes_U, Si_processes_U]
processes_int_U = [PMMA_processes_int_U, Si_processes_int_U]
E_bind = [PMMA_E_bind, Si_E_bind]
