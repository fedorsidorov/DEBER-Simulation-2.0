#%% Import
import numpy as np
import matplotlib.pyplot as plt

import my_constants as mc
import my_utilities as mu

import prepare_PMMA_sample as pp
import prepare_Si_sample as ps

import importlib
mc = importlib.reload(mc)
mu = importlib.reload(mu)
pp = importlib.reload(pp)
ps = importlib.reload(ps)


#%%
E_bind = ps.E_bind

u_processes = [pp.u_processes, ps.u_processes]
sigma_diff_sample_processes = [pp.sigma_diff_sample_processes, 
                               ps.sigma_diff_sample_processes]

# E_bind = [PMMA_E_bind, Si_E_bind]

E_cut = [mc.E_cut_PMMA, mc.E_cut_Si]


#%%
plt.loglog(mc.EE, pp.u_processes[:, 0])
plt.loglog(mc.EE, ps.u_processes[:, 0])

