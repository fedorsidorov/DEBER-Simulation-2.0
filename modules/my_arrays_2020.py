#%% Import
import numpy as np
import my_constants as mc
import my_utilities as mu

import prepare_PMMA as pp
import prepare_Si as ps
import scission_functions as sf

import importlib
mc = importlib.reload(mc)
mu = importlib.reload(mu)
pp = importlib.reload(pp)
ps = importlib.reload(ps)
sf = importlib.reload(sf)

#import matplotlib.pyplot as plt


#%%
PMMA_proc_U = pp.u_proc
PMMA_proc_tau_int = pp.tau_int_list
PMMA_E_bind = pp.E_bind
PMMA_sc_prob_gryz = pp.scission_probs
PMMA_val_Eb = sf.Eb_Nel[:, 0]


#%%
Si_processes_U = ps.Si_processes_U
Si_processes_int_U = ps.Si_processes_int_U
Si_E_bind = ps.Si_E_bind


#%%
proc_u = [PMMA_processes_U, Si_processes_U]
proc_tau_int = [PMMA_processes_int_U, Si_processes_int_U]

E_bind = [PMMA_E_bind, Si_E_bind]
