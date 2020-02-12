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
PMMA_processes_U = pp.PMMA_processes_U
PMMA_processes_int_U = pp.PMMA_processes_int_U
PMMA_E_bind = pp.PMMA_E_bind

scission_prob_gryz = sf.scission_probs_gryz(mc.EE)
PMMA_val_Eb = sf.Eb_Nel[:, 0]


#%%
Si_processes_U = ps.Si_processes_U
Si_processes_int_U = ps.Si_processes_int_U
Si_E_bind = ps.Si_E_bind


#%%
processes_U = [PMMA_processes_U, Si_processes_U]
processes_int_U = [PMMA_processes_int_U, Si_processes_int_U]
E_bind = [PMMA_E_bind, Si_E_bind]
