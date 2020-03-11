#%% Import
import numpy as np

import my_constants as mc
import my_utilities as mu
import prepare_PMMA as pp
import prepare_Si as ps
#import scission_functions_2020 as sf

import importlib

mc = importlib.reload(mc)
mu = importlib.reload(mu)
pp = importlib.reload(pp)
ps = importlib.reload(ps)
#sf = importlib.reload(sf)

#import matplotlib.pyplot as plt


#%%
proc_u = [pp.u_table, ps.u_table]
proc_u_norm = [pp.u_table_norm, ps.u_table_norm]

proc_tau_cumulated = [pp.tau_cumulated_table, ps.tau_cumulated_table]

E_bind = [pp.E_bind, ps.E_bind]
E_cut = [pp.E_cut, ps.E_cut]

PMMA_val_E_bind = pp.val_E_bind

scission_probs = pp.scission_probs

