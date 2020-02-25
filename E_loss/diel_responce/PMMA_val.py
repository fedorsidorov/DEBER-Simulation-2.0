#%% Import
import numpy as np
import os
import importlib
import my_constants as mc
import my_utilities as mu

from itertools import product

import matplotlib.pyplot as plt

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce'
        ))


#%%
## Total
S = np.load('PMMA_dapor2015/final/PMMA_S_f_dapor2015.npy')
u = np.load('PMMA_dapor2015/final/PMMA_u_f_dapor2015.npy')
tau = np.load('PMMA_dapor2015/final/PMMA_tau_f_dapor2015.npy')

