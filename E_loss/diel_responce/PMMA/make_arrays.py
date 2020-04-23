#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
from scipy import integrate

import my_constants as mc
import my_utilities as mu
import mermin

mc = importlib.reload(mc)
mu = importlib.reload(mu)
mermin = importlib.reload(mermin)

os.chdir(os.path.join(mc.sim_folder,
        'E_loss', 'diel_responce', 'Dapor'
        ))


#%%
DIIMFP = np.zeros((len(mc.EE), len(mc.EE)))
IIMFP = np.zeros(len(mc.EE))


for T, 