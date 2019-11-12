#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

import numpy.random as rnd


#%%


def scission_probs_ones(EE):
    
    return np.ones(len(EE))


def scission_probs_room(EE):
    
    result = np.zeros(len(EE))
    
    result = np.ones(len(EE)) * 4/40
    result[np.where(EE < 815 * 0.0103)] = 4/(40 - 8)
    result[np.where(EE < 420 * 0.0103)] = 4/(40 - 8 - 4)
    result[np.where(EE < 418 * 0.0103)] = 4/(40 - 8 - 4 - 12)
    result[np.where(EE < 406 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4)
    result[np.where(EE < 383 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2)
    result[np.where(EE < 364 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4)
    result[np.where(EE < 356 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4 - 2)
    result[np.where(EE < 354 * 0.0103)] = 0
    
    return result


