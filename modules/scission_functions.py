#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

import numpy.random as rnd

import my_arrays_Dapor as ma

ma = importlib.reload(ma)


#%%
def scission_probs_ones(EE):
    
    return np.ones(len(EE))


def scission_probs_2СС(EE):
    
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


def scission_probs_2CC_ester(EE):

    result = np.ones(len(EE)) * 6/40
    result[np.where(EE < 815 * 0.0103)] = 6/(40 - 8)
    result[np.where(EE < 420 * 0.0103)] = 6/(40 - 8 - 4)
    result[np.where(EE < 418 * 0.0103)] = 6/(40 - 8 - 4 - 12)
    result[np.where(EE < 406 * 0.0103)] = 6/(40 - 8 - 4 - 12 - 4)
    result[np.where(EE < 383 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2)
    result[np.where(EE < 364 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4)
    result[np.where(EE < 356 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4 - 2)
    result[np.where(EE < 354 * 0.0103)] = 0
    
    return result


def scission_probs_2CC_3H(EE):
    
    result = np.ones(len(EE)) * (4 + 6)/40
    result[np.where(EE < 815 * 0.0103)] = (4 + 6)/(40 - 8)
    result[np.where(EE < 420 * 0.0103)] = (4 + 6)/(40 - 8 - 4)
    result[np.where(EE < 418 * 0.0103)] = 4/(40 - 8 - 4 - 12)
    result[np.where(EE < 406 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4)
    result[np.where(EE < 383 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2)
    result[np.where(EE < 364 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4)
    result[np.where(EE < 356 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4 - 2)
    result[np.where(EE < 354 * 0.0103)] = 0
    
    return result


## 160 C
def scission_probs_2CC_2H(EE):
        
    result =            np.ones(len(EE))* (4 + 4)/40
    result[np.where(EE < 815 * 0.0103)] = (4 + 4)/(40 - 8)
    result[np.where(EE < 420 * 0.0103)] = (4 + 4)/(40 - 8 - 4)
    result[np.where(EE < 418 * 0.0103)] =      4 / (40 - 8 - 4 - 12)
    result[np.where(EE < 406 * 0.0103)] =      4 / (40 - 8 - 4 - 12 - 4)
    result[np.where(EE < 383 * 0.0103)] =      4 / (40 - 8 - 4 - 12 - 4 - 2)
    result[np.where(EE < 364 * 0.0103)] =      4 / (40 - 8 - 4 - 12 - 4 - 2 - 4)
    result[np.where(EE < 356 * 0.0103)] =      4 / (40 - 8 - 4 - 12 - 4 - 2 - 4 - 2)
    result[np.where(EE < 354 * 0.0103)] = 0
    
    return result


def scission_probs_2CC_1p5H(EE):

    result = np.ones(len(EE)) * (4 + 3)/40
    result[np.where(EE < 815 * 0.0103)] = (4 + 3)/(40 - 8)
    result[np.where(EE < 420 * 0.0103)] = (4 + 3)/(40 - 8 - 4)
    result[np.where(EE < 418 * 0.0103)] = 4/(40 - 8 - 4 - 12)
    result[np.where(EE < 406 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4)
    result[np.where(EE < 383 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2)
    result[np.where(EE < 364 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4)
    result[np.where(EE < 356 * 0.0103)] = 4/(40 - 8 - 4 - 12 - 4 - 2 - 4 - 2)
    result[np.where(EE < 354 * 0.0103)] = 0
    
    return result


#%%
#end_ind = 200
#
#plt.plot(ma.EE[:end_ind], scission_probs_2CC_ester_H(ma.EE[:end_ind]))
#
#plt.title('Scission probability')
#plt.xlabel('E, eV')
#plt.ylabel('scission probability')
#
#plt.grid()


