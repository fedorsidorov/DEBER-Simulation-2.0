import numpy as np
import sys
from scipy import interpolate
import importlib
#import matplotlib.pyplot as plt
import my_constants as mc

mc = importlib.reload(mc)


#%% Non-simulation functions
def pbar(progress, total):
    
    barLength, status = 20, ''
    progress = float(progress) / float(total)
    
    if progress >= 1.:
    
        progress, status = 1, '\r\n'
    
    block = int(round(barLength * progress))
    
    text = '\r[{}] {:.0f}% {}'.format(
        '#' * block + '-' * (barLength - block), round(progress * 100, 0),\
        status)
    
    sys.stdout.write(text)
    sys.stdout.flush()


def log_log_interp(xx, yy, kind='linear'):
    
    logx = np.log10(xx)
    logy = np.log10(yy)

    lin_interp = interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    
    return log_interp


def log_log_interp_2d(xx, yy, zz, kind='linear'):
    
    logx = np.log10(xx)
    logy = np.log10(yy)
    logz = np.log10(zz)
    
    xm, ym = np.meshgrid(logx, logy)
    
    lin_interp = interpolate.interp2d(logx, logy, logz, kind='linear')
    log_interp = lambda zz, tt: np.power(10.0, lin_interp(np.log10(zz), np.log10(tt)))
    
    return log_interp


def lin_log_interp(xx, yy, kind='linear'):
    
    logy = np.log10(yy)

    lin_interp = interpolate.interp1d(xx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(zz))
    
    return log_interp


def log_lin_interp(xx, yy, kind='linear'):
    
    logx = np.log10(xx)

    lin_interp = interpolate.interp1d(logx, yy, kind=kind)
    log_interp = lambda zz: lin_interp(np.log10(zz))
    
    return log_interp


def get_closest_el_ind(array, val):
    
    return np.argmin(np.abs(array - val))


# def diff2int2d(diff_array, V, H):
    
#     int_array = np.zeros((len(V), len(H)))


#     for i in range(len(V)):
        
#         pbar(i, len(V))
        
#         integral = np.trapz(diff_array[i, :], x=H)
        
        
#         if integral == 0:
#             continue
        
        
#         for j in range(1, len(H)):
            
#             int_array[i, j] = np.trapz(diff_array[i, :j+1], x=H[:j+1]) / integral
    
    
#     return int_array


def get_cumulated_array(array):
    
    result = np.zeros((len(array)))
    
    
    for i in range(len(array)):
        
        if np.all(array == 0):
            continue
        
        result[i] = np.sum(array[:i+1])
    
    
    return result


# def diff2int_1d(yy, xx):
    
#     int_array = np.zeros(len(xx))

#     integral = np.trapz(yy, x=xx)
    
    
#     for j in range(1, len(xx)):
        
#         int_array[j] = np.trapz(yy[:j+1], x=xx[:j+1]) / integral
    
    
#     return int_array


# def normalize_u_array(arr):
    
#     arr_norm = np.zeros(np.shape(arr))
    
    
#     for i in range(len(arr)):
        
#         if np.all(arr[i, :] == 0):
#             continue
        
#         arr_norm[i, :] = arr[i, :] / np.sum(arr[i, :])
    
    
#     return arr_norm
            




