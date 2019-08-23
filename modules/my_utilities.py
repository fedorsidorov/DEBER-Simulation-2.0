import numpy as np
import sys
from scipy import interpolate
import importlib
import matplotlib.pyplot as plt
import my_constants as mc

mc = importlib.reload(mc)


#%% Non-simulation functions
def upd_progress_bar(progress, total):
    
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


def log_interp1d(xx, yy, kind='linear'):
    
    logx = np.log10(xx)
    logy = np.log10(yy)

    lin_interp = interpolate.interp1d(logx, logy, kind=kind)

    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    
    return log_interp


def diff2int(DIFF, V=mc.EE, H=mc.EE):
    
    INT = np.zeros((len(V), len(H)))

    for i in range(len(V)):

        integral = np.trapz(DIFF[i, :], x=H)
        
        if integral == 0:
            continue
        
        for j in range(1, len(H)):
            INT[i, j] = np.trapz(DIFF[i, :j+1], x=H[:j+1]) / integral
    
    return INT


#%%
#s = np.linspace(0, 10, 1001)
#
#l1 = 6
#l2 = 3
#
#s1 = 5
#
#f1 = np.exp(-s/l1)
#f2 = np.exp(-s1/l1 - (s-s1)/l2)
#
#plt.plot(s, f1, label='f1')
#plt.plot(s, f2, label='f2')
#
#plt.xlabel('s')
#plt.ylabel('P')
#
#plt.grid()
#plt.show()


