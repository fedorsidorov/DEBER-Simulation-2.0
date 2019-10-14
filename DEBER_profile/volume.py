#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

from itertools import product

import my_constants as mc
import my_utilities as mu
import my_arrays_Dapor as ma

mc = importlib.reload(mc)
mu = importlib.reload(mu)
ma = importlib.reload(ma)

os.chdir(mc.sim_folder + 'DEBER_profile')


#%%
profile = np.loadtxt('EXP_3_new.txt')

profile = profile[profile[:, 0].argsort()]

height = 0.94

xx, yy = profile[:, 0], profile [:, 1]
yy_900 = yy + height - yy[-1]

plt.plot(xx, yy)


#%%
plt.plot(xx, yy_900)
plt.ylim(0, 1)

volume_um2  = (xx[-1] - xx[0]) * height - np.trapz(yy_900, x=xx)

print(volume_um2)




