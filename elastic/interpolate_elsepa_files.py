#%% Import
import numpy as np
import os
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import importlib

import my_constants as mc
import my_utilities as mu

mc = importlib.reload(mc)
mu = importlib.reload(mu)

os.chdir(os.path.join(mc.sim_folder, 'elastic'))


#%%
def interpolate_diff_cs(diff_cs_raw, EE_raw, THETA_deg_raw):
    
    diff_cs_pre_dummy = np.concatenate((diff_cs_raw[:1, :]*0,
                                        diff_cs_raw, diff_cs_raw[-1:, :]*0), axis=0)
    
    EE_raw_ext = np.concatenate((mc.EE[:1], EE_raw, mc.EE[-1:]))
    
    diff_cs_pre_dummy[0, :] = diff_cs_pre_dummy[1, :]
    diff_cs_pre_dummy[-1, :] = diff_cs_pre_dummy[-2, :]
    
    diff_cs_pre = np.zeros((len(mc.EE), len(THETA_deg_raw)))
    
    
    for j in range(len(THETA_deg_raw)):
        
        diff_cs_pre[:, j] = mu.log_interp1d(EE_raw_ext, diff_cs_pre_dummy[:, j])(mc.EE)
    
    
    diff_cs = np.zeros((len(mc.EE), len(mc.THETA_deg)))
    
    
    for i in range(len(mc.EE)):
        
        now_diff_cs_pre = [diff_cs_pre[i, 0]]
        now_THETA_deg_raw = [THETA_deg_raw[0]]
    
    
        for j in range(1, len(THETA_deg_raw)):
            
            if diff_cs_pre[i, j] != now_diff_cs_pre[-1]:
                
                now_diff_cs_pre.append(diff_cs_pre[i, j])
                now_THETA_deg_raw.append(THETA_deg_raw[j])
                
        
        diff_cs[i, :] = mu.semilogy_interp1d(now_THETA_deg_raw, now_diff_cs_pre)(mc.THETA_deg)
    
    
    return diff_cs


def interpolate_cs(cs_raw, EE_raw):
    
    cs_dummy = np.concatenate((cs_raw[:1]*0,
                                        cs_raw, cs_raw[-1:]*0), axis=0)
    
    EE_raw_ext = np.concatenate((mc.EE[:1], EE_raw, mc.EE[-1:]))
    
    cs_dummy[0] = cs_dummy[1]
    cs_dummy[-1] = cs_dummy[-2]
    
    cs = mu.log_interp1d(EE_raw_ext, cs_dummy)(mc.EE)
    
    return cs


#%%
el = 'Si'

EE_raw = np.load('raw_arrays/elsepa_EE.npy')
THETA_deg_raw = np.load('raw_arrays/elsepa_theta.npy')

diff_cs_raw = np.load('raw_arrays/Si_atomic_diff_cs.npy')
cs_raw = np.load('raw_arrays/Si_atomic_cs.npy')

final_diff_cs = interpolate_diff_cs(diff_cs_raw, EE_raw, THETA_deg_raw)
final_cs = interpolate_cs(cs_raw, EE_raw)

final_int_cs = mu.diff2int(final_diff_cs, mc.EE, mc.THETA_deg)


#%%
#for i in range(228, len(mc.EE), 100):
#    
#    E_str = "{:6.1f}".format(mc.EE[i])
#    
#    plt.semilogy(THETA_deg_raw, ans[i, :], label=E_str + ' eV')
#    plt.semilogy(mc.THETA_deg[::50], bns[i, ::50], '.')
#    
#    now_x = np.concatenate((-np.deg2rad(THETA_deg_raw), np.deg2rad(THETA_deg_raw)))
#    now_y = np.concatenate((np.log10(ans[i, :]), np.log10(ans[i, :])))
#    
#    plt.polar(now_x, now_y)
#
#
#plt.savefig('elastic_interpolation_polar.jpg', dpi=500)

#%%
plt.title('Differential elastic cross-section for Si')
plt.xlabel('E, eV')
plt.ylabel('DESCS, cm$^2$/sr')

plt.xlim(0, 180)
plt.ylim(1e-22, 1e-14)

plt.legend()
plt.grid()


plt.savefig('elastic_interpolation_polar.jpg', dpi=500)
