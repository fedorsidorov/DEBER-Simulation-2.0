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

os.chdir(os.path.join(mc.sim_folder, 'elastic', 'elsepa'))


#%%
def get_elsepa_diff_cs(fname):
    
    diff_cs = np.zeros((606, 2))
    

    with open(os.path.join(fname), 'r', encoding='utf-8') as f:
    
        file_lines = f.readlines()
    
    
    j = 0
    data = []
    

    for line in file_lines:
        
        if line[:2] == ' #':
            continue
        
        else:

            data.append(line)
            
            line_arr = line.split()
            
            diff_cs[j, 0] = line_arr[0]
            diff_cs[j, 1] = line_arr[2]
            
            j += 1
    
    
    return diff_cs


#%%
D = np.loadtxt('Au_1keV/Dapor.txt')
N = np.loadtxt('Au_1keV/NIST.txt')

e1 = get_elsepa_diff_cs('Au_1keV/MELEC3.dat')
e2 = get_elsepa_diff_cs('Au_1keV/MELEC3_IHEF0.dat')
e3 = get_elsepa_diff_cs('Au_1keV/easy.dat')
#e3 = get_elsepa_diff_cs('Au_1keV/MELEC3_IHEF0_MUFFIN1.dat')
#e4 = get_elsepa_diff_cs('Au_1keV/MELEC3_IHEF0_MUFFIN_my.dat')
e4 = get_elsepa_diff_cs('Au_1keV/test.dat')

plt.semilogy(D[:, 0], D[:, 1] * 1e-16, '.', label='Dapor')
plt.semilogy(N[:, 0], N[:, 1] * 2.8e-17, '-', label='NIST')
plt.semilogy(e3[:, 0], e3[:, 1], '--', label='elsepa, no parameters')
plt.semilogy(e1[:, 0], e1[:, 1], label='elsepa, MELEC=3')
#plt.semilogy(e2[:, 0], e2[:, 1], '--', label='elsepa, MELEC=3, IHEF=0')
#plt.semilogy(e3[:, 0], e3[:, 1], '--', label='elsepa, MELEC=3, IHEF=0, MUFFIN=1')
plt.semilogy(e4[:, 0], e4[:, 1], '--', label='elsepa, MELEC=3, IHEF=0, MUFFIN=1.59$\AA$')

plt.grid()
plt.legend()

plt.title('Differential elastic cross-section for Au, 1 keV')
plt.xlabel('theta, deg')
plt.ylabel('DESCS, cm$^2$/sr')

plt.xlim(0, 180)
plt.ylim(1e-19, 1e-14)

#plt.savefig('elastic_compare.jpg', dpi=500)
#plt.savefig('elastic_compare.pdf')
