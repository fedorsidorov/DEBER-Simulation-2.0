#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib

import my_constants as mc
import my_utilities as mu
import chain_functions as cf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
cf = importlib.reload(cf)

import time

os.chdir(mc.sim_folder + 'PMMA_sim')


#%%
source_dir = '/Volumes/ELEMENTS/Chains/'

print(os.listdir(source_dir))


#%%
chains = []

for folder in os.listdir(source_dir):
    
    if '.' in folder:
        continue
    
    print(folder)
    
    for i, fname in enumerate(os.listdir(os.path.join(source_dir, folder))):
        
        if 'DS_Store' in fname or '._' in fname:
            continue
        
        chains.append(np.load(os.path.join(source_dir, folder, fname)))


#%% prepare histograms
l_xyz = np.array((100, 100, 500))
space = 20

x_beg, y_beg, z_beg = -l_xyz[0]/2, -l_xyz[0]/2, 0
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

step_10nm = 10
step_2nm = 2

bins_total = np.array(np.hstack((xyz_beg.reshape(3, 1), xyz_end.reshape(3, 1))))

x_bins_10nm = np.arange(x_beg, x_end+1, step_10nm)
y_bins_10nm = np.arange(y_beg, y_end+1, step_10nm)
z_bins_10nm = np.arange(z_beg, z_end+1, step_10nm)

x_bins_2nm = np.arange(x_beg, x_end+1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end+1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end+1, step_2nm)

bins_10nm = [x_bins_10nm, y_bins_10nm, z_bins_10nm]
bins_2nm = [x_bins_2nm, y_bins_2nm, z_bins_2nm]


#%% create chain_list and check density
chain_list = []

m = np.load('harris_x_before.npy')
#mw = np.load('harris_y_before_fit.npy')
mw = np.load('harris_y_before_SZ.npy')

plt.semilogx(m, mw, 'ro')


#%%
V = np.prod(l_xyz) * (1e-7)**3 ## cm^3
m_mon = 1.66e-22 ## g
rho = 1.19 ## g / cm^3

n_mon_required = rho * V / m_mon


#%%
hist_total = np.zeros((1, 1, 1))
#hist_10nm = np.zeros((len(x_bins_10nm) - 1, len(y_bins_10nm) - 1, len(z_bins_10nm) - 1))
hist_2nm = np.zeros((len(x_bins_2nm) - 1, len(y_bins_2nm) - 1, len(z_bins_2nm) - 1))


#%%
t0 = time.time()

i = 0

n_mon_now = 0
n_mon_pre = 0


while True:
    
    if n_mon_pre > n_mon_required:
        print('Needed density is achieved')
        break
    
    mu.upd_progress_bar(n_mon_now, n_mon_required)
    
    chain_ind = np.random.randint(len(chains))
    now_chain = chains[chain_ind]
    
    now_len = cf.get_chain_len(m, mw)
    beg_ind = np.random.randint(len(now_chain) - now_len)
    
    fragment = now_chain[beg_ind:beg_ind+now_len]
    
    f_max = np.max(fragment, axis=0)
    f_min = np.min(fragment, axis=0)
    
    shift = np.random.uniform(xyz_beg - f_min - space, xyz_end - f_max + space)
    fragment_f = fragment + shift
    
    hist_total += np.histogramdd(fragment_f, bins=bins_total)[0]
#    hist_10nm += np.histogramdd(fragment_f, bins=bins_10nm)[0]
    hist_2nm += np.histogramdd(fragment_f, bins=bins_2nm)[0]
    
    n_mon_now = np.sum(hist_total)
    
    if n_mon_now > n_mon_pre:
        chain_list.append(fragment_f)
    
    n_mon_pre = n_mon_now
    
    i += 1
    

t1 = time.time()
dt = t1 - t0


#%%
lens = np.zeros(len(chain_list))

for i in range(len(chain_list)):
    lens[i] = len(chain_list[i])


#%%
lens_arr = np.array(lens)


#%% check density
density_total = hist_total[0][0][0] * m_mon / V
#density_prec = hist_prec * m_mon / V * (len(x_bins_prec) - 1) * (len(y_bins_prec) - 1) *\
#    (len(z_bins_prec) - 1)


#%%
n_mon_max = hist_2nm.max()
print(np.sum(hist_2nm) * m_mon / V)


#%% save chains to files
source_dir = '/Volumes/ELEMENTS/Chains_Harris'

i = 0

for chain in chain_list:
    
    mu.upd_progress_bar(i, len(chain_list))
    np.save(os.path.join(source_dir, 'chain_shift_' + str(i) + '.npy'), chain)
    i += 1


#%% cut chains to cube shape
#chain_cut_list = []
#
#for chain in chain_list:
#    
#    statements = [chain[:, 0] >= x_beg, chain[:, 0] <= x_end,
#                  chain[:, 1] >= y_beg, chain[:, 1] <= y_end]
#    inds = np.where(np.logical_and.reduce(statements))[0]
#    
#    beg = 0
#    end = -1
#    
#    for i in range(len(inds) - 1):
#        if inds[i+1] > inds[i] + 1 or i == len(inds) - 2:
#            end = i + 1
#            chain_cut_list.append(chain[inds[beg:end], :])
#            beg = i + 1


#%% get nice 3D picture
#l_x, l_y, l_z = l_xyz
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#for chain in chain_list[0:-1:50]:
#    ax.plot(chain[:, 0], chain[:, 1], chain[:, 2])
#
#ax.plot(np.linspace(x_beg, x_end, l_x), np.ones(l_x)*y_beg, np.ones(l_x)*z_beg, 'k')
#ax.plot(np.linspace(x_beg, x_end, l_x), np.ones(l_x)*y_end, np.ones(l_x)*z_beg, 'k')
#ax.plot(np.linspace(x_beg, x_end, l_x), np.ones(l_x)*y_beg, np.ones(l_x)*z_end, 'k')
#ax.plot(np.linspace(x_beg, x_end, l_x), np.ones(l_x)*y_end, np.ones(l_x)*z_end, 'k')
#
#ax.plot(np.ones(l_y)*x_beg, np.linspace(y_beg, y_end, l_y), np.ones(l_y)*z_beg, 'k')
#ax.plot(np.ones(l_y)*x_end, np.linspace(y_beg, y_end, l_y), np.ones(l_y)*z_beg, 'k')
#ax.plot(np.ones(l_y)*x_beg, np.linspace(y_beg, y_end, l_y), np.ones(l_y)*z_end, 'k')
#ax.plot(np.ones(l_y)*x_end, np.linspace(y_beg, y_end, l_y), np.ones(l_y)*z_end, 'k')
#
#ax.plot(np.ones(l_z)*x_beg, np.ones(l_z)*y_beg, np.linspace(z_beg, z_end, l_z), 'k')
#ax.plot(np.ones(l_z)*x_end, np.ones(l_z)*y_beg, np.linspace(z_beg, z_end, l_z), 'k')
#ax.plot(np.ones(l_z)*x_beg, np.ones(l_z)*y_end, np.linspace(z_beg, z_end, l_z), 'k')
#ax.plot(np.ones(l_z)*x_end, np.ones(l_z)*y_end, np.linspace(z_beg, z_end, l_z), 'k')
#
#plt.xlim(x_beg, x_end)
#plt.ylim(y_beg, y_end)
#plt.title('Polymer chain simulation')
#ax.set_xlabel('x, nm')
#ax.set_ylabel('y, nm')
#ax.set_zlabel('z, nm')
#plt.show()

