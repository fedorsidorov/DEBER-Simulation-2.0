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

os.chdir(mc.sim_folder + 'PMMA_sim')


#%%
folders = ['chains_1M', 'chains_1M_FTIAN']

chains = []

for folder in folders:
    
    print(folder)
    
    now_dir = '../CHAINS/' + folder
    
    fnames = os.listdir(now_dir)
    
    for i, fname in enumerate(fnames):
        
        mu.upd_progress_bar(i, len(fnames))
        
        if fname == '.DS_Store':
            continue
        
        chains.append(np.load(now_dir + '/' + fname))


#%% prepare histograms
l_xyz = np.array((100, 100, 500))
space = 100

x_beg, y_beg, z_beg = -l_xyz[0]/2, -l_xyz[0]/2, 0
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

step_10nm = 10
step_2nm = 2

bins_total = np.array(np.hstack((xyz_beg.reshape(3, 1), xyz_end.reshape(3, 1))))

x_bins_10nm = np.arange(x_beg, x_end + 1, step_10nm)
y_bins_10nm = np.arange(y_beg, y_end + 1, step_10nm)
z_bins_10nm = np.arange(z_beg, z_end + 1, step_10nm)

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)

bins_10nm = [x_bins_10nm, y_bins_10nm, z_bins_10nm]
bins_2nm = [x_bins_2nm, y_bins_2nm, z_bins_2nm]

hist_total = np.zeros((1, 1, 1))
hist_10nm = np.zeros((len(x_bins_10nm) - 1, len(y_bins_10nm) - 1, len(z_bins_10nm) - 1))
hist_2nm = np.zeros((len(x_bins_2nm) - 1, len(y_bins_2nm) - 1, len(z_bins_2nm) - 1))


#%% create chain_list and check density
chain_list = []

m = np.load('harris_x_before.npy')
#mw = np.load('harris_y_before_fit.npy')
mw = np.load('harris_y_before_SZ.npy')

V = np.prod(l_xyz) * (1e-7)**3 ## cm^3
m_mon = 1.66e-22 ## g
rho = 1.19 ## g / cm^3

i = 0

while True:
    
    if i % 1000 == 0:
        print(i, 'chains are added')
    
    if np.sum(hist_total) * m_mon / V >= rho:
        print('Needed density is achieved')
        break
    
    else:
        
        now_len = cf.get_chain_len(m, mw)
        
        print(now_len)
        
        chain_ind = np.random.randint(len(chains))
        
        now_chain = chains[chain_ind]
        
        beg_ind = np.random.randint(len(now_chain) - now_len)
        
        fragment = now_chain[beg_ind:beg_ind+now_len]
        fragment_0 = fragment - fragment[0]
#        fragment_rot = cf.rotate_chain(fragment_0)
        
        fragment_rot = fragment_0
        
        fragment_c = fragment_rot +\
            (np.max(fragment_rot, axis=0) + np.min(fragment_rot, axis=0)) / 2
        
#        f_max_z = np.max(fragment_c, axis=0)[2]
#        f_min_z = np.min(fragment_c, axis=0)[2]
        
#        f_max_z = np.max(fragment_c, axis=0)[0, 2]
#        f_min_z = np.min(fragment_c, axis=0)[0, 2]
        
#        x_shift = np.random.uniform(x_beg - space, x_end + space)
#        y_shift = np.random.uniform(y_beg - space, y_end + space)
#        z_shift = np.random.uniform(0 - f_min_z, 500 - f_max_z)
        
#        fragment_f = now_chain + np.array((x_shift, y_shift, z_shift))
        
#        if fragment_f.max(axis=0)[0] < x_beg or\
#           fragment_f.max(axis=0)[1] < y_beg or\
#           fragment_f.min(axis=0)[0] > x_end or\
#           fragment_f.min(axis=0)[1] > y_end:
#            continue
        
        fragment_f = fragment_c
        
        chain_list.append(fragment_f)
        
        hist_total += np.histogramdd(fragment_f, bins=bins_total)[0]
        hist_10nm += np.histogramdd(fragment_f, bins=bins_10nm)[0]
        hist_2nm += np.histogramdd(fragment_f, bins=bins_2nm)[0]
        
    i += 1

#%% check density
density_total = hist_total[0][0][0] * m_mon / V
density_prec = hist_prec * m_mon / V * (len(x_bins_prec) - 1) * (len(y_bins_prec) - 1) *\
    (len(z_bins_prec) - 1)

#%%
n_mon_max = hist_2nm.max()
print(np.sum(hist_2nm) * m_mon / V)

#%% save chains to files
i = 0

for chain in chain_list:
    
    mf.upd_progress_bar(i, len(chain_list))
    np.save(dest_dir + 'chain_shift_' + str(i) + '.npy', chain)
    i += 1

#%% cut chains to cube shape
chain_cut_list = []

for chain in chain_list:
    
    statements = [chain[:, 0] >= x_beg, chain[:, 0] <= x_end,
                  chain[:, 1] >= y_beg, chain[:, 1] <= y_end]
    inds = np.where(np.logical_and.reduce(statements))[0]
    
    beg = 0
    end = -1
    
    for i in range(len(inds) - 1):
        if inds[i+1] > inds[i] + 1 or i == len(inds) - 2:
            end = i + 1
            chain_cut_list.append(chain[inds[beg:end], :])
            beg = i + 1

#%% get nice 3D picture
l_x, l_y, l_z = l_xyz

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for chain in chain_list[0:-1:50]:
    ax.plot(chain[:, 0], chain[:, 1], chain[:, 2])

ax.plot(np.linspace(x_beg, x_end, l_x), np.ones(l_x)*y_beg, np.ones(l_x)*z_beg, 'k')
ax.plot(np.linspace(x_beg, x_end, l_x), np.ones(l_x)*y_end, np.ones(l_x)*z_beg, 'k')
ax.plot(np.linspace(x_beg, x_end, l_x), np.ones(l_x)*y_beg, np.ones(l_x)*z_end, 'k')
ax.plot(np.linspace(x_beg, x_end, l_x), np.ones(l_x)*y_end, np.ones(l_x)*z_end, 'k')

ax.plot(np.ones(l_y)*x_beg, np.linspace(y_beg, y_end, l_y), np.ones(l_y)*z_beg, 'k')
ax.plot(np.ones(l_y)*x_end, np.linspace(y_beg, y_end, l_y), np.ones(l_y)*z_beg, 'k')
ax.plot(np.ones(l_y)*x_beg, np.linspace(y_beg, y_end, l_y), np.ones(l_y)*z_end, 'k')
ax.plot(np.ones(l_y)*x_end, np.linspace(y_beg, y_end, l_y), np.ones(l_y)*z_end, 'k')

ax.plot(np.ones(l_z)*x_beg, np.ones(l_z)*y_beg, np.linspace(z_beg, z_end, l_z), 'k')
ax.plot(np.ones(l_z)*x_end, np.ones(l_z)*y_beg, np.linspace(z_beg, z_end, l_z), 'k')
ax.plot(np.ones(l_z)*x_beg, np.ones(l_z)*y_end, np.linspace(z_beg, z_end, l_z), 'k')
ax.plot(np.ones(l_z)*x_end, np.ones(l_z)*y_end, np.linspace(z_beg, z_end, l_z), 'k')

plt.xlim(x_beg, x_end)
plt.ylim(y_beg, y_end)
plt.title('Polymer chain simulation')
ax.set_xlabel('x, nm')
ax.set_ylabel('y, nm')
ax.set_zlabel('z, nm')
plt.show()
