#%% Import
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import my_functions as mf
import my_variables as mv

mf = importlib.reload(mf)
mv = importlib.reload(mv)
os.chdir(mv.sim_path_MAC + 'make_chain_arrays')

#%%
source_dir = mv.sim_path_MAC + 'CHAINS/Harris_500nm/raw/'
dest_dir = mv.sim_path_MAC + 'CHAINS/Harris_500nm/comb_100x100x500_center/'

#%%
N_chains = 204

chain_bank = []

for i in range(N_chains):
    
    mf.upd_progress_bar(i, N_chains)
    
    now_chain = np.load(source_dir + 'chain_' + str(i) + '.npy')
    chain_bank.append(now_chain)

#%% prepare histograms
l_xyz = np.array((100, 100, 500))
space = 50

x_beg, y_beg, z_beg = (-l_xyz[0]/2, -l_xyz[0]/2, 0)
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

step_prec = 10
step_2nm = 2

bins_total = np.array(np.hstack((xyz_beg.reshape(3, 1), xyz_end.reshape(3, 1))))

x_bins_prec = np.arange(x_beg, x_end + 1, step_prec)
y_bins_prec = np.arange(y_beg, y_end + 1, step_prec)
z_bins_prec = np.arange(z_beg, z_end + 1, step_prec)

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)

bins_prec = [x_bins_prec, y_bins_prec, z_bins_prec]
bins_2nm = [x_bins_2nm, y_bins_2nm, z_bins_2nm]

hist_total = np.zeros((1, 1, 1))
hist_prec = np.zeros((len(x_bins_prec) - 1, len(y_bins_prec) - 1, len(z_bins_prec) - 1))
hist_2nm = np.zeros((len(x_bins_2nm) - 1, len(y_bins_2nm) - 1, len(z_bins_2nm) - 1))

#%% create chain_list and check density
chain_list = []

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
        
        n_1, n_2, n_3 = mf.choice(N_chains, size=3, replace=False)

        chain_1, chain_2, chain_3 = chain_bank[n_1], chain_bank[n_2], chain_bank[n_3]
        
        chain_12 = np.vstack([
                chain_1,
                chain_2[1:, :] - chain_2[:1, :] + chain_1[-1:, :]
                ])
        
        chain_123 = np.vstack([
                chain_12,
                chain_3[1:, :] - chain_3[:1, :] + chain_12[-1:, :]
                ])
        
        chain_arr = chain_123 - chain_123[:1, :]
#        chain_arr = chain_123
        
        z_min = chain_arr[:, 2].min(axis=0)
        z_max = chain_arr[:, 2].max(axis=0)
        
        x_shift = mf.uniform(x_beg - space, x_end + space)
        y_shift = mf.uniform(y_beg - space, y_end + space)
        
        z_shift = mf.uniform(z_beg - z_min, z_end - z_max)
#        z_shift = mf.uniform(z_beg, z_end)
        
        now_chain_shift = chain_arr + np.array((x_shift, y_shift, z_shift))
#        now_chain_shift = now_chain + np.array((x_shift, y_shift, 0))
        
        if now_chain_shift.max(axis=0)[0] < x_beg or\
           now_chain_shift.max(axis=0)[1] < y_beg or\
           now_chain_shift.min(axis=0)[0] > x_end or\
           now_chain_shift.min(axis=0)[1] > y_end:
            continue
        
        chain_list.append(now_chain_shift)
        
        hist_total += np.histogramdd(now_chain_shift, bins=bins_total)[0]
        hist_prec += np.histogramdd(now_chain_shift, bins=bins_prec)[0]
        hist_2nm += np.histogramdd(now_chain_shift, bins=bins_2nm)[0]
        
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

#%%
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#beg=0
#end=7500
#
#ax.plot(chain_arr[beg-1:beg+1, 0], chain_arr[beg-1:beg+1, 1], chain_arr[beg-1:beg+1, 2],\
#        'b--')
#ax.plot(chain_arr[beg:end, 0], chain_arr[beg:end, 1], chain_arr[beg:end, 2], 'b-')
#ax.plot(chain_arr[end-1:end+1, 0], chain_arr[end-1:end+1, 1], chain_arr[end-1:end+1, 2],\
#        'b--')
##plt.title('Polymer chain simulation, L = 3300')
#
#ax.set_xlabel('x, nm')
#ax.set_ylabel('y, nm')
#ax.set_zlabel('z, nm')
#plt.show()
