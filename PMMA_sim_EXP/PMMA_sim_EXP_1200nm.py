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

os.chdir(mc.sim_folder + 'PMMA_sim_EXP')


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
l_xyz = np.array((1200, 10, 900))

space = np.array((500, 500, 10))

x_min, y_min, z_min = -l_xyz[0]/2, -l_xyz[0]/2, 0
xyz_min = np.array((x_min, y_min, z_min))
xyz_max = xyz_min + l_xyz
x_max, y_max, z_max = xyz_max

step_2nm = 2

bins_total = np.array(np.hstack((xyz_min.reshape(3, 1), xyz_max.reshape(3, 1))))

x_bins_2nm = np.arange(x_min, x_max+1, step_2nm)
y_bins_2nm = np.arange(y_min, y_max+1, step_2nm)
z_bins_2nm = np.arange(z_min, z_max+1, step_2nm)

bins_2nm = [x_bins_2nm, y_bins_2nm, z_bins_2nm]


#%% create chain_list and check density

m = np.load('x_EXP_log.npy')
mw = np.load('y_EXP_log.npy')

#plt.semilogx(m, mw, 'ro')


#%%
#lens = np.zeros(10000)
#
#for i in range(len(lens)):
#    
#    mu.pbar(i, len(lens))
#    
#    now_len = cf.get_chain_len(m, mw)
#    lens[i] = now_len 
#
#
##%%
#xx = np.load('x_EXP_log.npy')
#yy = np.load('y_EXP_log.npy')
#
#mass = np.array(lens)*100
#
#bins = np.logspace(2, 7.1, 21)
#
#plt.hist(mass, bins)
#plt.gca().set_xscale('log')
#
#plt.plot(xx, yy*5.3e+2, label='Schulz-Zimm')
#
#plt.title('Experiment chain sample')
#plt.xlabel('molecular weight')
#plt.ylabel('density')
#
##plt.legend()
#plt.grid()
#plt.show()


#%%
V = np.prod(l_xyz) * (1e-7)**3 ## cm^3
m_mon = 1.66e-22 ## g
rho = 1.19 ## g / cm^3

n_mon_required = rho * V / m_mon


#%%
hist_total = np.zeros((1, 1, 1))
hist_2nm = np.zeros((len(x_bins_2nm) - 1, len(y_bins_2nm) - 1, len(z_bins_2nm) - 1))

chain_list = []
chain_lens = []

dest_dir = '/Volumes/ELEMENTS/Chains_EXP_1200nm/'

t0 = time.time()

i = 0

n_mon_now = 0
n_mon_pre = 0

lens = []


while True:
    
    if n_mon_pre > n_mon_required:
        print('Needed density is achieved')
        break
    
    mu.pbar(n_mon_now, n_mon_required)
    
    chain_ind = np.random.randint(len(chains))
    now_chain = chains[chain_ind]
    
    now_len = cf.get_chain_len(m, mw)
    chain_lens.append(now_len)
    
    beg_ind = np.random.randint(len(now_chain) - now_len)
    fragment = now_chain[beg_ind:beg_ind+now_len]
    
    f_max = np.max(fragment, axis=0)
    f_min = np.min(fragment, axis=0)
    
    lens.append(np.max(f_max - f_min))
    
    shift = np.random.uniform(xyz_min - f_min - space, xyz_max - f_max + space)
    
    fragment_f = fragment + shift
    
    cf.snake_chain(fragment_f, xyz_min, xyz_max)            
    
    if np.any(np.min(fragment_f, axis=0) < xyz_min) or\
           np.any(np.max(fragment_f, axis=0) >= xyz_max):
        print(i)
    
    hist_total += np.histogramdd(fragment_f, bins=bins_total)[0]
    hist_2nm += np.histogramdd(fragment_f, bins=bins_2nm)[0]
    
    n_mon_now = np.sum(hist_total)
    
    if n_mon_now > n_mon_pre:
        np.save(dest_dir + 'chain_shift_' + str(i) + '.npy', fragment_f)
        i += 1
    
    n_mon_pre = n_mon_now    
    

t1 = time.time()
dt = t1 - t0


lens_arr = np.array(lens)


#%%
print('2nm average =', np.average(hist_2nm))
print('2nm average density =', np.average(hist_2nm)*m_mon/(2e-7)**3)


#%%
total_rho = np.sum(chain_lens)*m_mon / V

print('total rho =', total_rho)


#%%
n_empty = 600*5*450 - np.count_nonzero(hist_2nm)
part_empty = n_empty / (600*5*450)

print(part_empty)


#%%
xx = np.load('x_EXP_log.npy')
yy = np.load('y_EXP_log.npy')

mass = np.array(chain_lens)*100

bins = np.logspace(2, 7.1, 21)

plt.hist(mass, bins)
plt.gca().set_xscale('log')

plt.plot(xx, yy*3.9e+3, label='Schulz-Zimm')

plt.title('Experiment chain sample')
plt.xlabel('molecular weight')
plt.ylabel('density')

#plt.legend()
plt.grid()
plt.show()

plt.savefig('EXP_sample_period_1200nm.png', dpi=300)


#%% check density
density_total = hist_total[0][0][0] * m_mon / V
#density_prec = hist_prec * m_mon / V * (len(x_bins_prec) - 1) * (len(y_bins_prec) - 1) *\
#    (len(z_bins_prec) - 1)


#%%
n_mon_max = hist_2nm.max()
print(np.sum(hist_2nm) * m_mon / V)


#%% cut chains to cube shape
#chain_cut_list = []
#
#for chain in chain_list:
#    
#    statements = [chain[:, 0] >= x_min, chain[:, 0] <= x_max,
#                  chain[:, 1] >= y_min, chain[:, 1] <= y_max]
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
#ax.plot(np.linspace(x_min, x_max, l_x), np.ones(l_x)*y_min, np.ones(l_x)*z_min, 'k')
#ax.plot(np.linspace(x_min, x_max, l_x), np.ones(l_x)*y_max, np.ones(l_x)*z_min, 'k')
#ax.plot(np.linspace(x_min, x_max, l_x), np.ones(l_x)*y_min, np.ones(l_x)*z_max, 'k')
#ax.plot(np.linspace(x_min, x_max, l_x), np.ones(l_x)*y_max, np.ones(l_x)*z_max, 'k')
#
#ax.plot(np.ones(l_y)*x_min, np.linspace(y_min, y_max, l_y), np.ones(l_y)*z_min, 'k')
#ax.plot(np.ones(l_y)*x_max, np.linspace(y_min, y_max, l_y), np.ones(l_y)*z_min, 'k')
#ax.plot(np.ones(l_y)*x_min, np.linspace(y_min, y_max, l_y), np.ones(l_y)*z_max, 'k')
#ax.plot(np.ones(l_y)*x_max, np.linspace(y_min, y_max, l_y), np.ones(l_y)*z_max, 'k')
#
#ax.plot(np.ones(l_z)*x_min, np.ones(l_z)*y_min, np.linspace(z_min, z_max, l_z), 'k')
#ax.plot(np.ones(l_z)*x_max, np.ones(l_z)*y_min, np.linspace(z_min, z_max, l_z), 'k')
#ax.plot(np.ones(l_z)*x_min, np.ones(l_z)*y_max, np.linspace(z_min, z_max, l_z), 'k')
#ax.plot(np.ones(l_z)*x_max, np.ones(l_z)*y_max, np.linspace(z_min, z_max, l_z), 'k')
#
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)
#plt.title('Polymer chain simulation')
#ax.set_xlabel('x, nm')
#ax.set_ylabel('y, nm')
#ax.set_zlabel('z, nm')
#plt.show()

