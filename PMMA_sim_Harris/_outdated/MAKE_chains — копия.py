#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)
os.chdir(mv.sim_path_MAC + 'make_chains')

#%%
def get_L_array(n_chains):

    x_FS = np.load('../L_distribution_simulation/x_FS.npy')[1:]
    y_FS = np.load('../L_distribution_simulation/y_FS.npy')[1:]
    
    x_FS_log = np.log10(x_FS)
    S_tot = np.trapz(y_FS, x=x_FS_log)
    
    plt.plot(x_FS_log, y_FS/S_tot, label='density')
    plt.title('Chain mass distribution')
    plt.xlabel('log(m$_w$)')
    plt.ylabel('probability')
    
    S_FS_log = np.zeros(len(x_FS_log))
    
    for i in range(len(x_FS)):
        S_FS_log[i] = np.trapz(y_FS[:i+1], x=x_FS_log[:i+1]) / S_tot
    
    plt.plot(x_FS_log, S_FS_log, label='sum')
    
    n = 10000
    M_arr = np.zeros(n)
    
    for i in range(len(M_arr)):
        S_rand = mf.random()
        m = x_FS_log[mf.get_closest_el_ind(S_FS_log, S_rand)]
        M_arr[i] = 10 ** m
    
    plt.hist(np.log10(M_arr), bins=20, normed=True, label='sample')
    plt.legend()
    plt.grid()
    plt.show()
    
    L_arr = M_arr / mv.u_PMMA
    
    return L_arr

L_arr = get_L_array(10000)

#%%
def check_chain(chain_coords, now_mon_coords, d_2):
    for mon_coords in chain_coords[:-1, :]:    
        if np.sum((mon_coords - now_mon_coords)**2) < d_2:
            return False
    return True

#%%
#L_arr = np.load('L_arr_10k_experiment.npy')
#plt.hist(np.log10(L_arr*mv.u_PMMA), bins=20, normed=True, label='sample')
#plt.title('Chain mass distribution')
#plt.xlabel('log(m$_w$)')
#plt.ylabel('probability')
#plt.legend()
#plt.grid()
#plt.show()

#%%
d_mon = 0.28
d_mon_2 = d_mon**2

## Experiment
#lz = 122

## Aktary
lz = 100

#theta = np.deg2rad(180 - 109)
theta = np.deg2rad(109)

#n_chains = len(L_arr)
n_chains = 1

#chain_num = 7961
chain_num = 0

chains_list = []

while chain_num < n_chains:
    
#    L = int(L_arr[chain_num])
    L = 500
    print('New chain, L =', L)
    
    chain_coords = np.zeros((L, 3))
    chain_coords[0, :] = 0, 0, lz * mf.random()
    
    On = mf.get_O_matrix(2 * np.pi * mf.random(), theta, np.eye(3))
    
    ## collision counter
    jam_cnt = 0
    ## collision link number
    jam_pos = 0
    
    i = 1
    
    while i < L:
        
        mf.upd_progress_bar(i, L)
            
        while True:
            
            dxdydz = On.transpose() * np.mat([[0], [0], [1]]) * d_mon
            chain_coords[i, :] = chain_coords[i-1, :] + dxdydz.A1
            On = mf.get_O_matrix(2 * np.pi * mf.random(), theta, On)
            
            ## height is OK
            st1 = 0 <= chain_coords[i, 2] <= lz
            ## distance is OK
            st2 = check_chain(chain_coords[:i, :], chain_coords[i, :], d_mon_2)
            
            if st1 and st2:
                break
            
            else: ## if no free space
                
                if np.abs(jam_pos - i) < 10: ## if new jam is near current link
                    jam_cnt += 1 ## increase collision counter
                
                else: ## if new jam is on new link
                    jam_pos = i ## set new collision position
                    jam_cnt = 0 ## set collision counter to 0
                
                print(i, ': No free space,', jam_cnt)
                
                ## if possible, make rollback proportional to jam_cnt
                rollback_step = jam_cnt // 10
                
                if i - (rollback_step + 1) >= 0:
                    i -= rollback_step
                    continue
                
                else:
                    print('Jam in very start!')
                    break
        
        i += 1
    
    dirname = '../CHAINS/950K_100nm/'
    filename = 'chain_' + str(chain_num) + '.npy'
#    np.save(dirname + filename, chain_coords)
    print(filename + ' is saved')
    chains_list.append(chain_coords)
    chain_num += 1

#%%
chain_arr = chain_coords
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

beg=0
end=500

ax.plot(chain_arr[beg-1:beg+1, 0], chain_arr[beg-1:beg+1, 1], chain_arr[beg-1:beg+1, 2],\
        'b--')
ax.plot(chain_arr[beg:end, 0], chain_arr[beg:end, 1], chain_arr[beg:end, 2], 'b-')
ax.plot(chain_arr[end-1:end+1, 0], chain_arr[end-1:end+1, 1], chain_arr[end-1:end+1, 2],\
        'b--')
#plt.title('Polymer chain simulation, L = 3300')

ax.set_xlabel('x, nm')
ax.set_ylabel('y, nm')
ax.set_zlabel('z, nm')
plt.show()

