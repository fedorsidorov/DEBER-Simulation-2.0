#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

import numpy.random as rnd

import my_constants as mc
import my_utilities as mu
import scission_functions_2020 as sf
import e_matrix_functions as emf

mc = importlib.reload(mc)
mu = importlib.reload(mu)
sf = importlib.reload(sf)
emf = importlib.reload(emf)

os.chdir(os.path.join(mc.sim_folder, 'e-matrix_Harris'))


#%%
path = os.path.join(mc.sim_folder, 'e_DATA', 'Harris')

folders = ['Harris_2020_MAC_0_G', 'Harris_2020_MAC_1_G', 'Harris_2020_UBU_0_G']


#%%
DATA_PMMA_list = []


for now_ind in range(len(folders)):
    
    mu.pbar(now_ind, len(folders))
    
    now_folder = os.path.join(path, folders[now_ind])
    now_folder_files = os.listdir(now_folder)
    
    pos = 0
    
    for file in now_folder_files:
        
        if 'DS' in file:
            continue
        
        now_DATA_PMMA = np.load(os.path.join(now_folder, file))
        now_DATA_PMMA[:, 5:8] *= 1e+7
        DATA_PMMA_list.append(now_DATA_PMMA)


#%%
l_xyz = np.array((100, 100, 500))
lx, ly, lz = l_xyz

x_beg, y_beg, z_beg = -lx/2, -ly/2, 0
xyz_beg = np.array((x_beg, y_beg, z_beg))
xyz_end = xyz_beg + l_xyz
x_end, y_end, z_end = xyz_end

step_2nm = 2

x_bins_2nm = np.arange(x_beg, x_end + 1, step_2nm)
y_bins_2nm = np.arange(y_beg, y_end + 1, step_2nm)
z_bins_2nm = np.arange(z_beg, z_end + 1, step_2nm)

bins_2nm = x_bins_2nm, y_bins_2nm, z_bins_2nm


#%%
def get_scissions(b_map_sc, now_DATA_PMMA):
    
    scissions = np.zeros(len(now_DATA_PMMA))
    
    
    for b in b_map_sc:
        
        pr_ind = 10 + np.where(sf.Eb_Nel[:, 0] == sf.MMA_bonds[b][0])[0][0]
        
#        inds = np.where(now_DATA_PMMA[:, 3] == pr_ind)[0]
        inds = np.where(np.abs(now_DATA_PMMA[:, 3]) == pr_ind)[0] ## ABS!!
        
        scissions[inds] = rnd.rand(len(inds)) < np.ones(len(inds)) *\
                b_map_sc[b] / sf.MMA_bonds[b][1]
    
    
    return scissions


#%%
e_matrix_shape = (len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1)

e_matrix_val = np.zeros(e_matrix_shape)
e_matrix_dE = np.zeros(e_matrix_shape)

borders_nm = 250

x_min, x_max = x_beg - borders_nm, x_end + borders_nm
y_min, y_max = y_beg - borders_nm, y_end + borders_nm

n_electrons_required = emf.get_n_electrons_2D(1e-4, lx, ly, borders_nm)
n_electrons = 0


##########
bond_dict_sc = {"C-C2": 4}
#bond_dict_sc = {"C-C2": 4, "C-C'": 2}
#bond_dict_sc = {"C-C3": 6, "C-C'": 2}
#bond_dict_sc = {"C-C2": 4, "C-Cp": 2, "C-C3": 6}
##########


while n_electrons < n_electrons_required:

    mu.pbar(n_electrons, n_electrons_required)
    
    electrons_in_file = 10
    n_files = 100
    
    inds = rnd.choice(len(DATA_PMMA_list), size=n_files, replace=False)
    
    now_DATA_PMMA = np.vstack(list(DATA_PMMA_list[i] for i in inds))
    
    phi=2*np.pi*rnd.random()
    
    emf.rotate_DATA(now_DATA_PMMA, phi)
        
    x_shift, y_shift = rnd.uniform(x_min, x_max), rnd.uniform(y_min, y_max)
    
    emf.add_xy_shift_easy(now_DATA_PMMA, x_shift, y_shift)
    
    scissions = get_scissions(bond_dict_sc, now_DATA_PMMA)
    
    e_matrix_val += np.histogramdd(now_DATA_PMMA[:, 5:8], bins=bins_2nm,
                                   weights=scissions.astype(int))[0]
    
    e_matrix_dE += np.histogramdd(now_DATA_PMMA[:, 5:8], bins=bins_2nm,
                                  weights=now_DATA_PMMA[:, 8])[0]
    
    n_electrons += n_files * electrons_in_file


#%%
np.save('Harris_e_matrix_val_2СС_1.0ester_2020_G.npy', e_matrix_val)
np.save('Harris_e_matrix_dE_2СС_1.0ester_2020_G.npy', e_matrix_dE)

#np.save('Harris_e_matrix_val_3СС3_1.0ester_2020_G_abs.npy', e_matrix_val)
#np.save('Harris_e_matrix_dE_3СС3_1.0ester_2020_G_abs.npy', e_matrix_dE)


#%%
#e_matrix_val = np.load('Harris_e_matrix_val_2CC.npy')
#e_matrix_dE = np.load('Harris_e_matrix_dE_2CC.npy')

e_matrix_val = np.load('Harris_e_matrix_val_2СС_1.0ester_2020_G_abs.npy')
e_matrix_dE = np.load('Harris_e_matrix_dE_2СС_1.0ester_2020_G_abs.npy')


#%%

print(np.sum(e_matrix_val) / np.sum(e_matrix_dE) * 100)


#%%
#print(get_Gs({'C-C2': 4, 'Cp-Cg': 2}))

weights = np.linspace(0, 2, 100)
Gs_array = np.zeros(len(weights))


for i, w in enumerate(weights):
    
    print(i)
    
    Gs_array[i] = get_Gs({'C-C2': 4, 'Cp-Cg': w})
    
    print(Gs_array[i])



#%%
sf.get_Gs_charlesby(27)


#%%
for i in np.arange(0, 2, 0.1):
    
    print(i, get_Gs({'C-C2': 4, 'Cp-Cg': i}))


#%%
weights = np.zeros(6)

for i, t in enumerate((50, 70, 90, 110, 130, 150)):
    
    print(t)
    
    G_req = sf.get_Gs_charlesby(t)
    
    l, r = 0, 2
    
    w = 1
    
    G_now = 0
    
    print('required:', G_req)
    
    while np.abs(G_now - G_req) > 0.1:
        
        G_now = get_Gs({'C-C2': 4, 'Cp-Cg': w})
        
        print('now:', G_now)
        
        if G_now < G_req:
            l = w
            w = (w + r) / 2
            r = r
        
        else:
            r = w
            w = (w + l) / 2
            l = l
    
    
    weights[i] = w
    
    print(w)

 
#%%
tt = np.array((40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150))
ww = np.array((0.1,0.2,0.4,0.5,0.6,0.8,0.9,1.05,1.2,1.4,1.45,1.65))

plt.plot(tt, ww)

plt.xlabel('T, C')
plt.ylabel('weight of ester group bond')

plt.xlim(30, 160)
plt.ylim(0, 2)

plt.grid()



