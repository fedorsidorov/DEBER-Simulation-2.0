#%% Import
import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import copy

import numpy.random as rnd

import my_constants as mc
import my_utilities as mu
import my_arrays_Dapor as ma

mc = importlib.reload(mc)
mu = importlib.reload(mu)
ma = importlib.reload(ma)

os.chdir(mc.sim_folder + 'e-matrix_Harris')

import e_matrix_functions as emf
emf = importlib.reload(emf)

import scission_functions as sf
sf = importlib.reload(sf)


#%%
path = '../e_DATA/Harris/'

folders = ['e_DATA_Harris_MY_PRE', 'e_DATA_Harris_MY_MAC', 'e_DATA_Harris_MY_FTIAN']


#%%
DATA_PMMA_list = []
DATA_PMMA_val_list = []
DATA_PMMA_dE_list = []
#DATA_PMMA_dE_total_list = []


for now_ind in range(len(folders)):
    
    mu.pbar(now_ind, len(folders))
    
    now_folder = path + folders[now_ind]
    now_folder_files = os.listdir(now_folder)
    
    pos = 0
    
    for file in now_folder_files:
        
        if 'DS' in file:
            continue
        
        now_DATA_PMMA = np.load(now_folder + '/' + file)

        now_DATA_PMMA[:, 5:8] *= 1e+7
        
        DATA_PMMA_list.append(now_DATA_PMMA)
        
        now_DATA_PMMA_val = now_DATA_PMMA[np.where(now_DATA_PMMA[:, 3] == 1)]
        DATA_PMMA_val_list.append(now_DATA_PMMA_val)
        
        now_DATA_PMMA_dE = copy.deepcopy(now_DATA_PMMA_val)
        now_DATA_PMMA_dE[np.where(now_DATA_PMMA_dE[:, 3] == 1)[0], -1] = ma.PMMA_E_bind
        DATA_PMMA_dE_list.append(now_DATA_PMMA_dE)


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
def get_Gs(b_map_sc):

    e_matrix_shape = (len(x_bins_2nm)-1, len(y_bins_2nm)-1, len(z_bins_2nm)-1)
    
    e_matrix_val = np.zeros(e_matrix_shape)
    e_matrix_dE = np.zeros(e_matrix_shape)
    
    borders_nm = 250
    
    x_min, x_max = x_beg - borders_nm, x_end + borders_nm
    y_min, y_max = y_beg - borders_nm, y_end + borders_nm
    
    n_electrons_required = emf.get_n_electrons_2D(1e-4, lx, ly, borders_nm) / 50
    n_electrons = 0
    
    
    while n_electrons < n_electrons_required:
    
        mu.pbar(n_electrons, n_electrons_required)
        
        electrons_in_file = 10
        n_files = 100
        
        inds = rnd.choice(len(DATA_PMMA_dE_list), size=n_files, replace=False)
        
        now_DATA_PMMA_val = np.vstack(list(DATA_PMMA_val_list[i] for i in inds))
        now_DATA_PMMA_dE = np.vstack(list(DATA_PMMA_dE_list[i] for i in inds))
        
        phi=2*np.pi*rnd.random()
        
        emf.rotate_DATA(now_DATA_PMMA_val, phi)
        emf.rotate_DATA(now_DATA_PMMA_dE, phi)
            
        x_shift, y_shift = rnd.uniform(x_min, x_max), rnd.uniform(y_min, y_max)
        
        emf.add_xy_shift_easy(now_DATA_PMMA_val, x_shift, y_shift)
        emf.add_xy_shift_easy(now_DATA_PMMA_dE, x_shift, y_shift)
        
        now_EE = now_DATA_PMMA_val[:, 4]
        
        scission_probs = sf.get_stairway(b_map_sc, now_EE)
        
        scissions = rnd.rand(len(now_EE)) < scission_probs
        
        e_matrix_val += np.histogramdd(now_DATA_PMMA_val[:, 5:8], bins=bins_2nm,
                                       weights=scissions.astype(int))[0]
        
        e_matrix_dE += np.histogramdd(now_DATA_PMMA_dE[:, 5:8], bins=bins_2nm,
                                      weights=now_DATA_PMMA_dE[:, -1])[0]
        
        n_electrons += n_files * electrons_in_file
    
    
    return np.sum(e_matrix_val) / np.sum(e_matrix_dE) * 100


#%%
#b_map_sc = {'C-C2': 4}
b_map_sc = {'C-C2': 4, 'Cp-Cg': 2}
#b_map_sc = {'Cp-Cg': 2}
#b_map_sc = {'Op-Cp': 2}


#%%
end_ind = 300

st = sf.get_stairway(b_map_sc, mc.EE[:end_ind])

plt.plot(mc.EE[:end_ind], st)


plt.title('Scission probability')
plt.xlabel('E, eV')
plt.ylabel('scission probability')

plt.grid()


#%%
#print(get_Gs({'C-C2': 4, 'Cp-Cg': 2}))

weights = np.linspace(0, 2, 50)
Gs_array = np.zeros(len(weights))


for i, w in enumerate(weights):
    
    print(i)
    
    Gs_array[i] = get_Gs({'C-C2': 4, 'Cp-Cg': w})
    
    print(Gs_array[i])



#%%
sf.get_Gs_charlesby(160)


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
#[####################] 98% 0.0 2.00889548778
#[####################] 98% 0.1 2.06030251362
#[####################] 98% 0.2 2.13327518484
#[####################] 98% 0.3 2.18026447785
#[####################] 98% 0.4 2.22242803604
#[####################] 98% 0.5 2.28553205258
#[####################] 98% 0.6 2.34271027983
#[####################] 98% 0.7 2.38390040788
#[####################] 98% 0.8 2.42284013117
#[####################] 98% 0.9 2.47711488711
#[####################] 98% 1.0 2.52707233404
#[####################] 98% 1.1 2.6167299456
#[####################] 98% 1.2 2.62121341755
#[####################] 98% 1.3 2.68047397339
#[####################] 98% 1.4 2.71137868791
#[####################] 98% 1.5 2.79633581182
#[####################] 98% 1.6 2.79653544837
#[####################] 98% 1.7 2.88582773421
#[####################] 98% 1.8 2.96605388624
#[####################] 98% 1.9 2.98760937536
 
 
#%%
tt = np.array((40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150))
ww = np.array((0.1,0.2,0.4,0.5,0.6,0.8,0.9,1.05,1.2,1.4,1.45,1.65))

plt.plot(tt, ww)

plt.xlabel('T, C')
plt.ylabel('weight of ester group bond')

plt.xlim(30, 160)
plt.ylim(0, 2)

plt.grid()



