import numpy as np
import numpy.random as rnd

import importlib

import my_arrays_April as ma
import my_constants as mc
import my_utilities as mu

ma = importlib.reload(ma)
mc = importlib.reload(mc)
mu = importlib.reload(mu)

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


#%% Simulation functions
def get_closest_el_ind(array, val):
    
    return np.argmin(np.abs(array - val))


def get_collision_ind(layer_ind, E_ind):    
    
    values = ma.u_processes[layer_ind][E_ind, :]
    probs = values / np.sum(values)
    
    return rnd.choice(np.arange(len(values), dtype=int), p=probs)


def get_O_matrix(phi, theta, O):
    
    W = np.mat([[               np.cos(phi),                np.sin(phi),             0],
                [-np.sin(phi)*np.cos(theta),  np.cos(phi)*np.cos(theta), np.sin(theta)],
                [ np.sin(phi)*np.sin(theta), -np.cos(phi)*np.sin(theta), np.cos(theta)]])
    
    return np.matmul(W, O)


def get_layer_ind(d_PMMA, z):
    
    if z < 0:
        return -1
    
    elif z <= d_PMMA:
        return 0
    
    return 1


def get_elastic_On(layer_ind, E_ind, O):
    
    phi = 2 * np.pi * rnd.random()
    theta = np.random.choice(mc.THETA_rad, p=ma.sigma_diff_sample_processes[layer_ind][0, E_ind, :])
    
    return get_O_matrix(phi, theta, O)


def get_ee_On_O2nd(E, W, O):
    
    phi = 2 * np.pi * rnd.random()
    theta = np.arcsin(np.sqrt(W/E))
    
    # phi_s = phi + np.pi
    phi_s = np.pi * rnd.random()
    theta_s = np.pi * rnd.random()
    
    On = get_O_matrix(phi, theta, O)
    O2nd = get_O_matrix(phi_s, theta_s, O)
    
    return On, O2nd


def get_ee_W_On_O2nd(layer_ind, proc_ind, E, E_ind, O):
    
    W = rnd.choice(mc.EE, p=ma.sigma_diff_sample_processes[layer_ind][proc_ind, E_ind, :])
    On, O2nd = get_ee_On_O2nd(E, W, O)
        
    return W, On, O2nd


def get_phonon_W_On(E, O):
    
    W = mc.hw_phonon
    
    phi = 2*np.pi*rnd.random()
    
    E_prime = E - W
    B = (E + E_prime + 2*np.sqrt(E*E_prime)) / (E + E_prime - 2*np.sqrt(E*E_prime))
    
    u5 = rnd.random()
    cos_theta = (E + E_prime)/(2*np.sqrt(E*E_prime)) * (1 - B**u5) + B**u5
    
    theta = np.arccos(cos_theta)
    On = get_O_matrix(phi, theta, O)
    
    return W, On


def T_PMMA(E_cos2_theta):
    
    if E_cos2_theta >= mc.Wf_PMMA:
    
        T_PMMA = 4 * np.sqrt(1 - mc.Wf_PMMA/(E_cos2_theta)) /\
            (1 + np.sqrt(1 - mc.Wf_PMMA/(E_cos2_theta)))**2
    
        return T_PMMA
    
    else:
        
        return 0


def get_dxdydz(layer_ind, E_ind, O):
    
    s = -1 / np.sum(ma.u_processes[layer_ind][E_ind, :]) * np.log(rnd.random())
    dxdydz = np.matmul(O.transpose(), np.mat([[0], [0], [1]])) * s
    
    return dxdydz.A1


n_values = 9


def get_TT_and_sim_data(d_PMMA, TT, n_TT, tr_num, par_num, E0, x0y0z0, O):
    
    # track_num | parent_track_num | layer_ind | proc_ind | x | y | z | W | E - 9 values
    
    sim_data = np.ones((mc.DATA_tr_len, n_values)) * -100
    
    layer_ind = get_layer_ind(d_PMMA, x0y0z0[-1])
    
    sim_data[0, :] = np.array((tr_num, par_num, layer_ind, -1, *x0y0z0, 0, E0))
    
    E = E0
    pos = 0
    
    
    while E > 0:
        
        E_ind = get_closest_el_ind(mc.EE, E)
        layer_ind = get_layer_ind(d_PMMA, sim_data[pos, 6])
        
        
        if layer_ind == -1:
            break
        
        if pos+1 >= len(sim_data):
            sim_data = np.vstack((sim_data, np.zeros((mc.DATA_tr_len, n_values)) * -100))
            print('Add sim_data len')
        
        
        if E < ma.E_cut[layer_ind]:
            
            proc_ind = -5
            W, On = E, O*0
            E -= W
        
        else:
        
            proc_ind = get_collision_ind(layer_ind, E_ind)
            
            
            if proc_ind == 0: ## elastic scattering
            
                On = get_elastic_On(layer_ind, E_ind, O)
                W = 0
                E -= W
        
        
            elif layer_ind == 0: ## PMMA
                
                if proc_ind == 1: ## electron-electron interaction
                
                    W, On, O2nd = get_ee_W_On_O2nd(layer_ind, proc_ind, E, E_ind, O)
                    E -= W
            
                elif proc_ind == 2: ## PMMA phonons
                
                    W, On = get_phonon_W_On(mc.EE[E_ind], O)
                    E -= W
                
                else: ## PMMA polarons (proc_ind == 3)
                    
                    W, On = E, O*0
                    E -= W
            
            
            elif layer_ind == 1: ## Si
                
                if proc_ind == 1: ## plasmon
                    
                    _, On, O2nd = get_ee_W_On_O2nd(layer_ind, proc_ind, E, E_ind, O)
                    W = ma.E_bind[0]
                    E -= W
                
                else:
                    
                    W, On, O2nd = get_ee_W_On_O2nd(layer_ind, proc_ind, E, E_ind, O)
                    W += ma.E_bind[proc_ind-1]
                    E -= W
            
            
            else:
                print('WTF with layer_ind')
                print(E_ind, layer_ind, proc_ind, O)
        
        
        # if 'W' not in locals():
                # print(E_ind, layer_ind, proc_ind, O)
        
        dxdydz = get_dxdydz(layer_ind, E_ind, O)
        
            
        sim_data[pos+1, :4] = tr_num, par_num, layer_ind, proc_ind
        sim_data[pos+1, 4:7] = sim_data[pos, 4:7] + dxdydz
        sim_data[pos+1, 7:] = W, E
        
        O = On
        pos += 1
        
        
        if (layer_ind == 0 and proc_ind == 1) or\
           (layer_ind == 1 and proc_ind >= 1): ## create new task
            
            
            new_task = [tr_num, W, sim_data[pos, 4:7], O2nd]
            
            if n_TT >= len(TT):
             
                TT += [None] * mc.TT_len                
                print('Add TT len')
                
            
            TT[n_TT] = new_task
            n_TT += 1
    
    
    sim_data = np.delete(sim_data, np.where(sim_data[:, 0] == -100), axis=0)
    sim_data[:, 4:7] *= 1e+7
    
    return TT, n_TT, sim_data


def create_TT(E0, n_tracks):
    
    O0 = np.eye(3)
    
    TT = [None] * mc.TT_len * n_tracks
    n_TT = 0
    
    for i in range(n_tracks):
        
        coords = np.zeros(3)
        task = [np.nan, E0, coords, O0]
        
        TT[i] = task
        n_TT += 1
        
    return TT, n_TT


## SUPER VAZHNO
def get_DATA(d_PMMA, E0, n_tracks):
    
    TT, n_TT = create_TT(E0, n_tracks)

    DATA = np.ones((mc.DATA_tr_len*n_tracks, n_values)) * -100
    
    dataline_pos = 0
    track_num = 0
    
    while track_num < n_TT:
        
        task = TT[track_num]
        
        par_num, E0, coords, O0 = task[0], task[1], task[2], task[3]
        
        TT, n_TT, tr_data = get_TT_and_sim_data(d_PMMA, TT, n_TT, track_num, par_num, E0, coords, O0)
        
        if dataline_pos + len(tr_data) >= len(DATA):

            DATA = np.vstack((DATA, np.ones((mc.DATA_tr_len*n_tracks, n_values)) * -100))
            print('Add DATA len')
        
        
        DATA[dataline_pos:dataline_pos + len(tr_data), :] = tr_data
        dataline_pos += len(tr_data)
        
        mu.pbar(track_num + 1, n_TT)
        
        track_num += 1
        

    DATA = np.delete(DATA, np.where(DATA[:, 2] == -100), axis=0)
    
    
    return DATA


#%% Plot DATA
def plot_chain(chain_arr, beg=0, end=-1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(chain_arr[beg:end, 0], chain_arr[beg:end, 1], chain_arr[beg:end, 2])
    plt.show()

#%% Data functions
def l_to_logmw(L_arr):
    return np.log10((L_arr*100))

def print_histogram(array, user_bins=20, is_normed=True, alpha=1, name=None):
    hist, bins = np.histogram(array, bins=user_bins, normed=is_normed)
    width = (bins[1] - bins[0])*0.9
    center = (bins[:-1] + bins[1:])/2
    plt.bar(center, hist, align='center', width=width*alpha, label=name)
    plt.legend()
    plt.show()
    
def print_chain(chain_arr, beg=0, end=-1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(chain_arr[beg:end, 0], chain_arr[beg:end, 1], chain_arr[beg:end, 2])
    plt.show()    

def print_chains_list(chains_list, end=-1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for chain_arr in chains_list:
        ax.plot(chain_arr[:end, 0], chain_arr[:end, 1], chain_arr[:end, 2])
    plt.show()  

def delete_nan_rows(array):
    result_arr = np.delete(array, np.where(np.isnan(array[:, 0])), axis=0)
    return result_arr

