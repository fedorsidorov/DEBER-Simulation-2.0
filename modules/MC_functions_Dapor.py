import numpy as np
import random as rnd

import importlib

import prepare_PMMA_Dapor as ma
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


def get_closest_int_el(int_array, source, value):
    
    closest_ind = 0
    
    for i in range(len(int_array) - 1):           
        if int_array[i] <= value <= int_array[i + 1]:
            closest_ind = i
        
    return source[closest_ind]


def get_MC_ind(values):
    
    inds = list(range(len(values)))
    probs = values / np.sum(values)
    
    return np.random.choice(inds, p=probs)


def get_collision_ind(E_ind):    
    
    return get_MC_ind(ma.u_processes[E_ind, :])

    
def get_O_matrix(phi, theta, O):
    
    W = np.mat([[               np.cos(phi),                np.sin(phi),             0],
                [-np.sin(phi)*np.cos(theta),  np.cos(phi)*np.cos(theta), np.sin(theta)],
                [ np.sin(phi)*np.sin(theta), -np.cos(phi)*np.sin(theta), np.cos(theta)]])
    
    return np.matmul(W, O)


def get_elastic_On(E_ind, O):
    
    phi = 2 * np.pi * rnd.random()
    
    u_diff_cumulated = ma.u_el_diff_cumulated[E_ind, :]
    theta = get_closest_int_el(u_diff_cumulated, mc.THETA, rnd.random())
    
    return get_O_matrix(phi, theta, O)


def get_ee_On_O2nd(E, W, O):
    
    phi = 2 * np.pi * rnd.random()
    phi_s = phi + np.pi
    
    theta = np.arcsin(np.sqrt(W/E))
    theta_s = np.pi * rnd.random()
    
    On = get_O_matrix(phi, theta, O)
    O2nd = get_O_matrix(phi_s, theta_s, O)
    
    return On, O2nd


def get_ee_W_On_O2nd(E_ind, O):
    
    u_diff_cumulated = ma.u_ee_diff_cumulated[E_ind, :]
    W = get_closest_int_el(u_diff_cumulated, mc.EE, rnd.random())
    
    On, O2nd = get_ee_On_O2nd(mc.EE[E_ind], W, O)
        
    return W, On, O2nd


def get_phonon_dE_On(E, O):
    
    dE = mc.hw_phonon
        
    phi = 2*np.pi*rnd.random()
    
    E_prime = E - dE
    B = (E + E_prime + 2*np.sqrt(E*E_prime)) / (E + E_prime - 2*np.sqrt(E*E_prime))
    
    u5 = rnd.random()
    
    cos_theta = (E + E_prime)/(2*np.sqrt(E*E_prime)) * (1 - B**u5) + B**u5
    theta = np.arccos(cos_theta)
    
    On = get_O_matrix(phi, theta, O)
    
    return dE, On


def T_PMMA(E_cos2_theta):
    
    if E_cos2_theta >= mc.Wf_PMMA:
    
        T_PMMA = 4 * np.sqrt(1 - mc.Wf_PMMA/(E_cos2_theta)) /\
            (1 + np.sqrt(1 - mc.Wf_PMMA/(E_cos2_theta)))**2
    
        return T_PMMA
    
    else:
        
        return 0


def get_dxdydz(E_ind, O):
    
    s = -1 / np.sum(ma.u_processes[E_ind, :]) * np.log(rnd.random())
    
    dxdydz = np.matmul(O.transpose(), np.mat([[0], [0], [1]])) * s
    
    return dxdydz.A1


def get_TT_and_sim_data(TT, n_TT, tr_num, par_num, E0, x0y0z0, O):
    
    # track_num | parent_track_num | proc_ind | E | x | y | z | W
    
    sim_data = np.ones((mc.DATA_tr_len, 8)) * -100
    sim_data[0, :] = np.array((tr_num, par_num, -1, E0, *x0y0z0, 0))
    
    E = E0
    pos = 0
    
    while E > mc.Wf_PMMA:
        
        z = sim_data[pos, 6]
        
        if z < 0:
            break
        
        if pos+1 >= len(sim_data):
            sim_data = np.vstack((sim_data, np.zeros((mc.DATA_tr_len, 8)) * -100))
            print('Add sim_data len')
        
        E_ind = get_closest_el_ind(mc.EE, E)
        proc_ind = get_collision_ind(E_ind)
        
        if proc_ind == 0: ## elastic scattering
        
            On = get_elastic_On(E_ind, O)
            W = 0
    
        elif proc_ind == 1: ## electron-electron interaction
        
            W, On, O2nd = get_ee_W_On_O2nd(E_ind, O)
    
        elif proc_ind == 2: ## PMMA phonons
        
            W, On = get_phonon_dE_On(mc.EE[E_ind], O)
        
        else: ## PMMA polarons (proc_ind == 3)
            
            W, On = E, O*0
        
        
        E -= W
        
        dxdydz = get_dxdydz(E_ind, O)
        dz = dxdydz[2]
        
        if z + dz > 0: ## nothing serious happens
            
            sim_data[pos+1, :4] = tr_num, par_num, proc_ind, E            
            sim_data[pos+1, 4:7] = sim_data[pos, 4:7] + dxdydz
            sim_data[pos+1, 7] = W
            
            O = On
            pos += 1
            
        else:
            
            if dz > 0:
                print('dz > 0')
            
            cos_theta = np.dot(dxdydz, [0, 0, -1]) / (np.linalg.norm(dxdydz) * 1)
            
            if cos_theta < 0:
                print('cos_theta < 0')
            
            theta = np.arccos(cos_theta)
            
            if rnd.random() < T_PMMA(E * cos_theta**2): ## electron emerges
                
                sim_data[pos+1, :4] = tr_num, par_num, proc_ind, E
                sim_data[pos+1, 4:7] = sim_data[pos, 4:7] + dxdydz
                sim_data[pos+1, 7] = W
                
                O = On
                pos += 1
            
            else: ## electron is reflected from the interface
                
                scale = z / np.abs(dz)
                
                xyz_reflection = sim_data[pos, 4:7] + dxdydz*scale                
                
#                print('reflection', tr_num)
#                print('reflection coords =', xyz_reflection)
                
                dxdydz_reserved = dxdydz * [1, 1, -1] * (1 - scale)
                
                sim_data[pos+1, :4] = tr_num, par_num, -1, E
                sim_data[pos+1, 4:7] = xyz_reflection
                sim_data[pos+1, 7] = 0
                
                pos += 1                
                ## new position!!!   
                
                final_xyz = xyz_reflection + dxdydz_reserved
                
                if final_xyz[2] < 0:
                    print('final_z < 0')
                
                sim_data[pos+1, :4] = tr_num, par_num, proc_ind, E
                sim_data[pos+1, 4:7] = final_xyz
                sim_data[pos+1, 7] = W
                
#                O = get_O_matrix(0, np.pi - theta*2, On)
                O = On.copy()
                O[:, 2] *= -1
                
#                print(np.matmul(On.transpose(), np.mat(dxdydz.reshape((3, 1)))).A1)
#                print(np.matmul(O.transpose(), np.mat(dxdydz.reshape((3, 1)))).A1)
                
                pos += 1
    
    
        if proc_ind == 1: ## create new task

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
def get_DATA(E0, n_tracks):
    
    TT, n_TT = create_TT(E0, n_tracks)

    DATA = np.ones((mc.DATA_tr_len*n_tracks, 8)) * -100
    
    dataline_pos = 0
    track_num = 0
    
    while track_num < n_TT:
        
        task = TT[track_num]
        
        par_num, E0, coords, O0 = task[0], task[1], task[2], task[3]
        
        TT, n_TT, tr_data = get_TT_and_sim_data(TT, n_TT, track_num, par_num, E0, coords, O0)
        
        if dataline_pos + len(tr_data) >= len(DATA):

            DATA = np.vstack((DATA, np.ones((mc.DATA_tr_len*n_tracks, 8)) * -100))
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
