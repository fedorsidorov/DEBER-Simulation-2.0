import numpy as np
import random as rnd

import importlib

import my_arrays_osc as ma
import my_constants as mc
import my_utilities as mu
import scission_functions_2020 as sf

ma = importlib.reload(ma)
mc = importlib.reload(mc)
mu = importlib.reload(mu)
sf = importlib.reload(sf)

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


#%% Simulation functions
def get_closest_el_ind(array, val):
    return np.argmin(np.abs(array - val))


def get_closest_int_el(int_array, source, value):
    
    closest_ind = 0
    
    for i in range(len(int_array) - 1):           
        if int_array[i] <= value <= int_array[i + 1]:
            return source[i]
        
    return source[closest_ind]


def get_layer_ind(d_PMMA, z):
    
    if z <= d_PMMA:
        return 0
    
    return 1


def get_collision_ind(layer_ind, E_ind):
    
    probs = ma.proc_u_norm[layer_ind][E_ind, :]
    inds = list(range(len(probs)))
    
    return np.random.choice(inds, p=probs)


def get_PMMA_val_bond_ind(E_ind):
    
    if np.sum(ma.scission_probs[E_ind]) == 0:
        return -8
    
    return np.random.choice(sf.bond_inds, p=ma.scission_probs[E_ind, :])


    
def get_O_matrix(phi, theta, O_prev):
    
    Wn = np.mat([
                [               np.cos(phi),                np.sin(phi),             0],
                [-np.sin(phi)*np.cos(theta),  np.cos(phi)*np.cos(theta), np.sin(theta)],
                [ np.sin(phi)*np.sin(theta), -np.cos(phi)*np.sin(theta), np.cos(theta)]
                ])
    On = np.matmul(Wn, O_prev)
    
    return On


def get_elastic_On(layer_ind, E_ind, O_prev):
    
    phi = 2 * np.pi * rnd.random()
    
    int_array = ma.proc_tau_cumulated[layer_ind][0, E_ind, :]
    theta = get_closest_int_el(int_array, mc.THETA, rnd.random())
    
    return get_O_matrix(phi, theta, O_prev)


def get_ion_On_O2nd(E, hw, Eb, O_prev):
    
    dE = hw - Eb
    
    phi_p = 2 * np.pi * rnd.random()
    phi_s = phi_p + np.pi
    
    sin2theta_p = dE/E * 1/(1 + (1-dE/E)*E/(2 * mc.m_eV))
    sin2theta_s = (1-dE/E) * 1/(1/(dE/(2 * mc.m_eV)))
    
    theta_p = np.arcsin(np.sqrt(sin2theta_p))
    theta_s = np.arcsin(np.sqrt(sin2theta_s))
    
    On = get_O_matrix(phi_p, theta_p, O_prev)
    O2nd = get_O_matrix(phi_s, theta_s, O_prev)
    
    return On, O2nd


def get_binary_On(E, hw, O_prev):
    
    phi_p = 2 * np.pi * rnd.random()
    phi_s = phi_p + np.pi
    
    sin2theta_p = hw/E
    theta_p = np.arcsin(np.sqrt(sin2theta_p))
    
    theta_s = np.arcsin(np.cos(theta_p))
    
    On = get_O_matrix(phi_p, theta_p, O_prev)
    O2nd = get_O_matrix(phi_s, theta_s, O_prev)
    
    return On, O2nd


def get_PMMA_phonon_dE_On(E, O_prev):
    
    dE = mc.hw_phonon
        
    phi = 2*np.pi*rnd.random()
    
    E_prime = E - dE
    
    if E_prime < 0:
        print(E, E_prime)
    
    B = (E + E_prime + 2*np.sqrt(E*E_prime)) /\
        (E + E_prime - 2*np.sqrt(E*E_prime))
    u5 = rnd.random()
    
    cos_theta = (E + E_prime)/(2*np.sqrt(E*E_prime)) * (1 - B**u5) + B**u5
    
    if np.abs(cos_theta) > 1:
        print(E, E_prime, cos_theta)
    
    theta = np.arccos(cos_theta)
    
    On = get_O_matrix(phi, theta, O_prev)
    
    return dE, On


def get_ion_dE_E2nd_On_O2nd(layer_ind, proc_ind, E, E_ind, O_prev):
    
    if layer_ind == 0 and proc_ind == 2: ## valence electrons, but NO bond contribution
        int_array = ma.proc_tau_cumulated[layer_ind][2, E_ind, :]
        hw = get_closest_int_el(int_array, mc.EE, rnd.random())
        On, _ = get_binary_On(E, hw, O_prev)
        return hw, 0, On, O_prev*0
        
    elif layer_ind == 0 and proc_ind >= 10: ## valence electrons, bond contribution
        
        bond_ind = proc_ind - 10
        
        Eb = ma.PMMA_val_E_bind[bond_ind]
        int_array = ma.proc_tau_cumulated[layer_ind][2, E_ind, :]
        hw = get_closest_int_el(int_array, mc.EE, rnd.random())
        
        if hw > Eb:
            E2nd = hw - Eb
            On, O2nd = get_ion_On_O2nd(E, hw, Eb, O_prev)
            return hw, E2nd, On, O2nd
        else:
            On, _ = get_binary_On(E, hw, O_prev)
            return hw, 0, On, O_prev*0
        
    else: ## ordinary ionization event
        Eb = ma.E_bind[layer_ind][proc_ind]
        int_array = ma.proc_tau_cumulated[layer_ind][proc_ind, E_ind, :]
        hw = get_closest_int_el(int_array, mc.EE, rnd.random())
        
        if hw > Eb:
            E2nd = hw - Eb
            On, O2nd = get_ion_On_O2nd(E, hw, Eb, O_prev)
            return hw, E2nd, On, O2nd
        else:
            On, _ = get_binary_On(E, hw, O_prev)
            return hw, 0, On, O_prev*0


def get_dE_E2nd_On_O2nd(layer_ind, proc_ind, E, E_ind, O_prev):
    
    if E < ma.E_cut[layer_ind]: ## cutoff
        return E, 0, O_prev*0, O_prev*0
    
    if proc_ind == 0: ## elastic scattering
        On = get_elastic_On(layer_ind, E_ind, O_prev)
        return 0, 0, On, On*0
    
    elif proc_ind == 1: ## plasmon, no 2nd!
        hw = ma.E_bind[layer_ind][proc_ind]
        On, _ = get_binary_On(E, hw, O_prev)
        return hw, 0, On, O_prev*0
    
    elif (layer_ind == 0 and proc_ind not in [0, 5, 6]) or\
         (layer_ind == 1 and proc_ind != 0): ## ionization
        
        return get_ion_dE_E2nd_On_O2nd(layer_ind, proc_ind, E, E_ind, O_prev)
    
    elif layer_ind == 0 and proc_ind == 5: ## PMMA phonons
        dE, On = get_PMMA_phonon_dE_On(E, O_prev)
        return dE, 0, On, O_prev*0
        
    elif layer_ind == 0 and proc_ind == 6: ## PMMA polarons
        return E, 0, O_prev*0, O_prev*0


def get_dxdydz(layer_ind, E_ind, d_PMMA, z, On):
    
    now_u = np.sum(ma.proc_u[layer_ind][E_ind, :])
    now_mfp = 1 / now_u
    
    R = rnd.random()
    s = -now_mfp * np.log(R)
    
    dxdydz = np.matmul(On.transpose(), np.mat([[0], [0], [1]])) * s
    dz = dxdydz[2]
    
    ## han2002.pdf
    if (z - d_PMMA) * (z + dz - d_PMMA) < 0:
        
        new_layer_ind = get_layer_ind(d_PMMA, z + dz)
        new_u = np.sum(ma.proc_u[new_layer_ind][E_ind, :])
        
        l1 = now_mfp
        l2 = 1 / new_u

        s1 = s * np.abs(d_PMMA - z) / np.linalg.norm(dxdydz)
        s = s1 + l2*(-np.log(R) - s1/l1)
        
        dxdydz = np.matmul(On.transpose(), np.mat([[0], [0], [1]])) * s
        
        return dxdydz
    
    return dxdydz


def get_coll_data(d_PMMA, E_prev, O_prev, x, z):
    
    ## x - in future!!
    
    E_ind = get_closest_el_ind(mc.EE, E_prev)
    layer_ind = get_layer_ind(d_PMMA, z)
    proc_ind = get_collision_ind(layer_ind, E_ind)
    
    if layer_ind == 0 and proc_ind == 2:
        proc_ind = 10 + get_PMMA_val_bond_ind(E_ind)
    
    dE, E2nd, On, O2nd = get_dE_E2nd_On_O2nd(layer_ind, proc_ind, E_prev, E_ind, O_prev)
    
    dxdydz = get_dxdydz(layer_ind, E_ind, d_PMMA, z, On)
            
    return layer_ind, proc_ind, E_prev-dE, dxdydz.transpose(), dE, E2nd, On, O2nd


def get_TT_and_sim_data(TT, n_TT, d_PMMA, tr_num, par_num, E0, x0y0z0, O_prev, z_cut_Si):
    
    # DATA array structure:
    # track_num | parent_track_num | atom_ind | proc_ind | E | x | y | z | Edep | E2nd
    
    E = E0

    sim_data = np.ones((mc.DATA_tr_len, 10)) * -100
    
    pos = 0
    
    sim_data[pos, :] = np.array((tr_num, par_num, np.nan, np.nan, E0, x0y0z0[0],\
            x0y0z0[1], x0y0z0[2], np.nan, np.nan))

    
    while E > 0:
        
        x = sim_data[pos, 5]
        z = sim_data[pos, 7]
        
        if z < 0:
            break
        
        layer_ind, proc_ind, E, dxdydz, dE, E2nd, On, O2nd,  =\
            get_coll_data(d_PMMA, E, O_prev, x, z)
        

        sim_data[pos, 2] = layer_ind
        sim_data[pos, 3] = proc_ind
        sim_data[pos, 8] = dE - E2nd
        sim_data[pos, 9] = E2nd
        
        if E2nd > 0:
            new_task = [tr_num, E2nd, sim_data[pos, 5:-2], O2nd]
            
            if n_TT >= len(TT):
              
                TT += [None] * mc.TT_len                
                print('Add TT len')
                
            
            TT[n_TT] = new_task
            n_TT += 1
        
        
        if pos+1 >= len(sim_data):
            sim_data = np.vstack((sim_data, np.zeros((mc.DATA_tr_len, 10)) * -100))
            print('Add sim_data len')
            print('E =', E)
        
        
        sim_data[pos + 1, :] = np.concatenate(([[tr_num]], [[par_num]], [[np.nan]],\
            [[np.nan]], [[E]], sim_data[pos, 5:-2] + dxdydz, [[np.nan]], [[np.nan]]), axis=1)
        
        O_prev = On
        pos += 1
        
        if z > z_cut_Si:
            break

    
    sim_data = np.delete(sim_data, np.where(sim_data[:, 0] == -100), axis=0)
    
    return TT, n_TT, sim_data


def create_TT(E0, n_tracks):
    
    O0 = np.eye(3)
    
    TT = [None] * mc.TT_len * n_tracks
    n_TT = 0
    
    for i in range(n_tracks):
        
        x0, y0 = 0, 0 ## to improve?
        coords = np.array(np.hstack((x0, y0, 0)))
        task = [np.nan, E0, coords, O0]
        
        TT[i] = task
        n_TT += 1
        
    return TT, n_TT


## SUPER VAZHNO
def get_DATA(d_PMMA, E0, n_tracks, z_cut_Si=100):
    
    TT, n_TT = create_TT(E0, n_tracks)

    DATA = np.ones((mc.DATA_tr_len*n_tracks, 10)) * -100
    
    dataline_pos = 0
    track_num = 0
    
    while track_num < n_TT:
        
        task = TT[track_num]
        
        par_num, E0, coords, O0 = task[0], task[1], task[2], task[3]
        
        TT, n_TT, tr_data = get_TT_and_sim_data(TT, n_TT, d_PMMA, track_num,\
                                                par_num, E0, coords, O0, z_cut_Si)
        
        if dataline_pos + len(tr_data) >= len(DATA):

            DATA = np.vstack((DATA, np.ones((mc.DATA_tr_len*n_tracks, 10)) * -100))
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

