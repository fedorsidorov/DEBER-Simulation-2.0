import numpy as np
import random as rnd

import importlib

import my_arrays as ma
import my_constants as mc
import my_utilities as mu

ma = importlib.reload(ma)
mc = importlib.reload(mc)
mu = importlib.reload(mu)

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from math import gamma


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
    
    inds = np.arange(len(values))
    probs = values / np.sum(values)
    
    return np.random.choice(inds, p=probs)


def get_layer_ind(d_PMMA, z):
    
    if z <= d_PMMA:
        return 0
    
    return 1


def get_collision_ind(layer_ind, E_ind):
    
    processes_U = ma.processes_U[layer_ind][E_ind, :]
    
    return get_MC_ind(processes_U)
        
    
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
    
    int_array = ma.processes_int_U[layer_ind][0][E_ind, :]
    theta = get_closest_int_el(int_array, ma.THETA, rnd.random())
    
    return get_O_matrix(phi, theta, O_prev)


def get_ion_On_O2nd(E, E_prime, E_bind, O_prev):
    
    phi = 2 * np.pi * rnd.random()
    phi_s = phi + np.pi
    
    p = np.sqrt(E / (2*mc.m))
    p_prime = np.sqrt(E_prime / (2*mc.m))
    
#    cos_theta = (E_prime + E_bind/2) / (np.sqrt(E*E_prime))
#    
#    theta = np.arccos(cos_theta)
#    sin_theta = np.sin(theta)
#    
#    q = np.sqrt(p**2 + p_prime**2 - 2*p*p_prime*cos_theta)
#    
#    sin_theta_s = np.sqrt(p_prime**2 / q**2 * sin_theta**2)
#    theta_s = np.arcsin(sin_theta_s)
    
    cos_theta = p_prime / p
    theta = np.arccos(cos_theta)
    
    theta_s = np.arcsin(cos_theta)
    
    On = get_O_matrix(phi, theta, O_prev)
    O2nd = get_O_matrix(phi_s, theta_s, O_prev)
    
    return On, O2nd


def get_ion_dE_E2nd_On_O2nd(layer_ind, proc_ind, E, E_ind, O_prev):
    
#    print('layer_ind =', layer_ind)
#    print('proc_ind =', proc_ind)
#    print('E_ind =', E_ind)
    
    E_bind = ma.E_bind[layer_ind][proc_ind][E_ind]
    
    int_array = ma.processes_int_U[layer_ind][proc_ind][E_ind, :]
    dE = get_closest_int_el(int_array, ma.EE, rnd.random())
    
    
    if dE > E_bind:
        
        E2nd = dE - E_bind
        On, O2nd = get_ion_On_O2nd(E, E - dE, E_bind, O_prev)
        
        return dE, E2nd, On, O2nd
    
    else:
        return dE, 0, O_prev, O_prev*0


def get_dE_E2nd_On_O2nd(layer_ind, proc_ind, E, E_ind, O_prev):
    
    if proc_ind == 0: ## elastic scattering
        
        On = get_elastic_On(layer_ind, E_ind, O_prev)
        
        return 0, 0, On, On*0
    
    elif (layer_ind == 0 and proc_ind in [1, 2, 3]) or\
         (layer_ind == 1 and proc_ind in [1, 2, 3, 4]): ## ionization
        
        return get_ion_dE_E2nd_On_O2nd(layer_ind, proc_ind, E, E_ind, O_prev)
    
    if proc_ind == 4: ## PMMA phonons
        
        dE = mc.hw_phonon
        
        phi = 2*np.pi*rnd.random()
        
        E_prime = E - dE
        B = (E + E_prime + 2*np.sqrt(E*E_prime)) / (E + E_prime - 2*np.sqrt(E*E_prime))
        u5 = rnd.random()
        
        cos_theta = (E + E_prime)/(2*np.sqrt(E*E_prime)) * (1 - B**u5) + B**u5
        theta = np.arccos(cos_theta)
        
        On = get_O_matrix(phi, theta, O_prev)
        
        return dE, 0, On, On*0
        
    else: ## PMMA polarons
        return E, 0, O_prev*0, O_prev*0


def get_dxdydz(layer_ind, E_ind, d_PMMA, z, On):
    
    now_U = np.sum(ma.processes_U[layer_ind][E_ind, :])
    now_mfp = 1 / now_U
    
    R = rnd.random()
    s = -now_mfp * np.log(R)
    
    dxdydz = np.matmul(On.transpose(), np.mat([[0], [0], [1]])) * s
    dz = dxdydz[2]
    
    ## Han2002.pdf
    if (z - d_PMMA) * (z + dz - d_PMMA) < 0:
        
        new_layer_ind = get_layer_ind(d_PMMA, z + dz)
        new_U = np.sum(ma.processes_U[new_layer_ind][E_ind, :])
        
        l1 = now_mfp
        l2 = 1 / new_U

        s1 = s * np.abs(d_PMMA - z) / np.linalg.norm(dxdydz)
        s = s1 + l2*(-np.log(R) - s1/l1)
        
        dxdydz = np.matmul(On.transpose(), np.mat([[0], [0], [1]])) * s
        
        return dxdydz
    
    return dxdydz


def get_coll_data(d_PMMA, E_prev, O_prev, x, z):
    
    ## x - in future!!
    
    E_ind = get_closest_el_ind(ma.EE, E_prev)
    layer_ind = get_layer_ind(d_PMMA, z)
    proc_ind = get_collision_ind(layer_ind, E_ind)
    
    dE, E2nd, On, O2nd = get_dE_E2nd_On_O2nd(layer_ind, proc_ind, E_prev, E_ind, O_prev)
    
    dxdydz = get_dxdydz(layer_ind, E_ind, d_PMMA, z, On)
            
    return layer_ind, proc_ind, E_prev-dE, dxdydz.transpose(), dE, E2nd, On, O2nd


def get_TT_and_sim_data(TT, n_TT, d_PMMA, tr_num, par_num, E0, x0y0z0, O_prev, z_cut_Si):
    
    # DATA array structure:
    # track_num | parent_track_num | atom_ind | proc_ind | E | x | y | z | dE
    
    E = E0
    
    sim_data = np.zeros((mc.TT_len, 9))*np.nan
    
    pos = 0
    
    sim_data[pos, :] = np.hstack((tr_num, par_num, np.nan, np.nan, E0, x0y0z0, np.nan))
    
    while E > mc.E_cut_PMMA:
        
        x = sim_data[pos, 5]
        z = sim_data[pos, 7]
        
        if z < 0:
            break
        
        layer_ind, proc_ind, E, dxdydz, dE, E2nd, On, O2nd,  =\
            get_coll_data(d_PMMA, E, O_prev, x, z)
        
        ## CUT!
        if z > z_cut_Si:
            break
        
        if layer_ind == 1 and E < 10:
            break
        
        sim_data[pos, 2] = layer_ind
        sim_data[pos, 3] = proc_ind
        sim_data[pos, 8] = dE
        
        if E2nd > 0:
            new_task = [tr_num, E2nd, sim_data[pos, 5:-1], O2nd]
            TT[n_TT] = new_task
            n_TT += 1
            
        else:
            sim_data[pos, 3] = proc_ind*(-1)
        
        
        sim_data[pos + 1, :] = np.concatenate(([[tr_num]], [[par_num]], [[np.nan]],\
                 [[np.nan]], [[E]], sim_data[pos, 5:-1] + dxdydz, [[np.nan]]), axis=1)
        
        O_prev = On
        pos += 1
    
    sim_data = np.delete(sim_data, np.where(np.isnan(sim_data[:, 0])), axis=0)
    
    return TT, n_TT, sim_data


def create_TT(E0, n_tracks):
    
    O0 = np.eye(3)
    TT = [None] * (n_tracks*1000)
    n_TT = 0
    
    for i in range(n_tracks):
        
        x0, y0 = 0, 0 ## to improve?
        coords = np.array(np.hstack((x0, y0, 0)))
        task = [np.nan, E0, coords, O0]
        
        TT[i] = task
        n_TT += 1
        
    return TT, n_TT


def get_DATA(d_PMMA, E0, n_tracks, z_cut_Si):
    
    TT, n_TT = create_TT(E0, n_tracks)
    
    DATA = np.zeros((mc.TT_len*n_tracks, 9))*np.nan
    
    dataline_pos = 0
    track_num = 0
    
    while track_num < n_TT:
        
        task = TT[track_num]
        
        par_num, E0, coords, O0 = task[0], task[1], task[2], task[3]
        
        TT, n_TT, tr_data = get_TT_and_sim_data(TT, n_TT, d_PMMA, track_num,\
                                                par_num, E0, coords, O0, z_cut_Si)
        
        DATA[dataline_pos:dataline_pos + len(tr_data), :] = tr_data
        
        dataline_pos += len(tr_data)
        
        mu.upd_progress_bar(track_num + 1, n_TT)
        
        track_num += 1

    DATA = np.delete(DATA, np.where(np.isnan(DATA[:, 2])), axis=0)
    
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

#def add_xy_rotation(arr, phi):
#    rot_mat = np.mat([[np.cos(phi), -np.sin(phi)],
#                      [np.sin(phi),  np.cos(phi)]])
#    result = np.dot(rot_mat, np.mat(arr).transpose())
#    return np.array(result.transpose())
#
#
#def rotate_DATA(DATA, phi=2*np.pi*random()):
##    DATA[:, 5:7] = add_xy_rotation(DATA[:, 5:7], 2 * np.pi * random())
#    DATA[:, 5:7] = add_xy_rotation(DATA[:, 5:7], phi)
#    return DATA
#
#
#def add_xy_shift(DATA, tr_num, x_shift, y_shift):
#    
#    ## get indices wuth current track
#    inds = np.where(DATA[:, 0] == tr_num)
#    
#    ## make primary shift
#    for i in inds[0]:
#        DATA[i, 5] += x_shift
#        DATA[i, 6] += y_shift
#    
#    ## get indices with 1st gen of 2nd electrons
#    inds_2nd = np.where(DATA[:, 1] == tr_num)[0]
#    
#    ## if no 2ndaries, return DATA
#    if len(inds_2nd) == 0:
#        return
##        return DATA
#    
#    ## else RECURSION!
#    else:
#        ## find tr_nums with 2ndaries as primaries
#        tr_nums_2nd = np.unique(DATA[inds_2nd, 0])
#        
#        ## for every tr_num make recursive call
#        for tr_num_2nd in tr_nums_2nd:
##            DATA = add_xy_shift(DATA, tr_num_2nd, x_shift, y_shift)
#            add_xy_shift(DATA, tr_num_2nd, x_shift, y_shift)
#        
##        return DATA
#
#
#def shift_DATA(DATA, xlim, ylim):
#    n_tr_prim = int(DATA[np.where(np.isnan(DATA[:, 1]))][-1, 0] + 1)
#    for track_num in range(n_tr_prim):
#        
#        x0, y0 = uniform(*xlim), uniform(*ylim)          
#        add_xy_shift(DATA, track_num, x0, y0)
#        
#        '''
#            ## in case of only elastic events in PMMA
#            if len(np.where(DATA[:, 0] == track_num)[0]) == 0:
#                continue
#            ## in normal case
#            else:
#                x0, y0 = uniform(*xlim), uniform(*ylim)          
##                DATA = add_xy_shift(DATA, track_num, x0, y0)
#                add_xy_shift(DATA, track_num, x0, y0)
#        '''
##    return DATA
#
#
def get_n_electrons(dose_C_cm2, square_side_nm):
    q_el_C = 1.6e-19
    A_cm2 = (square_side_nm * 1e-7)**2
    Q_C = dose_C_cm2 * A_cm2
    return int(np.round(Q_C / q_el_C))

#%%
def get_schulz_zimm(Mn, Mw, x):
    
    z = Mn / (Mw - Mn)
    l = 1 / (Mw - Mn)
    
#    f = l**z / gamma(z) * np.power(x, z-1) * np.exp(-l*x)
    f = l**z / (gamma(z) * Mn) * np.power(x, z) * np.exp(-l*x)
    
    return f

