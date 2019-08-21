import numpy as np
from random import random
from random import randint
from numpy.random import choice
from numpy.random import randint
from random import uniform
import sys
from scipy import interpolate

import importlib

import my_arrays_25keV as ma
ma = importlib.reload(ma)

import my_variables as mv
mv = importlib.reload(mv)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from math import gamma

#%% Non-simulation functions
def upd_progress_bar(progress, total):
    barLength, status = 20, ''
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, '\r\n'
    block = int(round(barLength * progress))
    text = '\r[{}] {:.0f}% {}'.format(
        '#' * block + '-' * (barLength - block), round(progress * 100, 0),\
        status)
    sys.stdout.write(text)
    sys.stdout.flush()

def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

#%% Beam functions
def get_norm_density(x, mu, sigma):
    y = 1/(sigma*np.sqrt(2*np.pi))*np.exp((-1)*(x - mu)**2/(2*sigma**2))
    return y

def get_x_y_round_beam(x0, y0, R):
    rho = random()*R
    phi = 2*np.pi*random()
    x = x0 + rho*np.cos(phi)
    y = y0 + rho*np.sin(phi)
    return x, y

def get_x_y_square_beam(x0, y0, D):
    x = x0 + D*random()
    y = y0 + D*random()
    return x, y

#%% Simulation functions
def get_closest_el_ind(array, val):
    ind = np.argmin(np.abs(array - val))
    return ind
   
def get_closest_norm_int_ind(array, value):
    if value > array.max():
        print('Out of range!')
    
    for i in range(len(array) - 1):           
        if array[i] <= value <= array[i + 1]:
            return i
            break
    return len(array)


def get_atom_id(d_PMMA, E_ind, z): ## no negative z values are assumed
    
    if z >= d_PMMA: ## current material is Si
        return 3
    
    elif z >= 0: ## current material is PMMA
        cs_sum_norm_array = ma.PMMA_ATOMS_CS_SUM_NORM[int(E_ind), :]
        return get_closest_norm_int_ind(cs_sum_norm_array, random())


def get_collision_id(atom_id, E_ind):
    
    coll_array = ma.ATOMS_CS_SUM_NORM[atom_id][E_ind, :]
    return get_closest_norm_int_ind(coll_array, random())


def get_On(atom_id, coll_id, E_ind, O_prev):
    
    if coll_id == 0: ## elastic scattering
        theta_rand_arr = ma.ATOMS_DIFF_CS_INT_NORM[atom_id][E_ind, :]
        theta_ind = get_closest_norm_int_ind(theta_rand_arr, random())
        On = get_O_matrix(2*np.pi*random(), ma.theta_arr[theta_ind], O_prev)
        return On
    
    else: ## inelastic scattering, in case of ionization, deal with it later
        return O_prev


def get_O_matrix(phi, theta, O_prev):
    
    Wn = np.mat([[      np.cos(phi),                np.sin(phi),             0],\
        [-np.sin(phi)*np.cos(theta),  np.cos(phi)*np.cos(theta), np.sin(theta)],\
        [ np.sin(phi)*np.sin(theta), -np.cos(phi)*np.sin(theta), np.cos(theta)]])
    
    return Wn*O_prev


def get_dxdydz(atom_id, E_ind, d_PMMA, On, z):
    
    ## get mean free path in current material
    mfp_now = 1/(mv.CONC[atom_id]*np.sum(ma.ATOMS_CS[atom_id][E_ind, :])) * 1e+7
    
    ## the random number for the determination of free path
    r = random()
    ds = -mfp_now * np.log(r)
    dxdydz = On.transpose() * np.mat([[0], [0], [1]]) * ds
    dz = dxdydz[2]
    
    ## deal with PMMA/Si interfce transition
    if (z - d_PMMA) * (z + dz - d_PMMA) < 0:
        
        ## determine new atom id
        atom_id_new = get_atom_id(d_PMMA, E_ind, z + dz)
        ## determine new mean freee path
        mfp_new = 1/(mv.CONC[atom_id]*np.sum(ma.ATOMS_CS[atom_id_new][E_ind, :])) * 1e+7
        ## following Han, determine new ds
        ds1 = ds * (d_PMMA - z) / np.linalg.norm(dxdydz)
        ds = -mfp_new * (np.log(r) + ds1/mfp_now) + ds1
        
        dxdydz = On.transpose() * np.mat([[0], [0], [1]]) * ds
    
    return dxdydz

def get_final_On_and_O2nd(E_prev, E2nd, On):
    phi_ion = 2*np.pi*random()
    omega, t = E2nd/E_prev, E_prev/mv.m_eV
    alpha = np.arcsin(np.sqrt(2*omega/(2 + t - t*omega)))
    gamma = np.arcsin(np.sqrt(2*(1 - omega)/(2 + t*omega)))
    O_final = get_O_matrix(phi_ion, alpha, On)
    O2nd = get_O_matrix(np.pi + phi_ion, gamma, On)
    return O_final, O2nd

## The function for making next step of simulation
def get_coll_data(d_PMMA, E_prev, O_prev, x, z):
    is2nd = False
    E2nd = 0
    O2nd = O_prev*0
    
    ## get E index
    E_ind = get_closest_el_ind(ma.E_arr, E_prev)
    
    ## get atom id
    atom_id = get_atom_id(d_PMMA, E_ind, z)
    
    ## get collision id
    coll_id = get_collision_id(atom_id, E_ind)
    
    ## get current flight direction matrix
    On = get_On(atom_id, coll_id, E_ind, O_prev)
    
    ## get (dx, dy, dz)
    dxdydz = get_dxdydz(atom_id, E_ind, d_PMMA, On, z)
    
    ## if an electron enters im PMMA, continue moving without changing direction
    if z == 0:# or np.abs(z - d_PMMA) < 1e-10:
#        return (atom_id, coll_id, E_prev, dxdydz.transpose(), O_prev, False, 0, 0, 0)
        return (100, -100, E_prev, dxdydz.transpose(), O_prev, False, 0, 0, 0)
    
    if coll_id == 0: ## elastic scattering
        dE = 0
        E = E_prev
    
    elif coll_id == 1: ## excitation
        dE = ma.ATOMS_EXC_DE[atom_id][E_ind]
        E = E_prev - dE
    
    else: ## ionization
        subshell_id = coll_id - 2
        
        E_bind = ma.ATOMS_ION_E_BIND[atom_id][subshell_id]
        
        spectra_line = ma.ATOMS_ION_SPECTRA[atom_id][subshell_id][E_ind, :]
        E_ext = ma.ATOMS_ION_E_SPECTRA[atom_id][subshell_id]
        
        flag = False
        
        E2nd = np.nan
        
        while not flag:
            
            E2nd = E_ext[get_closest_el_ind(spectra_line, random())]
            
            if E2nd < E_prev - E_bind:
                flag = True
        
        is2nd = True
        On, O2nd = get_final_On_and_O2nd(E_prev, E2nd, On)
            
#        dE = E2nd + ma.ATOMS_ION_E_BIND[atom_id][subshell_id]
        dE = E_bind
        
        E = E_prev - E_bind - E2nd
    
    
    if E_prev - dE < 15:
        dE = E_prev
        E = 0
    
#    if dE > 2e+4:
#        print('atom_id=', atom_id, 'coll_id=', coll_id, 'E=', E, 'E2nd=', E2nd, 'dE=', dE)
        
    return atom_id, coll_id, E, dxdydz.transpose(), On, is2nd, E2nd, O2nd, dE

def get_TASKS_and_sim_data(d_PMMA, TASKS, tr_num, par_num, E0, x0y0z0, O_prev):
    # DATA array structure:
    # track_num | parent_num | aid | pid | E | x | y | z | dE
    E = E0
    sim_data = np.zeros((int(2e+4), 9))*np.nan
    pos = 0
    sim_data[pos, :] = np.hstack((tr_num, par_num, np.nan, np.nan, E0, x0y0z0, np.nan))
    
    while E > 15:
        x = sim_data[pos, 5]
        z = sim_data[pos, 7]
        if z < 0:
            break
        aid, cid, E, dxdydz, On, is2nd, E2nd, O2nd, dE =\
            get_coll_data(d_PMMA, E, O_prev, x, z)
        new_task = []
        if (is2nd):
            new_task = [tr_num, E2nd, sim_data[pos, 5:-1], O2nd]
            TASKS.append(new_task)
        sim_data[pos, 2] = aid
        sim_data[pos, 3] = cid
        sim_data[pos, 8] = dE
        sim_data[pos + 1, :] = np.concatenate(([[tr_num]], [[par_num]], [[np.nan]],\
                 [[np.nan]], [[E]], sim_data[pos, 5:-1] + dxdydz, [[np.nan]]), axis=1)
        O_prev = On
        pos += 1
    
    sim_data = np.delete(sim_data, np.where(np.isnan(sim_data[:, 0])), axis=0)    
    return TASKS, sim_data

def create_TASKS(E0, n_tracks):
    O0 = np.eye(3)
    TASKS = [None]*n_tracks
    for i in range(n_tracks):
        x0, y0 = 0, 0
        coords = np.array(np.hstack((x0, y0, 0)))
        task = [np.nan, E0, coords, O0]
        TASKS[i] = task
    return TASKS

def get_DATA(E0, D, d_PMMA, n_tracks):
    n_coords = int(5e+3)
    TASKS = create_TASKS(E0, n_tracks)
    DATA = np.zeros((n_coords*n_tracks, 9))*np.nan
    dataline_pos = 0
    track_num = 0
    
    # create DATA file for TASKS
    while track_num < len(TASKS):
        upd_progress_bar(track_num + 1, len(TASKS))
        task = TASKS[track_num]
        par_num, E0, coords, O0 = task[0], task[1], task[2], task[3]
        TASKS, tr_data = get_TASKS_and_sim_data(d_PMMA, TASKS, track_num,\
                                                par_num, E0, coords, O0)
        DATA[dataline_pos:dataline_pos + len(tr_data), :] = tr_data
        dataline_pos += len(tr_data)
        track_num += 1

    DATA = np.delete(DATA, np.where(np.isnan(DATA[:, 2])), axis=0)
    
    return DATA

#%% Plot DATA
def plot_DATA(DATA, d_PMMA=0, coords=[0, 2]):
    fig, ax = plt.subplots()
    for tn in range(int(np.max(DATA[:, 0]))):
        if len(np.where(DATA[:, 0] == tn)[0]) == 0:
            continue
        beg = np.where(DATA[:, 0] == tn)[0][0]
        end = np.where(DATA[:, 0] == tn)[0][-1] + 1
        ax.plot(DATA[beg:end, 5 + coords[0]], DATA[beg:end, 5 + coords[1]], linewidth=0.7)
        
#        inds_el = beg + np.where(DATA[beg:end, 3] == 0)[0]
#        inds_ion = beg + np.where(DATA[beg:end, 3] >= 2)[0]
#        inds_exc = beg + np.where(DATA[beg:end, 3] == 1)[0]
#        ax.plot(DATA[inds_el, 5], DATA[inds_el, 7], 'r.')
#        ax.plot(DATA[inds_ion, 5], DATA[inds_ion, 7], 'b.')
#        ax.plot(DATA[inds_exc, 5], DATA[inds_exc, 7], 'g.')
    
#    if coords[1] == 2:
#        points = np.arange(-3e+3, 3e+3, 10)
#        ax.plot(points, np.zeros(np.shape(points)), 'k')
#        ax.plot(points, np.ones(np.shape(points))*d_PMMA, 'k')
    
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    plt.gca().set_aspect('equal', adjustable='box')
#    plt.title('Direct Monte-Carlo simulation')
    plt.xlabel('x, nm')
    plt.ylabel('z, nm')
    plt.axis('on')
    plt.grid('on')
    plt.gca().invert_yaxis()
    plt.show()

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










