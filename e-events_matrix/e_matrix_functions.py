import numpy as np
import numpy.random as rnd


#%%
def get_x0y0(lx, ly, space):
    
    return rnd.uniform(-space, lx + space), rnd.uniform(-space, ly + space)


def rotate_DATA(DATA, phi=2*np.pi*rnd.random()):
    
    rot_mat = np.mat([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi),  np.cos(phi)]])
    
    DATA[:, 5:7] = np.dot(rot_mat, DATA[:, 5:7].transpose()).transpose()


def add_xy_shift_easy(DATA, x_shift, y_shift):
    
    DATA[:, 5] += x_shift
    DATA[:, 6] += y_shift


def add_xy_shift(DATA, tr_num, x_shift, y_shift):
    
    inds = np.where(DATA[:, 0] == tr_num)
    
    for i in inds[0]:
        DATA[i, 5] += x_shift
        DATA[i, 6] += y_shift
    
    inds_2nd = np.where(DATA[:, 1] == tr_num)[0]
    
    if len(inds_2nd) == 0:
        return
    
    else:
        tr_nums_2nd = np.unique(DATA[inds_2nd, 0])
        for tr_num_2nd in tr_nums_2nd:
            add_xy_shift(DATA, tr_num_2nd, x_shift, y_shift)
            

def shift_DATA(DATA, x_range, y_range):
    
    n_tr_prim = int(DATA[np.where(np.isnan(DATA[:, 1]))][-1, 0] + 1)
    
    for track_num in range(n_tr_prim):
        x0, y0 = rnd.uniform(*x_range), rnd.uniform(*y_range)
        add_xy_shift(DATA, track_num, x0, y0)


def get_n_electrons(dose_C_cm2, lx_nm, ly_nm, borders_nm):
    
    q_el_C = 1.6e-19
    A_cm2 = (lx_nm + borders_nm*2) * (ly_nm + borders_nm*2) * 1e-14
    Q_C = dose_C_cm2 * A_cm2
    
    return int(np.round(Q_C / q_el_C))

