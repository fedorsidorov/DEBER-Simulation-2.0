import numpy as np
import importlib
import my_functions as mf
import my_variables as mv

mf = importlib.reload(mf)
mv = importlib.reload(mv)

def rewrite_mon_type(chain_inv_matrix, part_matrix, n_chain, n_mon, new_type):
    chain_inv_matrix[n_chain, n_mon, -1] = new_type
    
    Z, XY, x, y, z = chain_inv_matrix[n_chain, n_mon, :-2].astype(int)
    pos = chain_inv_matrix[n_chain, n_mon, -2]
    
    if not np.isnan(pos):
        part_matrix[Z, XY, x, y, z, int(pos), -1] = new_type


def add_ester_group(part_matrix, cell_coords):
    Z, XY, x, y, z = cell_coords
    ester_ind = np.where(np.isnan(part_matrix[Z, XY, x, y, z, :, 0]))[0][0]
    part_matrix[Z, XY, x, y, z, ester_ind, :] = -100
    return 1


def delete_ester_group(part_matrix, cell_coords, ind, d_CO, d_CO2):
    Z, XY, x, y, z = cell_coords
    part_matrix[Z, XY, x, y, z, ind, :] = np.nan
    
    ester_add = -1
    CO_add = 0
    CO2_add = 0
    CH4_add = 1
    
    easter_decay = mf.choice((mv.ester_CO, mv.ester_CO2), p=(d_CO, d_CO2))
    
    if easter_decay == mv.ester_CO:
        CO_add = 1
    else:
        CO2_add = 1
    
    return ester_add, CO_add, CO2_add, CH4_add


def get_part_ind(part_matrix, cell_coords):
    Z, XY, x, y, z = cell_coords
    part_inds = np.where(np.logical_not(np.isnan(part_matrix[Z, XY, x, y, z, :, 0])))[0]
    
    if len(part_inds) == 0:
            return -1
    
    return mf.choice(part_inds)


#def get_ester_decay(d_CO, d_CO2):
#    return mf.choice((mv.ester_CO, mv.ester_CO2), p=(d_CO, d_CO2))


def get_process_3(d1, d2, d3):
    probs = (d1, d2, d3)/np.sum((d1, d2, d3))
    return mf.choice((mv.sci_ester, mv.sci_direct, mv.ester), p=probs)
    

def get_mon_kind():
    return mf.choice([-1, 1])


def get_inv_line(chain_inv_matrix, n_chain, n_mon):
    return chain_inv_matrix[n_chain, n_mon]


def get_n_events(e_matrix, cell_coords):
    Z, XY, x, y, z = cell_coords
    return e_matrix[Z, XY, x, y, z].astype(int)


def get_part_line(part_matrix, cell_coords, part_ind):
    Z, XY, x, y, z = cell_coords
    return part_matrix[Z, XY, x, y, z, part_ind, :]


def mon_type_to_kind(mon_type):
    if mon_type in [-1, 0, 1]:
        return mon_type
    else:
        return mon_type - 10