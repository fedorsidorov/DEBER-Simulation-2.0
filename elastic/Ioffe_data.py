#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import my_constants as mc
import E_loss_functions as elf
import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)
mc = importlib.reload(mc)
elf = importlib.reload(elf)

os.chdir(mv.sim_path_MAC + 'elastic')

#%%
def get_Ioffe_raw_data(el):
    
    text_file = open('DATA/' + el + '.txt', 'r')
    lines = text_file.readlines()
    
    theta_corr = list(map(lambda s: '0'+s, lines[12].split()[2:]))
    
    theta_arr_deg_final = np.array(list(map(float, theta_corr)))
    
    E_arr_list = []
    diff_cs_list = []
    total_cs_list = []
    
    for line in lines[14:]:
        
        cs_line = line[1:]
        
        now_diff_cs, now_total_cs = cs_line.split(sep='|')
        
        now_e_diff_cs_corr = list(map(lambda s: '0'+s, now_diff_cs.split()))
        now_total_cs_corr = float('0' + now_total_cs.strip())
        
        E_arr_list.append(now_e_diff_cs_corr[0])
        diff_cs_list.append(now_e_diff_cs_corr[1:])
        total_cs_list.append(now_total_cs_corr)
    
    
    diff_cs = np.zeros((len(diff_cs_list), len(diff_cs_list[0])))
    
    for i in range(len(diff_cs_list)):
        
        diff_cs[i, :] = np.array(list(map(float, diff_cs_list[i])))
        
    
    total_cs = np.array(list(map(float, total_cs_list)))
    
    E_arr = np.array(list(map(float, E_arr_list)))
    
    E_arr_final = E_arr[::-1]
    diff_cs_final = diff_cs[::-1]
    total_cs_final = total_cs[::-1]

    return E_arr_final, theta_arr_deg_final, diff_cs_final, total_cs_final


#%%
def get_Ioffe_final_data(el, EE=mv.EE, THETA_deg=mv.THETA_deg):

    EE = mv.EE
    
    EE_raw, THETA_deg_raw, DIFF_CS_raw, TOTAL_CS_raw =\
        get_Ioffe_raw_data(el)
    
    EE_raw = np.concatenate((np.array([1]), EE_raw))
    DIFF_CS_raw = np.concatenate((DIFF_CS_raw[:1, :], DIFF_CS_raw), axis=0)
    TOTAL_CS_raw = np.concatenate(((TOTAL_CS_raw[:1]), TOTAL_CS_raw))
    
    DIFF_CS_pre = np.zeros((len(EE), len(THETA_deg_raw)))
    
    for i in range(len(THETA_deg_raw)):
        
        DIFF_CS_pre[:, i] = mf.log_interp1d(EE_raw, DIFF_CS_raw[:, i])(EE)
    
    
#    THETA_deg = np.linspace(0.1, 180, 1000)
    DIFF_CS = np.zeros((len(EE), len(THETA_deg)))
    
    for i in range(len(EE)):
        
        DIFF_CS[i, :] = mf.log_interp1d(THETA_deg_raw, DIFF_CS_pre[i, :])(THETA_deg)
    
    
    TOTAL_CS = mf.log_interp1d(EE_raw, TOTAL_CS_raw)(EE)
    
    THETA = np.deg2rad(THETA_deg)

    return EE, THETA, DIFF_CS, TOTAL_CS


#%%
el = 'Si'

EE_raw, THETA_deg_raw, DIFF_CS_raw, TOTAL_CS_raw = get_Ioffe_raw_data(el)
EE, THETA, DIFF_CS, TOTAL_CS = get_Ioffe_final_data(el)

plt.loglog(EE_raw, TOTAL_CS_raw, 'ro', label='raw')
plt.loglog(EE, TOTAL_CS, label='final')

#%%
## 1keV: 19, 681

plt.semilogy(THETA_deg_raw, DIFF_CS_raw[19, :], 'ro', label='raw')
plt.semilogy(np.rad2deg(THETA), DIFF_CS[681, :], label='final')


#%%
## 10keV: 28, 908

plt.semilogy(THETA_deg_raw, DIFF_CS_raw[28, :], 'ro', label='raw')
plt.semilogy(np.rad2deg(THETA), DIFF_CS[908, :], label='final')


#%%
_, _, H_DIFF_CS, H_TOTAL_CS = get_Ioffe_final_data('H')
_, _, C_DIFF_CS, C_TOTAL_CS = get_Ioffe_final_data('C')
_, _, O_DIFF_CS, O_TOTAL_CS = get_Ioffe_final_data('O')
_, _, Si_DIFF_CS, Si_TOTAL_CS = get_Ioffe_final_data('Si')

PMMA_DIFF_CS = elf.mc.n_C_PMMA*C_DIFF_CS + elf.mc.n_H_PMMA*H_DIFF_CS + elf.mc.n_O_PMMA*O_DIFF_CS
PMMA_INT_CS = elf.diff2int(PMMA_DIFF_CS, mv.EE, mv.THETA)


#%%
PMMA_TOTAL_CS = elf.mc.n_C_PMMA*C_TOTAL_CS + elf.mc.n_H_PMMA*H_TOTAL_CS + elf.mc.n_O_PMMA*O_TOTAL_CS

#%%
plt.loglog(EE, Si_TOTAL_CS)


#%%
Si_INT_CS = elf.diff2int(Si_DIFF_CS, mv.EE, mv.THETA)


#%%












