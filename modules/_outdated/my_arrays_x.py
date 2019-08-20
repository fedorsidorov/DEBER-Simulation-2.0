#%% Import
import numpy as np

path = '/Users/fedor/.yandex.disk/434410540/Yandex.Disk.localized/Study/Simulation/ARRAYS_X/'
atoms = ['H', 'C', 'O', 'Si']

#%% E and theta arrays
E_arr = np.load(path + 'E_arr_x.npy')
theta_arr = np.load(path + 'theta_arr.npy')

#%% Cross sections
# Total
ATOMS_CS = [np.load(path + 'CS/' + atom + '_CS.npy') for atom in atoms]

#%% PMMA atoms cross sections, normed to 1
PMMA_ATOM_CS_SUM_NORM = np.zeros((len(E_arr), 4)) # H*8 + C*5 + O*2

atom_cs_sum = [np.sum(CS) for i in range(len(E_arr))]

#atom_cs_sum = np.sum(H_cs_total*8, axis=1) + np.sum(C_cs_total*5, axis=1) + np.sum(O_cs_total*2, axis=1)
#PMMA_ATOM_CS_SUM_NORM[:, 1] = np.sum(H_cs_total*8, axis=1)/atom_cs_sum
#PMMA_ATOM_CS_SUM_NORM[:, 2] = (np.sum(H_cs_total*8, axis=1) + np.sum(C_cs_total*5, axis=1))/atom_cs_sum
#PMMA_ATOM_CS_SUM_NORM[:, 3] = (np.sum(H_cs_total*8, axis=1) + np.sum(C_cs_total*5, axis=1)\
#       + np.sum(O_cs_total*2, axis=1))/atom_cs_sum
#
## Total with sum normed to 1
#H_cs_total_sum_norm  = np.load(path + 'H/H_CS_SUM_NORM.npy' )
#C_cs_total_sum_norm  = np.load(path + 'C/C_CS_SUM_NORM.npy' )
#O_cs_total_sum_norm  = np.load(path + 'O/O_CS_SUM_NORM.npy' )
#Si_cs_total_sum_norm = np.load(path + 'Si/Si_CS_SUM_NORM.npy')
#CS_TOTAL_SUM_NORM = [H_cs_total_sum_norm, C_cs_total_sum_norm,\
#                     O_cs_total_sum_norm, Si_cs_total_sum_norm]

#%%
# Dirrerntial cross section integral, normed to 1
H_diff_cs_int_norm  = np.load(path + 'H/H_el_diff_cs.npy')
C_diff_cs_int_norm  = np.load(path + 'C/C_el_diff_cs.npy')
O_diff_cs_int_norm  = np.load(path + 'O/O_el_diff_cs.npy')
Si_diff_cs_int_norm = np.load(path + 'Si/Si_el_diff_cs.npy')
DIFF_CS_INT_NORM = [H_diff_cs_int_norm, C_diff_cs_int_norm,\
                    O_diff_cs_int_norm, Si_diff_cs_int_norm]

#%% ENERGY LOSS (WITH SUBSHELLS) - unused
#H_dE_full  = np.load(path + 'dE/H_dE.npy' )
#C_dE_full  = np.load(path + 'dE/C_dE.npy' )
#O_dE_full  = np.load(path + 'dE/O_dE.npy' )
#Si_dE_full = np.load(path + 'dE/Si_dE.npy')
#DE_FULL = [H_dE_full, C_dE_full, O_dE_full, Si_dE_full]

#%% Excitation energy loss
H_exc_dE = np.load(path + 'H/H_exc_dE.npy')
C_exc_dE = np.load(path + 'C/C_exc_dE.npy')
O_exc_dE = np.load(path + 'O/O_exc_dE.npy')
Si_exc_dE = np.load(path + 'Si/Si_exc_dE.npy')

DE_EXC = [H_exc_dE, C_exc_dE, O_exc_dE, Si_exc_dE]

#%% IONIZATION STUFF
# Binding energies
H_ion_Ebind  = np.load(path + 'H/H_ION_EBIND.npy' )
C_ion_Ebind  = np.load(path + 'C/C_ION_EBIND.npy' )
O_ion_Ebind  = np.load(path + 'O/O_ION_EBIND.npy' )
Si_ion_Ebind = np.load(path + 'Si/Si_ION_EBIND.npy')
ION_E_BIND = [H_ion_Ebind, C_ion_Ebind, O_ion_Ebind, Si_ion_Ebind]

# 2nd electron energies
#H_ion_E2nd  = np.load(path + 'E2nd/H_ion_E2nd_files.npy' )
#C_ion_E2nd  = np.load(path + 'E2nd/C_ion_E2nd_files.npy' )
#O_ion_E2nd  = np.load(path + 'E2nd/O_ion_E2nd_files.npy' )
#Si_ion_E2nd = np.load(path + 'E2nd/Si_ion_E2nd_files.npy')
#ION_E_2ND = [H_ion_E2nd, C_ion_E2nd, O_ion_E2nd, Si_ion_E2nd]

# Spectra of secondary electrons (integrals are normed to 1)
H_ion_K_spectra   = np.load(path + 'H/H_K_ion_spectra.npy'  )
C_ion_K_spectra   = np.load(path + 'C/C_K_ion_spectra.npy'  )
C_ion_L1_spectra  = np.load(path + 'C/C_L1_ion_spectra.npy' )
C_ion_L2_spectra  = np.load(path + 'C/C_L2_ion_spectra.npy' )
C_ion_L3_spectra  = np.load(path + 'C/C_L3_ion_spectra.npy' )
O_ion_K_spectra   = np.load(path + 'O/O_K_ion_spectra.npy'  )
O_ion_L1_spectra  = np.load(path + 'O/O_L1_ion_spectra.npy' )
O_ion_L2_spectra  = np.load(path + 'O/O_L2_ion_spectra.npy' )
O_ion_L3_spectra  = np.load(path + 'O/O_L3_ion_spectra.npy' )
Si_ion_K_spectra  = np.load(path + 'Si/Si_K_ion_spectra.npy' )
Si_ion_L1_spectra = np.load(path + 'Si/Si_L1_ion_spectra.npy')
Si_ion_L2_spectra = np.load(path + 'Si/Si_L2_ion_spectra.npy')
Si_ion_L3_spectra = np.load(path + 'Si/Si_L3_ion_spectra.npy')
Si_ion_M1_spectra = np.load(path + 'Si/Si_M1_ion_spectra.npy')
Si_ion_M2_spectra = np.load(path + 'Si/Si_M2_ion_spectra.npy')
Si_ion_M3_spectra = np.load(path + 'Si/Si_M3_ion_spectra.npy')

H_ion_spectra  = [H_ion_K_spectra]
C_ion_spectra  = [C_ion_K_spectra, C_ion_L1_spectra, C_ion_L2_spectra, C_ion_L3_spectra]
O_ion_spectra  = [O_ion_K_spectra, O_ion_L1_spectra, O_ion_L2_spectra, O_ion_L3_spectra]
Si_ion_spectra = [Si_ion_K_spectra, Si_ion_L1_spectra, Si_ion_L2_spectra, Si_ion_L3_spectra,\
                  Si_ion_M1_spectra, Si_ion_M2_spectra, Si_ion_M3_spectra]
ION_SPECTRA    = [H_ion_spectra, C_ion_spectra, O_ion_spectra, Si_ion_spectra]

#%% Arrays to obtain 2nd electron energy
H_ion_K_E_ext   = np.load(path + 'H/H_K_Eext.npy'  )
H_ion_E_ext = [H_ion_K_E_ext]

C_ion_K_E_ext   = np.load(path + 'C/C_K_Eext.npy'  )
C_ion_L1_E_ext  = np.load(path + 'C/C_L1_Eext.npy' )
C_ion_L2_E_ext  = np.load(path + 'C/C_L2_Eext.npy' )
C_ion_L3_E_ext  = np.load(path + 'C/C_L3_Eext.npy' )
C_ion_E_ext = [C_ion_K_E_ext, C_ion_L1_E_ext, C_ion_L2_E_ext, C_ion_L3_E_ext]

O_ion_K_E_ext   = np.load(path + 'O/O_K_Eext.npy'  )
O_ion_L1_E_ext  = np.load(path + 'O/O_L1_Eext.npy' )
O_ion_L2_E_ext  = np.load(path + 'O/O_L2_Eext.npy' )
O_ion_L3_E_ext  = np.load(path + 'O/O_L3_Eext.npy' )
O_ion_E_ext = [O_ion_K_E_ext, O_ion_L1_E_ext, O_ion_L2_E_ext, O_ion_L3_E_ext]

Si_ion_K_E_ext  = np.load(path + 'Si/Si_K_Eext.npy' )
Si_ion_L1_E_ext = np.load(path + 'Si/Si_L1_Eext.npy')
Si_ion_L2_E_ext = np.load(path + 'Si/Si_L2_Eext.npy')
Si_ion_L3_E_ext = np.load(path + 'Si/Si_L3_Eext.npy')
Si_ion_M1_E_ext = np.load(path + 'Si/Si_M1_Eext.npy')
Si_ion_M2_E_ext = np.load(path + 'Si/Si_M2_Eext.npy')
Si_ion_M3_E_ext = np.load(path + 'Si/Si_M3_Eext.npy')
Si_ion_E_ext = [Si_ion_K_E_ext, Si_ion_L1_E_ext, Si_ion_L2_E_ext, Si_ion_L3_E_ext,\
           Si_ion_M1_E_ext, Si_ion_M2_E_ext, Si_ion_M3_E_ext]

ION_E_EXT = [H_ion_E_ext, C_ion_E_ext, O_ion_E_ext, Si_ion_E_ext]
