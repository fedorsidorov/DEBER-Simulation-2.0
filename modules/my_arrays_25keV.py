#%% Import
import numpy as np
#import matplotlib.pyplot as plt

path = '/Users/fedor/Documents/DEBER-Simulation/arrays_25keV/'
#path = '/home/fedor/Yandex.Disk/Study/Simulation/arrays_25keV/'

atoms = ['H', 'C', 'O', 'Si']
subshells = {'H': ['K'],
             'C': ['K', 'L1', 'L2', 'L3'],
             'O': ['K', 'L1', 'L2', 'L3'],
             'Si':['K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3']
             }

PMMA_atom_weights = np.array([8, 5, 2])

#%% E and theta arrays
E_arr = np.load(path + 'E_arr_25keV.npy')
theta_arr = np.load(path + 'theta_arr_25keV.npy')

#%% Cross sections
ATOMS_CS = [np.load(path + 'CS/' + atom + '_CS.npy') for atom in atoms]

#%% PMMA atoms cross sections, normed to 1
ATOMS_CS_SUM = np.array([
        np.sum(ATOMS_CS[i], axis=1) for i in range(len(atoms))
        ]).transpose()

PMMA_ATOMS_CS_SUM = ATOMS_CS_SUM[:, :len(atoms) - 1] * PMMA_atom_weights

PMMA_ATOMS_CS_SUM_NORM = np.array([
        np.sum(PMMA_ATOMS_CS_SUM[:, :i], axis=1) for i in range(len(atoms))               
        ]).transpose() / np.sum(PMMA_ATOMS_CS_SUM, axis=1).reshape((len(E_arr), 1))

#%% Total with sum normed to 1
ATOMS_CS_SUM_NORM = [np.array([
        np.sum(CS[:, :i], axis=1) for i in range(len(CS[0, :]) + 1)
                ]).transpose() / np.sum(CS, axis=1).reshape((len(E_arr), 1))
        for CS in ATOMS_CS]

#%%
## Dirrerntial cross section integral, normed to 1
ATOMS_DIFF_CS_INT_NORM = [np.load(path + 'DIFF_CS_int_norm/' + atom + '_DIFF_CS_int_norm.npy')
        for atom in atoms]

#%% Excitation energy loss
ATOMS_EXC_DE = [np.load(path + 'EXC_dE/' + atom + '_EXC_dE.npy') for atom in atoms]

#%% Ionization stuff
## Binding energies
ATOMS_ION_E_BIND = [np.load(path + 'ION_Ebind/' + atom + '_ION_Ebind.npy') for atom in atoms]

## 2nd electron energies
ION_E_2ND = [np.load(path + 'ION_E2nd/' + atom + '_ION_E2nd.npy') for atom in atoms]

## Spectra of secondary electrons (integrals are normed to 1)
ATOMS_ION_SPECTRA = [[np.load(path + 'ION_spectra/' + atom + '_ION_' + subshell +
        '_spectra.npy') for subshell in subshells[atom]] for atom in atoms]

## Electron energy arrays in ionization spectra
ATOMS_ION_E_SPECTRA = [[np.load(path + 'ION_Espectra/' + atom + '_ION_' + subshell +
        '_Espectra.npy') for subshell in subshells[atom]] for atom in atoms]

#%%
#plt.loglog(E_arr, ATOMS_CS_SUM[:, 0], label='H')
#plt.loglog(E_arr, ATOMS_CS_SUM[:, 1], label='C')
#plt.loglog(E_arr, ATOMS_CS_SUM[:, 2], label='O')
#plt.loglog(E_arr, ATOMS_CS_SUM[:, 3], label='Si')
#plt.title('Total cross section')
#plt.xlabel('E, eV')
#plt.ylabel('$\sigma$, cm$^2$')
#plt.legend()
#plt.grid()

#%%
#C_CS = ATOMS_CS[1]
#label_list = ['elastic',
#              'excitation',
#              'K ionization',
#              'L1 ionization',
#              'L2 ionization',
#              'L3 ionization'
#              ]
#
#for i in range(6):
#    plt.loglog(E_arr, C_CS[:, i], label=label_list[i])
#
#plt.title('Total cross section')
#plt.xlabel('E, eV')
#plt.ylabel('$\sigma$, cm$^2$')
#plt.legend()
#plt.xlim([10, 25e+3])
#plt.grid()
#plt.show()