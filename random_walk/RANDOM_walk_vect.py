#%% Import
import numpy as np
import os
import importlib
import my_functions as mf
import my_variables as mv
import matplotlib.pyplot as plt

mf = importlib.reload(mf)
mv = importlib.reload(mv)

os.chdir(mv.sim_path_MAC + 'random_walk')

from numpy import sin, cos, arccos

#%%
def check_chain(chain_coords, now_mon_coords, step_2):
    for mon_coords in chain_coords[:-1, :]:    
        if np.sum((mon_coords - now_mon_coords)**2) < step_2:
            return False
    return True


def get_new_phi_theta(now_phi, now_theta):
    
    phi = 2*np.pi * mf.random()
    theta = np.deg2rad(180 - 109)
    
    if now_theta == 0:
        now_theta = 1e-5
    
    new_theta = arccos(
                    cos(now_theta) * cos(theta) +\
                    sin(now_theta) * sin(theta) * cos(phi)
                )
    
    cos_delta_phi = (cos(theta) - cos(new_theta)*cos(now_theta)) / (sin(now_theta)*sin(new_theta))
    
    if cos_delta_phi < -1:
        cos_delta_phi = -1
    elif cos_delta_phi > 1:
        cos_delta_phi = 1
    
    delta_phi = arccos(cos_delta_phi)
    
    if sin(theta)*sin(phi)/sin(new_theta) < 0:
        delta_phi *= -1
    
    new_phi = now_phi + delta_phi
    
    return new_phi, new_theta


def get_delta_xyz(L, new_phi, new_theta):
    
    delta_x = L * sin(new_theta) * cos(new_phi)
    delta_y = L * sin(new_theta) * sin(new_phi)
    delta_z = L * cos(new_theta)
    
    return np.array((delta_x, delta_y, delta_z))

#%%
step = 1
step_2 = step**2

L = 200

chain_coords = np.zeros((L, 3))
chain_coords[0, :] = 0, 0, 0

now_phi, now_theta = 0, 0

## collision counter
jam_cnt = 0
## collision link number
jam_pos = 0

i = 1

while i < L:
    
    mf.upd_progress_bar(i, L)
    
    while True:
        
        now_phi, now_theta = get_new_phi_theta(now_phi, now_theta)
        
        chain_coords[i, :] = chain_coords[i-1, :] + get_delta_xyz(step, now_phi, now_theta)
        
#        now_phi, now_theta = get_new_phi_theta(now_phi, now_theta)
        
        st = check_chain(chain_coords[:i, :], chain_coords[i, :], step_2)
        
#        if st:
        if True:
            break
        
        else: ## if no free space
            
            if np.abs(jam_pos - i) < 10: ## if new jam is near current link
                jam_cnt += 1 ## increase collision counter
            
            else: ## if new jam is on new link
                jam_pos = i ## set new collision position
                jam_cnt = 0 ## set collision counter to 0
            
            print(i, ': No free space,', jam_cnt)
            
            ## if possible, make rollback proportional to jam_cnt
            rollback_step = jam_cnt // 10
            
            if i - (rollback_step + 1) >= 0:
                i -= rollback_step
                continue
            
            else:
                print('Jam in very start!')
                break
    
    i += 1

#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(chain_coords[:, 0], chain_coords[:, 1], chain_coords[:, 2], 'bo-')

ax.set_xlabel('x, nm')
ax.set_ylabel('y, nm')
ax.set_zlabel('z, nm')

#%%
diff = chain_coords[1:, :] - chain_coords[:-1, :]
ans = np.linalg.norm(diff, axis=1)

#%%
def get_angles(chain_arr):
    
    def dotproduct(v1, v2):
      return sum((a*b) for a, b in zip(v1, v2))
    
    def length(v):
      return np.sqrt(dotproduct(v, v))
    
    def angle(v1, v2):
        return arccos(dotproduct(v1, v2) / (length(v1) * length(v2)))
    
    angles = []
    
    for i in range(len(chain_arr)-2):
        
        vector_1 =   chain_arr[i+1] - chain_arr[i]
        vector_2 = -(chain_arr[i+2] - chain_arr[i+1])
        
        angles.append(np.rad2deg(angle(vector_1, vector_2)))
        
    return np.array(angles)



angles = get_angles(chain_coords)



