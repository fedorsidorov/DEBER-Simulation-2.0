#%%

file = ''

cnt = 0


for i in range(1, 5):
    
    for j in range(1, 10):
        
        for k in range(10):
            
            file += 'EV      ' + str(j) + '.' + str(k) + '00E' + str(i) + '\n'
            cnt += 1