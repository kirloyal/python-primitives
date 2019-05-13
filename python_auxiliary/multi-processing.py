import os
import math

def get_primenumber(tu_range):
    res = []
    procid = os.getpid()
    for i in range(*tu_range):
        if i < 2:
            continue
        bFlag = False
        for j in range(2, int(math.sqrt(i)) + 1):
            if i % j == 0:
                bFlag = True
                break
        if not bFlag:
            print('prime number {} by proc {}'.format(i, procid))
            res.append(i)
    return res
#%%
 
import multiprocessing 

params = [(10000,20000), (20000, 30000), (30000, 40000)]
pool = multiprocessing.Pool(processes = 3)
res = pool.map(get_primenumber, params)
print res
