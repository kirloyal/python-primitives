import os
import math

def get_primenumber(tu_range):
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
#%%
 
from multiprocessing import Process


params = [(100000,200000), (200000, 300000), (300000, 400000)]
procs = []

for index, param in enumerate(params):
    proc = Process(target=get_primenumber, args=(param,))
    procs.append(proc)
    proc.start()

for proc in procs:
    proc.join()