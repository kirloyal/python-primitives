#%%
### Python 3

import os
import json

if not os.path.exists('tmp'):
    os.makedirs('tmp')

filename = 'tmp/tmp.json'
data = {"int":10, "bool":True, "float":3.14, "str":"hello world", 0:"int 0", "0":"str 0"}

with open(filename, 'w') as outfile:  
    json.dump(data, outfile)

#%%

import json

with open(filename) as json_file:  
    data = json.load(json_file)
print(data)

# 0:"int 0" has gone now. Don't duplicate keys.


#%%
