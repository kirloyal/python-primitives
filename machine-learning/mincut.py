#%%

from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

import matplotlib.pyplot as plt 

plt.gray() 
plt.matshow(digits.images[0]) 
plt.show() 

#%%

import numpy as np

img = digits.images[0]
img = img + np.random.normal(8, 1, img.shape)

img = img + img.min()
img = img / img.max()


plt.gray() 
plt.matshow(img) 
plt.show() 


#%%

import matplotlib.pyplot as plt
import networkx as nx


G = nx.DiGraph()

w_e = 0.
for i in range(0, len(img)):
    for j in range(0, len(img[i])):
        G.add_edge('source', (i,j), capacity = -np.log(img[i,j]) )
        G.add_edge((i,j), 'sink', capacity = -np.log(1-img[i,j]) )
        
        if i > 0:
            if j > 0:
                G.add_edge((i-1,j-1), (i,j), capacity = w_e)
                G.add_edge((i,j), (i-1,j-1), capacity = w_e)
            G.add_edge((i-1,j), (i,j), capacity = w_e)
            G.add_edge((i,j), (i-1,j), capacity = w_e)
            if j < len(img) - 1:
                G.add_edge((i-1,j+1), (i,j), capacity = w_e)
                G.add_edge((i,j), (i-1,j+1), capacity = w_e)
        if j > 0:
            G.add_edge((i,j-1), (i,j), capacity = w_e)
            G.add_edge((i,j), (i,j-1), capacity = w_e)
        

print G
pos = nx.spring_layout(G, iterations=100)
nx.draw(G, pos, node_color='k', node_size=5, with_labels=False)
plt.show()
        
#%%

cut_value, partition = nx.minimum_cut(G, 'source', 'sink')
reachable, non_reachable = partition

segmented = np.zeros_like(img)
filled = non_reachable.copy()
filled.remove('sink')
for i,j in filled:
    segmented[i,j] = 1

plt.gray() 
plt.matshow(segmented) 
plt.show() 


#%%
