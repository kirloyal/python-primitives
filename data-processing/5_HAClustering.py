#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data  # (150, 4)
y = iris.target # (150,)

#%%

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3)
clustering = ac.fit(X)

print clustering
print clustering.labels_


#%%
