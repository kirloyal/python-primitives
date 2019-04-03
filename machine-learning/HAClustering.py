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

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_proj = pca.transform(X) 

#%%

plt.style.use(['default'])
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.2, c = y, cmap='viridis', marker = 'o')
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.2, c = clustering.labels_, cmap='viridis', marker = 'x')
plt.axis('equal');


#%%

