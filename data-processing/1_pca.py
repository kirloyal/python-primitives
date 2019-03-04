#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data  # (150, 4)
y = iris.target # (150,)

#%%

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
print(pca.components_)
print(pca.explained_variance_)
print(pca.mean_)

#%%

X_proj = pca.transform(X) 
# X_proj = np.dot(X - pca.mean_, pca.components_.T)
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.2, c = y, cmap='viridis')
plt.axis('equal');

#%%
