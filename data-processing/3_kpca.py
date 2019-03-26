#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data  # (150, 4)
y = iris.target # (150,)

#%%

from sklearn.decomposition import KernelPCA
# kpca = KernelPCA(kernel="rbf", gamma=1)
kpca = KernelPCA(kernel="rbf", gamma=1, fit_inverse_transform=True)


#%%

X_proj = kpca.fit_transform(X)
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.2, c = y, cmap='viridis')
plt.axis('equal');

#%%


X_back = kpca.inverse_transform(X_proj)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2, c = y, cmap='viridis', marker = 'o')
plt.scatter(X_back[:, 0], X_back[:, 1], alpha=0.2, c = y, cmap='viridis', marker = 'x')

plt.axis('equal');


#%%
