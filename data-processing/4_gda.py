#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data  # (150, 4)
y = iris.target # (150,)


#%%

from sklearn.metrics.pairwise import polynomial_kernel
K = polynomial_kernel(X, coef0 = 1 , degree=2)

# from sklearn.metrics.pairwise import rbf_kernel
# K = rbf_kernel(X)


#%%

n_components = 2
n_samples = X.shape[0]

classes = list(set(y.flatten()))
W = np.zeros((n_samples,n_samples))

for j in range(len(classes)):
    y_tar = classes[j]
    l_j = (y==y_tar).sum()
    W[np.outer(y==y_tar,y==y_tar)] = 1./l_j * np.ones(l_j*l_j)

N = np.dot(K,K) + 0.01 * np.eye(n_samples)
M = np.dot(np.dot(K,W),K) 

w, v = np.linalg.eig(np.dot(np.linalg.inv(N),M))
idx = np.isreal(w)

w = np.real(w[idx])
v = np.real(v[:,idx])
alpha = np.real(v[:,:n_components])

#%%

X_proj = np.dot(K,alpha)
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.2, c = y, cmap='viridis')
plt.axis('equal');

