#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data  # (150, 4)
y = iris.target # (150,)

#%%

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

clf = LDA(n_components = 2)
clf.fit(X, y)
print(clf.xbar_)
print(clf.scalings_)
print(clf.coef_)
print(clf.intercept_)
print(clf.explained_variance_ratio_)
print(clf.means_)

#%%

X_proj = clf.transform(X) 
X_proj2 = np.dot(X - clf.xbar_, clf.scalings_)
plt.style.use(['default'])
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.2, c = y, cmap='viridis')
plt.axis('equal');

#%%

print(clf.predict(X))
print(y)


#%%
for v in vars(clf).keys():
    val = vars(clf)[v]
    if isinstance(val, np.ndarray):
        print(v,val.shape)

#%%


n_components = 2

classes = list(set(y.flatten()))

mu = X.mean(axis=0)
mu_k = np.zeros((len(classes),X.shape[1]))
S_w = np.zeros((X.shape[1],X.shape[1]))
S_b = np.zeros((X.shape[1],X.shape[1]))
for i in range(len(classes)):
    y_tar = classes[i]
    mu_k[i] = X[y==y_tar].mean(axis=0)
    X_centered = X[y==y_tar] - mu_k[i]
    S_w += (X_centered[:,:,np.newaxis] * X_centered[:,np.newaxis,:]).sum(axis=0)
    
    mu_centered = mu_k[i] - mu
    S_b += (y==y_tar).sum() * (mu_centered[np.newaxis,:] * mu_centered[:,np.newaxis])

w, v = np.linalg.eig(np.dot(np.linalg.inv(S_w),S_b))
print(w)

v = -v[:,:n_components]

X_proj = np.dot(X,v)
plt.style.use(['default'])
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.2, c = y, cmap='viridis')
plt.axis('equal');


#%%
