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
