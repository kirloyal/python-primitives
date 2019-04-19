#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data  # (150, 4)
y = iris.target # (150,)

#%%

classes = np.unique(y)
for i in range(len(classes)):
    y_tar = classes[i]
    mu = np.mean(X[y==y_tar], axis=0)
    cov = np.cov(X[y==y_tar], rowvar=False)
    print(mu)
    print(cov)

#%%
