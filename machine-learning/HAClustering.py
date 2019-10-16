# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data  # (150, 4)
y = iris.target # (150,)


#%%

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist


Z = linkage(X, 'ward')
# print(Z) # [i, j, dist, number of member]
c, coph_dists = cophenet(Z, pdist(X))
print(c)
cluster = fcluster(Z, t=10, criterion='distance')
print(cluster)
cluster_mean = []
for i in range(cluster.max()):
    cluster_mean.append(X[cluster == i+1].mean(axis=0))
cluster_mean = np.array(cluster_mean)
print(cluster_mean)

#%%

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.axhline(y=10, color='r', linestyle='--')
plt.show()

#%%

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_proj = pca.transform(X) 
cluster_mean_proj = pca.transform(cluster_mean) 

plt.style.use(['default'])
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.2, c = y, cmap='viridis', marker = '.')
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.2, c = cluster, cmap='viridis', marker = 'o')
plt.scatter(cluster_mean_proj[:, 0], cluster_mean_proj[:, 1])
plt.axis('equal');


#%%


from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3)
clustering = ac.fit(X)

print(clustering)
print(clustering.labels_)

#%%

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
X_proj = pca.transform(X) 

plt.style.use(['default'])
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.2, c = y, cmap='viridis', marker = '.')
plt.scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.2, c = clustering.labels_, cmap='viridis', marker = 'o')
plt.axis('equal');



#%%
