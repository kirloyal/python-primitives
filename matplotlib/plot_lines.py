# https://matplotlib.org/3.1.1/gallery/shapes_and_collections/line_collection.html

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

import numpy as np

fig, ax = plt.subplots()
colors = np.array([(1., 0., 0., 1.), (0., 1., 0., 1.)])
lines = np.random.random((2,3,2))
line_segments = LineCollection(lines, linewidths=0.5, colors=colors, linestyles='solid')
ax.add_collection(line_segments)
ax.set_title('draw multi lines')
plt.show()

#%%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import colors as mcolors

import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = np.array([(1., 0., 0., 1.), (0., 1., 0., 1.)])
lines = np.random.random((2,4,3))
line_segments = Line3DCollection(lines, linewidths=0.5, colors=colors, linestyles='solid')
ax.add_collection(line_segments)
ax.set_title('draw multi lines')
plt.show()