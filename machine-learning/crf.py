# ref)
# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_ncut.html#sphx-glr-auto-examples-segmentation-plot-ncut-py
# https://github.com/lucasb-eyer/pydensecrf.git

#%%

from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt


img = data.coffee()

slic = segmentation.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, slic, mode='similarity')
anno_lbl = graph.cut_normalized(slic, g) + 1
anno_rgb = color.label2rgb(anno_lbl, img, kind='avg')

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(img)
ax[1].imshow(anno_rgb)

for a in ax:
    a.axis('off')

plt.tight_layout()

#%%

import sys
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import cv2 

colors, labels = np.unique(anno_lbl, return_inverse=True)
HAS_UNK = 0 in colors
if HAS_UNK:
    print("has unkown")
    colors = colors[1:]

n_labels = len(set(labels.flat)) - int(HAS_UNK)
print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

#%%

d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
d.setUnaryEnergy(U)
d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                        compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

MAP = np.argmax(Q, axis=0)
MAP = colors[MAP].reshape(img.shape[:2])
#%%

out = color.label2rgb(MAP, img, kind='avg')
fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(img)
ax[1].imshow(anno_rgb)
ax[2].imshow(out)

for a in ax:
    a.axis('off')
plt.tight_layout()

#%%
