# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:56:28 2020

@author: kkrao
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from init import dir_root
# sphinx_gallery_thumbnail_number = 2

sns.set(style='ticks',font_scale = 1.5)

r = pd.read_excel(os.path.join(dir_root, "working.xlsx"), sheet_name = "r")
p = pd.read_excel(os.path.join(dir_root, "working.xlsx"), sheet_name = "p")

cmap =  mpl.cm.BrBG

cmap.set_bad("lightgrey")
ax = sns.heatmap(r, cmap = cmap, mask=r.isnull(),vmin = -0.8, vmax = 0.8, square = True, linewidths = 0.5, cbar_kws = {'ticks':np.linspace(-0.8,0.8,5)})
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")
p = (p<.05)
p.index = np.arange(p.shape[0])+0.5
p.columns = np.arange(p.shape[1])+0.5

for i in p.index:
    for j in p.columns:
        if p.loc[i,j]:
            ax.scatter(j,i,marker = "o", color = "white", s = 50, edgecolor = "k")

# harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


# fig, ax = plt.subplots()
# im = ax.imshow(harvest)

# # We want to show all ticks...
# ax.set_xticks(np.arange(len(farmers)))
# ax.set_yticks(np.arange(len(vegetables)))
# # ... and label them with the respective list entries
# ax.set_xticklabels(farmers)
# ax.set_yticklabels(vegetables)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# for i in range(len(vegetables)):
#     for j in range(len(farmers)):
#         text = ax.text(j, i, harvest[i, j],
#                        ha="center", va="center", color="w")

# ax.set_title("Harvest of local farmers (in tons/year)")
# fig.tight_layout()
# plt.show()