# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:07:45 2020

@author: kkrao
"""


import pandas as pd
import os
from init import dir_root, dir_data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

import seaborn as sns
import matplotlib as mpl
import gdal
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
sns.set(font_scale = 0.9, style = "ticks")


areas = {"NW":np.array([[0,0],[322,435]]),"NE":np.array([[0,435],[322,870]]),"SE":np.array([[322,435],[645,870]]),"SW":np.array([[322,0],[645,435]])}


filename = os.path.join(dir_root, "data","ba_vpd","0.tif")
ds = gdal.Open(filename)

vpd = np.array(ds.GetRasterBand(1).ReadAsArray())
vpd = np.expand_dims(vpd,0)
for i in range(1,19):

    filename = os.path.join(dir_root, "data","ba_vpd","%d.tif"%i)
    ds = gdal.Open(filename)

    vpd_ = np.array(ds.GetRasterBand(1).ReadAsArray())
    vpd_ = np.expand_dims(vpd_,0)
    vpd= np.concatenate((vpd,vpd_),axis = 0)

vpd = vpd - vpd.mean(axis = 0) # anomalies
vpd.shape
# plt.imshow(vpd_[0])

def segregate(array, area = "SW"):
    
    sub = array[:,areas[area][0,0]:areas[area][1,0],areas[area][0,1]:areas[area][1,1]]
    return sub
    

fig, ax = plt.subplots(figsize = (3,1))
for area in areas.keys():
    sub = segregate(vpd, area = area)
    
    mean = np.nanmean(np.nanmean(sub,axis = 1), axis = 1)
    std = np.nanstd(np.nanstd(sub,axis = 1), axis = 1)
    
    df = pd.DataFrame({"mean":mean,"std":std},index = range(2001,2020))
    
    ax.scatter(df.index, df['mean'], s = 5, label = area)
    ax.errorbar(df.index, df['mean'], yerr = df['std'],linewidth = 0,elinewidth=1,ecolor = "grey",zorder = -1)
ax.set_xticks([2000,2005,2010,2015,2020])
ax.set_ylabel("VPD Anomaly")
ax.xaxis.set_minor_locator(MultipleLocator(1))
plt.legend(bbox_to_anchor=[1, 1],loc = "upper left")
    
    
    

    