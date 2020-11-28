# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:21:00 2020

@author: kkrao
"""



import os
from init import dir_root, dir_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import gdal
import matplotlib as mpl
import seaborn as sns


def fill_neighbours(arr):
    y_size, x_size = arr.shape
    arr1 = np.empty((y_size + 2, x_size + 2))
    arr1[1:-1, 1:-1] = arr
    arr1[:-2,:-2] = arr+arr1[:-2,:-2]
    arr1[:-2,2:] = arr+arr1[:-2,2:]
    arr1[2:,:-2] = arr+arr1[2:,:-2]
    arr1[2:,2:] = arr+arr1[2:,2:]
    arr1 = (arr1>0)*1
    
    
    return arr1[1:-1,1:-1]

def subset_CA(wui):
    wuiCA = wui[200:450,:300]
    return wuiCA
    
sns.set(font_scale = 1.1, style = "ticks")

filenames = ["wui1990.tif","wui2000.tif","wui2010.tif"]

fig, axs = plt.subplots(1,3,figsize = (9,3))
ctr = 0
for filename in filenames:
    fullfilename = os.path.join(dir_root, "data","WUI",filename)
    ds = gdal.Open(fullfilename)
    wui = np.array(ds.GetRasterBand(1).ReadAsArray())
    wui = fill_neighbours(wui).copy()
    axs[ctr].imshow(wui, vmax = 1, vmin = 0)
    axs[ctr].set_title(1990+ctr*10)
    axs[ctr].axis("off")
    ctr+=1
plt.tight_layout()


fullfilename = os.path.join(dir_root, "data","WUI",filename)
ds = gdal.Open(fullfilename)
wui = np.array(ds.GetRasterBand(1).ReadAsArray())

plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","lfmc_dfmc_100hr_lag_6_lfmc_dfmc_norm_positive_coefSum.tif")
ds = gdal.Open(plantClimatePath)
plantClimate = np.array(ds.GetRasterBand(1).ReadAsArray())

fig, ax = plt.subplots(figsize = (3,3))
ax.imshow(plantClimate,vmin = 0, vmax = 2)
ax.axis('off')


fig, axs = plt.subplots(1,3,figsize = (9,3), sharey = True)
ctr = 0
for filename in filenames:
    ax = axs[ctr]
    fullfilename = os.path.join(dir_root, "data","WUI",filename)
    ds = gdal.Open(fullfilename)
    wui = np.array(ds.GetRasterBand(1).ReadAsArray())
    # sns.kdeplot(data = plantClimate[wui==0], ax = ax, color = "grey", label="Non-WUI")
    # sns.kdeplot(data = plantClimate[wui==1], ax = ax, color = "lightcoral", label = "WUI Intermix")
    # sns.kdeplot(data = plantClimate[wui==2], ax = ax, color = "firebrick",label = "WUI Interface")
    
    ### only CA
    # sns.kdeplot(data = subset_CA(np.where(wui==0, plantClimate, np.full(plantClimate.shape, np.nan))).flatten(), ax = ax, color = "grey", label="Non-WUI")
    # sns.kdeplot(data = subset_CA(np.where(wui==1, plantClimate, np.full(plantClimate.shape, np.nan))).flatten(), ax = ax, color = "lightcoral", label = "WUI Intermix")
    # sns.kdeplot(data = subset_CA(np.where(wui==2, plantClimate, np.full(plantClimate.shape, np.nan))).flatten(), ax = ax, color = "firebrick",label = "WUI Interface")

    
    ### fille neighbours
    wui = fill_neighbours(wui).copy()
    sns.kdeplot(data = plantClimate[wui==1], ax = ax, color = "lightcoral", label = "WUI neighbours")
    sns.kdeplot(data = plantClimate[wui==0], ax = ax, color = "grey", label="Others")   

    ### fille neighbours CA
    # wui = fill_neighbours(wui).copy()
    # sns.kdeplot(data = subset_CA(np.where(wui==1, plantClimate, np.full(plantClimate.shape, np.nan))).flatten(), ax = ax, color = "lightcoral", label = "WUI neighbours")
    # sns.kdeplot(data = subset_CA(np.where(wui==0, plantClimate, np.full(plantClimate.shape, np.nan))).flatten(), ax = ax, color = "grey", label="Others")
    # ax.axvline(x = np.nanmean(subset_CA(np.where(wui==0, plantClimate, np.full(plantClimate.shape, np.nan)))),linewidth=2, color='grey', label = "_nolegend_")
    # ax.axvline(x = np.nanmean(subset_CA(np.where(wui==1, plantClimate, np.full(plantClimate.shape, np.nan)))),linewidth=2, color='lightcoral', label = "_nolegend_")
    # # ax.set_ylim(0,1)
    ax.set_xlabel("$\sigma$")
    ax.set_title(1990+ctr*10)
    ax.get_legend().remove()
    ctr+=1
# ax.legend()
axs[2].legend(bbox_to_anchor = [1,1])
axs[0].set_ylabel("Density")
plt.tight_layout()




