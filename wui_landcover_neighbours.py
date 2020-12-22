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

import rasterio
from rasterio.enums import Resampling

from scipy.interpolate import interp2d
upscale_factor = 2

def resample(fullfilename):
    with rasterio.open(fullfilename) as dataset:
    
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(675),
                int(870)
            ),
            resampling=Resampling.bilinear
        )
    
        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        
        return data

def writeTif(array,outFileName, inFileName):
    
    ds = gdal.Open(inFileName)

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outFileName, array.shape[1], array.shape[0], 1, gdal.GDT_Byte)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(array)
    # outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band=None
    ds=None
        

def fill_neighbours(arr):
    y_size, x_size = arr.shape
    arr1 = np.empty((y_size + 2, x_size + 2),dtype = np.bool_)
    arr1[:,:] = False
    
    # arr1[:-2,:-2] = arr+arr1[:-2,:-2]
    # arr1[:-2,2:] = arr+arr1[:-2,2:]
    # arr1[2:,:-2] = arr+arr1[2:,:-2]
    # arr1[2:,2:] = arr+arr1[2:,2:]
    
    # arr1[:-2,1:-1] = arr+arr1[:-2,1:-1]
    # arr1[1:-1,2:] = arr+arr1[1:-1,2:]
    # arr1[1:-1,:-2] = arr+arr1[1:-1,:-2]
    # arr1[2:,1:-1] = arr+ arr1[2:,1:-1]
    
    arr1[:-2,:-2] = np.logical_or(arr,arr1[:-2,:-2])
    arr1[:-2,2:] = np.logical_or(arr,arr1[:-2,2:])
    arr1[2:,:-2] = np.logical_or(arr,arr1[2:,:-2])
    arr1[2:,2:] = np.logical_or(arr,arr1[2:,2:])
    
    arr1[:-2,1:-1] = np.logical_or(arr,arr1[:-2,1:-1])
    arr1[1:-1,2:] = np.logical_or(arr,arr1[1:-1,2:])
    arr1[1:-1,:-2] = np.logical_or(arr,arr1[1:-1,:-2])
    arr1[2:,1:-1] = np.logical_or(arr, arr1[2:,1:-1])  
    

    # arr1 = (arr1>0)*1

    return arr1[1:-1,1:-1]

def subset_CA(wui):
    wuiCA = wui[200:450,:300]
    return wuiCA
    
sns.set(font_scale = 1.1, style = "ticks")

filenames = ["urban2016mosaic.tif","urban1992mosaic.tif"]

# fig, axs = plt.subplots(1,2,figsize = (6,3))
ctr = 0
for filename in filenames:
    print(filename)
    fullfilename = os.path.join(dir_root, "data","WUI","30m",filename)
    ds = gdal.Open(fullfilename)
    print("reading file")
    wui = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype = np.bool_)
    # break
    print("computing neighbors")
    wui = fill_neighbours(wui).copy()
    outFileName = os.path.join(dir_root,"data","WUI","30m",filename[:-4]+"Neighbors.tif")
    inFileName = os.path.join(dir_root,"data","WUI","30m",filename)
    writeTif(wui.astype(np.uint8),outFileName, inFileName)
    print("reduce resolution")
    resampled = resample(outFileName)
    inFileName = os.path.join(dir_root,"data","WUI","urban1992.tif")
    outFileName = os.path.join(dir_root,"data","WUI","30m",filename[:-4]+"NeighborsResampled.tif")
    writeTif(resampled,outFileName, inFileName)
    
#     axs[ctr].imshow(wui, vmax = 1, vmin = 0)
#     axs[ctr].set_title(1990+ctr*20)
#     axs[ctr].axis("off")
#     ctr+=1
# plt.tight_layout()

