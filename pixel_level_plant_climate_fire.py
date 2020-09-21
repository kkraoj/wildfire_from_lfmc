# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:26:19 2020

@author: kkrao
"""
#trial comment
import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from init import dir_root, dir_data
import fiona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import mapping
from pandas.tseries.offsets import DateOffset
from sklearn.linear_model import LinearRegression
import sys
import gdal
import matplotlib as mpl

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))




def segregate_plantClimate(plantClimatePath, n = 20, binning = "equal_width"):
    ds = gdal.Open(plantClimatePath)
    plantClimateMap = np.array(ds.GetRasterBand(1).ReadAsArray())
    ##FOR R2 MAP ONLY. Delete Otherwise
    plantClimateMap[plantClimateMap<0] = np.nan
    plantClimate_seg, areas = [],[]
    
    if binning == "equal_width":
        minVal , maxVal = np.nanmin(plantClimateMap), np.nanmax(plantClimateMap)
        bounds = np.linspace(minVal, maxVal, n+1)
        
    elif binning == "equal_area":
        raw = plantClimateMap.flatten()
        raw = raw[~np.isnan(raw)]
        bounds = histedges_equalN(raw, n)
        
    for i in range(n):
        lower, upper = bounds[i], bounds[i+1]
        locs = (plantClimateMap>=lower)&(plantClimateMap<upper)
        plantClimate_seg.append(plantClimateMap[locs].mean())
        areas.append(locs)  
               
    return np.array(plantClimate_seg), np.array(areas)

def clean (ba, vpd):
    valid = ba>0
    ba = ba[valid]
    vpd = vpd[valid]
    return ba, vpd
    
    
    
    
def segregate_fireClimate(areas):
    r2 ,coefs =[],[]
    for area in areas:
        ba, vpd = [],[]
        for i in range(19):
            filename = os.path.join(dir_root, "data","ba_vpd","%d.tif"%i)
            ds = gdal.Open(filename)
            ba_ = np.nansum((np.array(ds.GetRasterBand(2).ReadAsArray())>0).astype(int)[area==True])*16 #km sq.
            ba.append(ba_)
            vpd.append(np.nanmean(np.array(ds.GetRasterBand(1).ReadAsArray())[area==True]))
        
        ba = np.array(ba)
        vpd = np.array(vpd)
        ba, vpd = clean(ba, vpd)
        
        if len(ba)<=8:
            r2.append(np.nan)
            coefs.append(np.nan)
        else:
            ba = np.log10(ba)
            vpd = vpd.reshape(-1, 1)
            reg = LinearRegression().fit(vpd,ba)
            r2.append(reg.score(vpd, ba))
            coefs.append(reg.coef_[0])

    return np.array(r2),np.array(coefs)


plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_raw","lfmc_dfmc_1000hr_normalized_r2.tif")

plantClimate_seg,areas = segregate_plantClimate(plantClimatePath, n = 20, binning = "equal_area")
r2,coefs = segregate_fireClimate(areas)


fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(plantClimate_seg, coefs, s = 50, color = "k", edgecolor = "grey")
ax.set_xlabel(r"$R2(LFM,DFMC_{1000hr})$")
ax.set_ylabel(r"$\frac{d(log(BA))}{d(VPD)}$")