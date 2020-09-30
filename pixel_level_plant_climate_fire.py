# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:26:19 2020

@author: kkrao
"""
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
import seaborn as sns
import matplotlib as mpl
from scipy import stats

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))




def segregate_plantClimate(plantClimatePath, n = 20, binning = "equal_area"):
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
    r2 ,coefs, stderrors =[],[], []
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
            # vpd = vpd.reshape(-1, 1)
            # reg = LinearRegression().fit(vpd,ba)
            # r2.append(reg.score(vpd, ba))
            # coefs.append(reg.coef_[0])
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(vpd,ba)
            r2.append(r_value**2)
            coefs.append(slope)
            stderrors.append(std_err)
            

    return np.array(r2),np.array(coefs), np.array(stderrors)

hr = "100hr"
var = "coefAbsSum"
lag = 6
folder = "lfmc_dfmc_anomalies"
norm = "lfmc_norm"
# for hr in ["1000hr", "100hr"]:
#     for var in ["coefAbsSum", "coefSum","r2"]:
    
# for norm in ["no_norm","lfmc_norm","dfmc_norm","lfmc_dfmc_norm"]:
    # for var in ["coefSum","coefPositiveSum","coefAbsSum","r2"]    :
plantClimatePath = os.path.join(dir_root, "data","arr_pixels_%s"%folder,"lfmc_dfmc_%s_lag_%d_%s_%s.tif"%(hr,lag,norm,var))
# plantClimatePath = os.path.join(dir_root, "data","mean","vpd_mean.tif")

plantClimate_seg,areas = segregate_plantClimate(plantClimatePath, n = 20, binning = "equal_area")
r2,coefs, stderrors = segregate_fireClimate(areas)

df = pd.DataFrame({"x":plantClimate_seg,"y":coefs})
fig, ax = plt.subplots(figsize = (3,3))
sns.regplot(x=plantClimate_seg, y=coefs,ax=ax)
ax.errorbar(plantClimate_seg, coefs, yerr = stderrors, color = "lightgrey", zorder = -1)
ax.scatter(plantClimate_seg, coefs, s = 50, color = "k", edgecolor = "grey")
ax.set_xlabel(r"%s, %s"%(var, hr))
# ax.set_xlabel(r"Mean VPD")
ax.set_ylabel(r"$\frac{d(log(BA))}{d(VPD)}$")
ax.set_title(norm)
print(df.corr()**2)
plt.show()