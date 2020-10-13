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


areas_dict = {"NW":np.array([[0,0],[322,435]]),"NE":np.array([[0,435],[322,870]]),"SE":np.array([[322,435],[645,870]]),"SW":np.array([[322,0],[645,435]])}

ecoregions_dict = {1: "Deserts",
                    # 2: "Med. Cal.",
                    # 4: "Temp. Sierras",
                    5: "NW Forest",
                    # 6: "Marine Forest",
                    8: "Great Plains"
                    }
filename = os.path.join(dir_root, "data","ecoregions","L1","L1_ecoregions_from_gee.tif")
ds = gdal.Open(filename)
ecoregions = np.array(ds.GetRasterBand(1).ReadAsArray())[:-1,:]                           
    
                    
def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

def segregate_plantClimate(plantClimatePath, n = 20, binning = "equal_area", localize = False, ecoregionalize = False):
    ds = gdal.Open(plantClimatePath)
    plantClimateMap = np.array(ds.GetRasterBand(1).ReadAsArray())
    ##FOR R2 MAP ONLY. Delete Otherwise
    # plantClimateMap[plantClimateMap<0] = np.nan
    plantClimate_seg, areas = [],[]
    
    if binning == "equal_width":
        minVal , maxVal = np.nanmin(plantClimateMap), np.nanmax(plantClimateMap)
        bounds = np.linspace(minVal, maxVal, n+1)
        
    elif binning == "equal_area":
        raw = plantClimateMap.flatten()
        raw = raw[~np.isnan(raw)]
        bounds = histedges_equalN(raw, n)
    
    if localize:
        for quadrant in areas_dict.keys():
            plantClimateQuadrant = localizeUSA(plantClimateMap, area = quadrant)
            for i in range(n):
                lower, upper = bounds[i], bounds[i+1]
                locs = (plantClimateQuadrant>=lower)&(plantClimateQuadrant<upper)
                plantClimate_seg.append(np.nanmean(plantClimateQuadrant[locs]))
                areas.append(locs)  
    elif ecoregionalize:
        for region in ecoregions_dict.keys():
            # print(region)
            plantClimateQuadrant = ecoregionalizeUSA(plantClimateMap, area = region)
            # raw = plantClimateQuadrant.flatten()
            # raw = raw[~np.isnan(raw)]
            # n = int(len(raw)/1e4)
            # print(n)
            # bounds = histedges_equalN(raw, n)

            for i in range(n):
                lower, upper = bounds[i], bounds[i+1]
                locs = (plantClimateQuadrant>=lower)&(plantClimateQuadrant<upper)
                plantClimate_seg.append(np.nanmean(plantClimateQuadrant[locs]))
                areas.append(locs)  
    else:
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
            # ba = np.log10(ba)
            # vpd = vpd.reshape(-1, 1)
            # reg = LinearRegression().fit(vpd,ba)
            # r2.append(reg.score(vpd, ba))
            # coefs.append(reg.coef_[0])
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(vpd,ba)
            r2.append(r_value**2)
            coefs.append(slope)
            stderrors.append(std_err)
            

    return np.array(r2),np.array(coefs), np.array(stderrors)

def segregate_fire(areas):
    bas, stderrors  =[], []
    for area in areas:
        ba =[]
        for i in range(19):
            filename = os.path.join(dir_root, "data","ba_vpd","%d.tif"%i)
            ds = gdal.Open(filename)
            ba_ = np.nansum((np.array(ds.GetRasterBand(2).ReadAsArray())>0).astype(int)[area==True])*16 #km sq.
            ba.append(ba_)
        
        bas.append(np.array(ba).sum())
        stderrors.append(np.array(ba).std())
        # ba, vpd = clean(ba, vpd)
    return np.array(bas), np.array(stderrors)


def localizeUSA(array, area = "SW"):
    array = array.copy()
    mask = np.ones_like(array, np.bool)
    mask[areas_dict[area][0,0]:areas_dict[area][1,0],areas_dict[area][0,1]:areas_dict[area][1,1]] = 0
    array[mask] = np.nan
    # sub = array[mask]
    # sub = array[areas_dict[area][0,0]:areas_dict[area][1,0],areas_dict[area][0,1]:areas_dict[area][1,1]]
    return array

def ecoregionalizeUSA(array, area = None):
    array = array.copy()
    mask = np.ones_like(array, np.bool)
    mask[np.where(ecoregions==area)] = 0
    array[mask] = np.nan
    # sub = array[mask]
    # sub = array[areas_dict[area][0,0]:areas_dict[area][1,0],areas_dict[area][0,1]:areas_dict[area][1,1]]
    return array

hr = "100hr"
var = "coefSum"
lag = 6
folder = "lfmc_dfmc_anomalies"
norm = "lfmc_dfmc_norm"
coefs_type = "positive"
nLocal = 10
nGlobal = nLocal*len(ecoregions_dict)
# for hr in ["1000hr", "100hr"]:
#     for var in ["coefAbsSum", "coefSum","r2"]:
    
# for norm in ["no_norm","lfmc_norm","dfmc_norm","lfmc_dfmc_norm"]:
    # for var in ["coefSum","coefPositiveSum","coefAbsSum","r2"]    :
plantClimatePath = os.path.join(dir_root, "data","arr_pixels_%s"%folder,"lfmc_dfmc_%s_lag_%d_%s_%s_%s.tif"%(hr,lag,norm,coefs_type,var))
# plantClimatePath = os.path.join(dir_root, "data","mean","vpd_mean.tif")

plantClimate_seg,areas = segregate_plantClimate(plantClimatePath, n = nLocal, binning = "equal_area", localize = False, ecoregionalize = True)
r2,coefs, stderrors = segregate_fireClimate(areas)

df = pd.DataFrame({"x":plantClimate_seg,"y":coefs})


#%% main plot
# fig, ax = plt.subplots(figsize = (3,3))
# sns.regplot(x=plantClimate_seg, y=coefs,ax=ax)

# ax.errorbar(plantClimate_seg, coefs, yerr = stderrors, color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 2)
# ax.scatter(plantClimate_seg, coefs, s = 50, color = "k", edgecolor = "grey")
# ax.set_xlabel(r"%s, %s"%(var, hr))
# # ax.set_xlabel(r"Mean VPD")
# ax.set_ylabel(r"$\frac{d(BA)}{d(VPD)}$")
# ax.set_title(norm)
# print(df.corr()**2)
# plt.show()

#%% segregated plot
ctr=0
fig, ax = plt.subplots(figsize = (3,3))
for quadrant in ecoregions_dict.keys():
    
    ax.errorbar(df.x.iloc[ctr*nLocal:ctr*nLocal+nLocal], df.y.iloc[ctr*nLocal:ctr*nLocal+nLocal], yerr = stderrors[ctr*nLocal:ctr*nLocal+nLocal], color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 2)
    ax.scatter(df.x.iloc[ctr*nLocal:ctr*nLocal+nLocal], df.y.iloc[ctr*nLocal:ctr*nLocal+nLocal], s = 50, edgecolor = "grey",label = ecoregions_dict[quadrant])
    ctr+=1
ax.set_xlabel(r"%s, %s"%(var, hr))
ax.set_ylabel(r"$\frac{d(BA)}{d(VPD)}$")
ax.set_title(norm)
plt.legend(bbox_to_anchor=[1, 1],loc = "upper left")
    

#%% Only plant plot
# ds = gdal.Open(plantClimatePath)
# plantClimateMap = np.array(ds.GetRasterBand(1).ReadAsArray())
# ctr=0
# fig, axs = plt.subplots(1, len(ecoregions_dict),figsize = (6,2), sharey = True)
# for quadrant in ecoregions_dict.keys():
#     sub = ecoregionalizeUSA(plantClimateMap, quadrant)
#     arr = sub.flatten()
#     arr = arr[~np.isnan(arr)]
#     axs[ctr].boxplot(arr)
#     axs[ctr].set_xticklabels([ecoregions_dict[quadrant]])
#     ctr+=1
# # ax.set_xlabel(r"%s, %s"%(var, hr))
# axs[0].set_ylabel(r"$\sum\beta$")
# ax.set_title(norm)
# plt.legend(bbox_to_anchor=[1, 1],loc = "upper left")


# #%% only fire plot
# bas,stderrors = segregate_fire(areas)
# df = pd.DataFrame({"x":plantClimate_seg,"y":bas})

# ctr=0
# n=10

# fig, ax = plt.subplots(figsize = (3,3))
# sns.regplot(x=plantClimate_seg, y=bas,ax=ax)

# # ax.errorbar(plantClimate_seg, bas, yerr = stderrors, color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 2)
# ax.scatter(plantClimate_seg, bas, s = 50, color = "k", edgecolor = "grey")
# ax.set_xlabel(r"%s, %s"%(var, hr))
# # ax.set_xlabel(r"Mean VPD")
# ax.set_ylabel(r"BA (km$^2$")
# ax.set_title(norm)

# plt.show()


# #%% only fire plot segregated

# for quadrant in areas_dict.keys():
#     ax.scatter(df.x.iloc[ctr*n:ctr*n+n], df.y.iloc[ctr*n:ctr*n+n], s = 50, edgecolor = "grey",label = quadrant)
#     # ax.scatter(plantClimate_seg, bas, s = 50, edgecolor = "grey")
#     ctr+=1
# ax.set_xlabel(r"%s, %s"%(var, hr))
# ax.set_ylabel(r"log(BA)")
# ax.set_title(norm)
# plt.legend(bbox_to_anchor=[1, 1],loc = "upper left")
# plt.show()

#%% r2 histogram

# r2Global = [0.58175267, 0.45508388, 0.49462511, 0.5185809 , 0.52328667,
#         0.51330114, 0.57383786, 0.4652417 , 0.47768383, 0.45337598,
#         0.4988171 , 0.57259561, 0.5209921 , 0.5461768 , 0.56529779]

# fig, axs = plt.subplots(1, 5, figsize = (5,2), sharey = True) 
# axs[4].boxplot(r2Global)
# # ax.hist(r2, histtype = "step",label = "local")
# axs[4].set_xticklabels(["Global"])

# ctr=0
# n=10
# for quadrant in areas_dict.keys():
#     axs[ctr].boxplot(r2[ctr*n:ctr*n+n])
#     # ax.hist(r2, histtype = "step",label = "local")
#     axs[ctr].set_xticklabels([quadrant])

#     ctr+=1
# axs[0].set_ylabel("r$^2$")
# axs[0].set_ylim(0,1)    
    
    