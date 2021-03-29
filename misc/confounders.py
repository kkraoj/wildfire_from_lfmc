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
from matplotlib.colors import ListedColormap
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
from pylab import cm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from plotmap import plotmap




areas_dict = {"NW":np.array([[0,0],[322,435]]),"NE":np.array([[0,435],[322,870]]),"SE":np.array([[322,435],[645,870]]),"SW":np.array([[322,0],[645,435]])}

ecoregions_dict = {1: "Deserts",
                    # 2: "Med. Cal.",
                    # 4: "Temp. Sierras",
                    5: "NW Forest",
                    # 6: "Marine Forest",
                    8: "Great Plains"
                    }

lc_dict = {'forest':[50,70,90,100, 160],
           'shrubs':[110,120,130],
           'grass':[140, 150]
           # 'crops':[14,20,30]
           }
ppt_dict = {'low':[0,250],
           'medium':[250,750],
           'high':[750,10000]
           }

# def get_keys(dictionary, val):
#     keys = []
#     for key, value in dictionary:
#         if value==val:
#             keys.append(key)
#     return keys

filename = os.path.join(dir_root, "data","ecoregions","L1","L1_ecoregions_from_gee.tif")
ds = gdal.Open(filename)
ecoregions = np.array(ds.GetRasterBand(1).ReadAsArray())[:-1,:]                           
    
filename = os.path.join(dir_root, "data","mean","landcover.tif")
ds = gdal.Open(filename)
landcover = np.array(ds.GetRasterBand(1).ReadAsArray())


filename = os.path.join(dir_root, "data","mean","ppt_mean.tif")
ds = gdal.Open(filename)
ppt = np.array(ds.GetRasterBand(1).ReadAsArray())*365 #(daily to yearly conversion)
ppt[landcover==0] = np.nan

sns.set(font_scale = 1., style = "ticks")
plt.style.use("pnas")
fig, ax=  plt.subplots(figsize = (3,3))       
colors = sns.color_palette("Blues",n_colors = 3).as_hex()
colors = list(np.repeat(colors[0],2)) + list(np.repeat(colors[1],4)) + list(np.repeat(colors[2],10))
plot = ax.imshow(ppt, vmin=0,vmax=2000, cmap =ListedColormap(colors))
plt.axis('off')
cax = fig.add_axes([1, 0.2, 0.05, 0.6])
cax.set_title("Annual\nprecipitation (mm)")
cbar = fig.colorbar(plot, cax=cax, orientation='vertical')
cbar.set_ticks([0,250,750,2000])
plt.show()

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

def segregate_plantClimate(plantClimatePath, n = 20, binning = "equal_area", localize = False, ecoregionalize = False, landcoverize = False, pptize = False):
    ds = gdal.Open(plantClimatePath)
    plantClimateMap = np.array(ds.GetRasterBand(1).ReadAsArray())
    ##FOR R2 MAP ONLY. Delete Otherwise
    # plantClimateMap[plantClimateMap<0] = np.nan
    plantClimate_seg, areas, nEcoregions = [],[], [0]
    
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
            raw = plantClimateQuadrant.flatten()
            raw = raw[~np.isnan(raw)]
            # n = int(len(raw)/1e4) # equal area of point for all ecoregions
            n = 10
            nEcoregions.append(n)
            # print(n)
            bounds = histedges_equalN(raw, n)

            for i in range(n):
                lower, upper = bounds[i], bounds[i+1]
                locs = (plantClimateQuadrant>=lower)&(plantClimateQuadrant<upper)
                plantClimate_seg.append(np.nanmean(plantClimateQuadrant[locs]))
                areas.append(locs)  
    elif landcoverize:
        for region in lc_dict.values():
            # print(region)
            plantClimateQuadrant = landcoverizeUSA(plantClimateMap, area = region)
            raw = plantClimateQuadrant.flatten()
            raw = raw[~np.isnan(raw)]
            # n = int(len(raw)/1e4) # equal area of point for all ecoregions
            n = 10
            nEcoregions.append(n)
            # print(n)
            bounds = histedges_equalN(raw, n)

            for i in range(n):
                lower, upper = bounds[i], bounds[i+1]
                locs = (plantClimateQuadrant>=lower)&(plantClimateQuadrant<upper)
                plantClimate_seg.append(np.nanmean(plantClimateQuadrant[locs]))
                areas.append(locs) 
    elif pptize:
        for region in ppt_dict.keys():
            # print(region)
            plantClimateQuadrant = pptizeUSA(plantClimateMap, area = region)
            raw = plantClimateQuadrant.flatten()
            raw = raw[~np.isnan(raw)]
            # n = int(len(raw)/1e4) # equal area of point for all ecoregions
            n = 10
            nEcoregions.append(n)
            # print(n)
            bounds = histedges_equalN(raw, n)

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
                
    return np.array(plantClimate_seg), np.array(areas), np.cumsum(nEcoregions)

def clean (ba, vpd):
    valid = ba>0
    ba = ba[valid]
    vpd = vpd[valid]
    return ba, vpd
    
def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))

    
    
def segregate_fireClimate(areas):
    r2 ,coefs, stderrors =[],[], []
    ctr = 0
    colors = ["#1565c0","#f2d600"]

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
           
            slope, intercept, r_value, p_value, std_err = stats.linregress(vpd,ba)
            r2.append(r_value**2)
            coefs.append(slope)
            stderrors.append(std_err)
            
            # fig_, ax_ =plt.subplots(figsize = (2,2))
            
            # ax_.set_ylabel("Burned area (km$^2$)")
            # ax_.set_xlim(21, 27) ## should be before plotting
            # sns.regplot(vpd, ba, color = "k", truncate = False,\
            #             scatter_kws ={'s':30,"edgecolor":"grey"},\
            #         line_kws = {"color":"k"}, ax =ax_)
            # ax_.set_ylim(0,4000)
            # ax_.set_xticks([21, 24, 27])
            # ax_.set_xlabel("VPD (hPa)")

    return np.array(r2),np.array(coefs), np.array(stderrors)

def segregate_ndviPpt(areas):
    r2 ,coefs, stderrors =[],[], []
    for area in areas:
        ndvi, ppt = [],[]
        for i in range(19):
            filename = os.path.join(dir_root, "data","precip_annual_oct_mar","%d.tif"%i)
            ds = gdal.Open(filename)
            ppt_ = np.nanmean(np.array(ds.GetRasterBand(1).ReadAsArray())[area==True])
            ppt.append(ppt_)
            
            filename = os.path.join(dir_root, "data","ndvi_annual_apr_sep","%d.tif"%(i+1)) # next year
            ds = gdal.Open(filename)
            ndvi_ = np.nanmean(np.array(ds.GetRasterBand(1).ReadAsArray())[area==True])
            ndvi.append(ndvi_)
            
        ppt = np.array(ppt)
        ndvi = np.array(ndvi)


        slope, intercept, r_value, p_value, std_err = stats.linregress(ppt,ndvi)
        r2.append(r_value**2)
        coefs.append(slope)
        stderrors.append(std_err)
        
        # fig, ax = plt.subplots(figsize = (3,3))
        # ax.scatter(ppt, ndvi)
        # ax.set_xlabel("ppt (mm)")
        # ax.set_ylabel("NDVI")
        # ax.annotate("r$^2$ = %0.2f"%r_value**2, (0.1,0.9),xycoords = "axes fraction",ha = "left", va = "top")
        # plt.show()
            

    return np.array(r2),np.array(coefs), np.array(stderrors)

def segregate_ndviBa(areas):
    r2 ,coefs, stderrors =[],[], []
    for area in areas:
        ba, ndvi = [],[]
        for i in range(19):
            filename = os.path.join(dir_root, "data","ba_vpd","%d.tif"%i)
            ds = gdal.Open(filename)
            ba_ = np.nansum((np.array(ds.GetRasterBand(2).ReadAsArray())>0).astype(int)[area==True])*16 #km sq.
            ba.append(ba_)
            
            filename = os.path.join(dir_root, "data","ndvi_annual_apr_sep","%d.tif"%(i)) # next year
            ds = gdal.Open(filename)
            ndvi_ = np.nanmean(np.array(ds.GetRasterBand(1).ReadAsArray())[area==True])
            ndvi.append(ndvi_)
        
        ba = np.array(ba)
        ndvi = np.array(ndvi)
        ba, ndvi = clean(ba, ndvi)
        
        if len(ba)<=8:
            r2.append(np.nan)
            coefs.append(np.nan)
        else:
            # ba = np.log10(ba)
            slope, intercept, r_value, p_value, std_err = stats.linregress(ndvi,ba)
            r2.append(r_value**2)
            coefs.append(slope)
            stderrors.append(std_err)
            
            # fig, ax = plt.subplots(figsize = (3,3))
            # ax.scatter(ndvi, ba)
            # ax.set_xlabel("NDVI")
            # ax.set_ylabel("BA (km$^2$)")
            # ax.annotate("r$^2$ = %0.2f"%r_value**2, (0.1,0.9),xycoords = "axes fraction",ha = "left", va = "top")
            # plt.show()
            

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


def landcoverizeUSA(array, area = None):
    array = array.copy()
    mask = np.ones_like(array, np.bool)
    
    mask[np.where(np.isin(landcover,area))] = 0
    array[mask] = np.nan
    # sub = array[mask]
    # sub = array[areas_dict[area][0,0]:areas_dict[area][1,0],areas_dict[area][0,1]:areas_dict[area][1,1]]
    return array

def pptizeUSA(array, area = None):
    array = array.copy()
    mask = np.ones_like(array, np.bool)
    
    mask[np.where((ppt>=ppt_dict[area][0])&(ppt<=ppt_dict[area][1]))] = 0
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
nLocal = 15
nGlobal = nLocal*len(ecoregions_dict)
# for hr in ["1000hr", "100hr"]:
#     for var in ["coefAbsSum", "coefSum","r2"]:
    
# for norm in ["no_norm","lfmc_norm","dfmc_norm","lfmc_dfmc_norm"]:
    # for var in ["coefSum","coefPositiveSum","coefAbsSum","r2"]    :
plantClimatePath = os.path.join(dir_root, "data","arr_pixels_%s"%folder,"lfmc_dfmc_%s_lag_%d_%s_%s_%s.tif"%(hr,lag,norm,coefs_type,var))
# plantClimatePath = os.path.join(dir_root, "data","mean","vpd_mean.tif")
# filename = os.path.join(dir_root, "data","mean","ndvi_mean.tif")
# filename = os.path.join(dir_root, "data","mean","vpdStd.tif")
# filename = os.path.join(dir_root, "data","mean","fireSeasonLength.tif")
filename = os.path.join(dir_root, "data","mean","vpd_mean.tif")

plantClimate_seg,areas, nEcoregions = segregate_plantClimate(filename, n = nLocal, \
                                    binning = "equal_area", localize = False,\
                                ecoregionalize = False, landcoverize = False,\
                                    pptize = False)
r2,coefs, stderrors = segregate_fireClimate(areas)

df = pd.DataFrame({"x":plantClimate_seg,"y":coefs})
# np.save(os.path.join(dir_root, "data","pws_bins","PWS_bin_1.npy"), areas[0])
# np.save(os.path.join(dir_root, "data","pws_bins","PWS_bin_15.npy"), areas[-1])

# %% main plot

slope, intercept, r_value, p_value, std_err = stats.linregress(plantClimate_seg,coefs)
print(p_value)

fig, ax = plt.subplots(figsize = (4,4))
# ax.set_xlim(0,2)
# ax.set_ylim(200,900)
sns.regplot(x=plantClimate_seg, y=coefs,ax=ax, ci = 99.5, color = "grey", seed = 0)

ax.errorbar(plantClimate_seg, coefs, yerr = stderrors, color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 1,capsize = 3)
ax.scatter(plantClimate_seg, coefs, s = 80, color = "k", edgecolor = "grey")
ax.scatter(plantClimate_seg[0], coefs[0], s = 80, color = "#1565c0", edgecolor = "grey")
ax.scatter(plantClimate_seg[-1], coefs[-1], s = 80, color = "#f2d600", edgecolor = "grey")

    
ax.set_xlabel(r"Plant-water sensitivity (PWS)")
ax.set_ylabel(r"$\rm  \frac{d(Burned\ area)}{d(VPD)}$                ", fontsize = 16)
fig.text(x = -0.05 ,y = 0.56, s = r'(km$^2$ hPa$^{-1}$)',rotation = 90)
# ax.set_ylim(100,900)
# ax.yaxis.set_major_locator(MultipleLocator(200))
# ax.set_yticks(np.linspace(100,900,5))
ax.annotate("R$^2$=%0.2f\n$p$<0.0001"%(r_value**2),xycoords = "axes fraction",ha = "left", va = "top",xy =(0.1,0.9))

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()


def bootstrap(x, y, yerr, samples = 100):
    yboot = np.random.normal(loc = np.array(y), scale = yerr, size = (samples, len(y))).flatten(order = "F")
    xboot = np.repeat(x, samples)
    
    return np.array(xboot), yboot
