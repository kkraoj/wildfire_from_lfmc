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

def segregate_plantClimate(plantClimatePath, n = 20, binning = "equal_area", 
                           localize = False, ecoregionalize = False, 
                           landcoverize = False, pptize = False,
                           control_for_vegetation = True):
    ds = gdal.Open(plantClimatePath)
    plantClimateMap = np.array(ds.GetRasterBand(1).ReadAsArray())
    ##FOR R2 MAP ONLY. Delete Otherwise
    # plantClimateMap[plantClimateMap<0] = np.nan
    plantClimate_seg, areas, nEcoregions = [],[], [0]
            
    if control_for_vegetation:
        ds = gdal.Open(r"D:\Krishna\projects\wildfire_from_lfmc\data\mean\landcover.tif")
        lcMap = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(float)

        mask = np.ones_like(lcMap, int)
        mask[lcMap>180]=0
        plantClimateMap[mask==0] = np.nan
            
    if binning == "equal_width":
        minVal , maxVal = np.nanmin(plantClimateMap), np.nanmax(plantClimateMap)
        bounds = np.linspace(minVal, maxVal, n+1)
        
    elif binning == "equal_area":
        raw = plantClimateMap.flatten()
        raw = raw[~np.isnan(raw)]
        bounds = histedges_equalN(raw, n)
    elif binning == "bounds":
        bounds = [0,1,1.5,4]

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

    
    
def segregate_fireClimate(areas, save_name, save = False):
    r2 ,coefs, stderrors =[],[], []
    ctr = 0
    colors = ["#1565c0","#f2d600"]
    # colors = ["#006600","#ad1457"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (88/25.4,48/25.4), sharey = True)

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
            print(p_value)
            
            # fig_, ax_ =plt.subplots(figsize = (2,2))
            
            # ax_.set_ylabel("Burned area (km$^2$)")
            # ax_.set_xlim(21, 27) ## should be before plotting
            # sns.regplot(vpd, ba, color = "k", truncate = False,\
            #             scatter_kws ={'s':30,"edgecolor":"grey"},\
            #         line_kws = {"color":"k"}, ax =ax_)
            # ax_.set_ylim(0,4000)
            # ax_.set_xticks([21, 24, 27])
            # ax_.set_xlabel("VPD (hPa)")
                
            if ctr==0:
                # ax1.set_xlabel("VPD (hPa)")
                
                ax1.set_ylabel("Burned area (km$^2$)")
                ax1.set_xlim(21, 27) ## should be before plotting
                sns.regplot(vpd, ba, color = colors[0], truncate = False,\
                            scatter_kws ={'s':30,"edgecolor":"grey"},\
                        line_kws = {"color":"k"}, ax =ax1)
                ax1.set_ylim(0,4000)
                ax1.set_xticks([21, 24, 27])
                ax1.set_yticklabels(["0k","2k","4k"])
                ax1.annotate(
                    r"$\rm \frac{d(BA)}{d(VPD)}\approx$350 km$^2$hPa$^{-1}$"%slope,\
                    (0.02,1.01),xycoords = "axes fraction",ha = "left", \
                        fontsize = 8, va = "top")
                ax1.annotate(
                    r"PWS$\leq$0.3",\
                    (0.5,1.05),xycoords = "axes fraction",ha = "center", \
                        weight = "bold")
                
            elif ctr==14:
                # ax2.set_xlabel("VPD (hPa)")
                ax2.set_ylabel("")
                ax2.set_xlim(21, 27) ## should be before plotting
                sns.regplot(vpd, ba, color = colors[1], truncate = False,\
                            scatter_kws ={'s':30,"edgecolor":"grey"},\
                        line_kws = {"color":"k"}, ax =ax2)
                # ax.set_ylim(0,3000)
                ax2.set_xticks([21, 24, 27])
                ax2.annotate(r"$\rm \frac{d(BA)}{d(VPD)}\approx$700 km$^2$hPa$^{-1}$"%slope,\
                    (0.02,1.01),xycoords = "axes fraction",ha = "left", \
                       fontsize = 8, va = "top")
                ax2.annotate(r"PWS$\geq$1.7",\
                    (0.5,1.05),xycoords = "axes fraction",ha = "center", \
                         weight = "bold")
                
        ctr+=1
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.text(x = 0.55 ,y = 0.1, s = r'Vapor pressure deficit (VPD) (hPa)', \
             ha ="center",va="top")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)

    if save:
        plt.savefig(save_name)
    plt.show()
    # exit
    return np.array(r2),np.array(coefs), np.array(stderrors)

def segregate_fireClimate_normalizedbyBA(areas):
    r2 ,coefs, stderrors =[],[], []
    ctr = 0
    colors = ["#1565c0","#f2d600"]
    colors = ["#006600","#ad1457"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (3.6,1.6), sharey = True)

    for area in areas:
        ba, vpd = [],[]
        for i in range(19):
            filename = os.path.join(dir_root, "data","ba_vpd","%d.tif"%i)
            ds = gdal.Open(filename)
            ba_ = np.nansum((np.array(ds.GetRasterBand(2).ReadAsArray())>0).astype(int)[area==True])*16 #km sq.
            ba.append(ba_)
            vpd.append(np.nanmean(np.array(ds.GetRasterBand(1).ReadAsArray())[area==True]))
        
        ba = np.array(ba)
        ba = ba/(np.min(ba)+1e-4)
        # NORMALIZED by the MIN BA!
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
                
            if ctr==0:
                # ax1.set_xlabel("VPD (hPa)")
                
                ax1.set_ylabel("Burned area (km$^2$)")
                ax1.set_xlim(21, 27) ## should be before plotting
                sns.regplot(vpd, ba, color = colors[0], truncate = False,\
                            scatter_kws ={'s':30,"edgecolor":"grey"},\
                        line_kws = {"color":"k"}, ax =ax1)
                ax1.set_ylim(0,4000)
                ax1.set_xticks([21, 24, 27])
  
                ax1.annotate(
                    r"$\rm \frac{d(BA)}{d(VPD)}\approx$350 km$^2$hPa$^{-1}$"%slope,\
                    (0.02,1.01),xycoords = "axes fraction",ha = "left", \
                        fontsize = 8, va = "top")
                ax1.annotate(
                    r"PWS$\leq$0.3",\
                    (0.5,1.05),xycoords = "axes fraction",ha = "center", \
                        fontsize = 9 , weight = "bold")
                
            elif ctr==14:
                # ax2.set_xlabel("VPD (hPa)")
                ax2.set_ylabel("")
                ax2.set_xlim(21, 27) ## should be before plotting
                sns.regplot(vpd, ba, color = colors[1], truncate = False,\
                            scatter_kws ={'s':30,"edgecolor":"grey"},\
                        line_kws = {"color":"k"}, ax =ax2)
                # ax.set_ylim(0,3000)
                ax2.set_xticks([21, 24, 27])
                ax2.annotate(r"$\rm \frac{d(BA)}{d(VPD)}\approx$700 km$^2$hPa$^{-1}$"%slope,\
                    (0.02,1.01),xycoords = "axes fraction",ha = "left", \
                        fontsize = 8, va = "top")
                ax2.annotate(r"PWS$\geq$1.7",\
                    (0.5,1.05),xycoords = "axes fraction",ha = "center", \
                        fontsize = 9, weight = "bold")
                
        ctr+=1
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.text(x = 0.5 ,y = -0.07, s = r'Vapor pressure deficit (VPD) (hPa)', ha ="center",va="top")

    plt.show()

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
# nLocal = 3
nGlobal = nLocal*len(ecoregions_dict)
# for hr in ["1000hr", "100hr"]:
#     for var in ["coefAbsSum", "coefSum","r2"]:
    
# for norm in ["no_norm","lfmc_norm","dfmc_norm","lfmc_dfmc_norm"]:
    # for var in ["coefSum","coefPositiveSum","coefAbsSum","r2"]    :
plantClimatePath = os.path.join(dir_root, "data","arr_pixels_%s"%folder,"lfmc_dfmc_%s_lag_%d_%s_%s_%s.tif"%(hr,lag,norm,coefs_type,var))
# plantClimatePath = os.path.join(dir_root, "data","mean","vpd_mean.tif")
ndviPath = os.path.join(dir_root, "data","mean","ndvi_mean.tif")

plantClimate_seg,areas, nEcoregions = segregate_plantClimate(plantClimatePath, n = nLocal, \
                                    binning = "equal_area", localize = False,\
                                ecoregionalize = False, landcoverize = False,\
                                    pptize = False)
r2,coefs, stderrors = segregate_fireClimate(areas, \
                                            save = True, \
    save_name = r"C:\Users\kkrao\Dropbox\meetingsKrishna\Figures\wildfire_from_lfmc\fig1c.pdf")
# _,coefs2, stderrors2 = segregate_fireClimate_normalizedbyBA(areas)

df = pd.DataFrame({"x":plantClimate_seg,"y":coefs})
# np.save(os.path.join(dir_root, "data","pws_bins","PWS_bin_1.npy"), areas[0])
# np.save(os.path.join(dir_root, "data","pws_bins","PWS_bin_15.npy"), areas[-1])

# %% main plot

slope, intercept, r_value, p_value, std_err = stats.linregress(plantClimate_seg,coefs)
print(p_value)

fig, ax = plt.subplots(figsize = (98/25.4,98/25.4))
ax.set_xlim(0,2)
# ax.set_ylim(200,900)
sns.regplot(x=plantClimate_seg, y=coefs,ax=ax, ci = 99.5, color = "grey", seed = 0)

ax.errorbar(plantClimate_seg, coefs, yerr = stderrors, color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 1,capsize = 3)
ax.scatter(plantClimate_seg, coefs, s = 80, color = "k", edgecolor = "grey")
ax.scatter(plantClimate_seg[0], coefs[0], s = 80, color = "#1565c0", edgecolor = "grey")
ax.scatter(plantClimate_seg[-1], coefs[-1], s = 80, color = "#f2d600", edgecolor = "grey")

    
ax.set_xlabel(r"Plant-water sensitivity (PWS)")
ax.set_ylabel(r"$\rm  \frac{d(Burned\ area)}{d(VPD)}$                ", fontsize = 16)
ax.annotate( r'(km$^2$ hPa$^{-1}$)',xy = (-0.31,0.64),xycoords = "axes fraction",rotation = 90)
ax.set_ylim(100,900)
# ax.yaxis.set_major_locator(MultipleLocator(200))
ax.set_yticks(np.linspace(100,900,5))
ax.annotate("R$^2$=%0.2f\n$p$<0.0001"%(r_value**2),xycoords = "axes fraction",\
            ha = "left", va = "top",xy =(0.1,0.9))
ax.annotate("Relationship between wildfire\nvulnerability and PWS",\
                    (0.5,1.05),xycoords = "axes fraction",ha = "center", \
                         weight = "bold")
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(r"C:\Users\kkrao\Dropbox\meetingsKrishna\Figures\wildfire_from_lfmc\fig1a.pdf")
plt.show()


def bootstrap(x, y, yerr, samples = 100):
    yboot = np.random.normal(loc = np.array(y), scale = yerr, size = (samples, len(y))).flatten(order = "F")
    xboot = np.repeat(x, samples)
    
    return np.array(xboot), yboot

#%% BA_normalized/VPD vs PWS


# slope, intercept, r_value, p_value, std_err = stats.linregress(plantClimate_seg,coefs2)
# print(p_value)

# fig, ax = plt.subplots(figsize = (4,4))
# ax.set_xlim(0,2)
# # ax.set_ylim(200,900)
# sns.regplot(x=plantClimate_seg, y=coefs2,ax=ax, ci = 99.5, color = "grey", seed = 0)

# ax.errorbar(plantClimate_seg, coefs2, yerr = stderrors2, color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 1,capsize = 3)
# ax.scatter(plantClimate_seg, coefs2, s = 80, color = "k", edgecolor = "grey")
# # ax.scatter(plantClimate_seg[0], coefs[0], s = 80, color = "#1565c0", edgecolor = "grey")
# # ax.scatter(plantClimate_seg[-1], coefs[-1], s = 80, color = "#f2d600", edgecolor = "grey")

    
# ax.set_xlabel(r"Plant-water sensitivity (PWS)")
# ax.set_ylabel(r"$\rm  \frac{d(Burned\ area_{norm})}{d(VPD)\times}$                    ", fontsize = 16)
# fig.text(x = -0.05 ,y = 0.56, s = r'(hPa$^{-1}$)',rotation = 90)
# # ax.set_ylim(100,900)
# # ax.yaxis.set_major_locator(MultipleLocator(200))
# # ax.set_yticks(np.linspace(100,900,5))
# ax.annotate("R$^2$=%0.2f\n$p$<0.0001"%(r_value**2),xycoords = "axes fraction",ha = "left", va = "top",xy =(0.1,0.9))

# # Hide the right and top spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.show()

#%% map of areas

# gt = ds.GetGeoTransform()
# # map_kwargs = dict(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
# #         projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

# map_kwargs = dict(llcrnrlon=-128,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
#         projection='cyl')

# cmap = ListedColormap(["#1565c0"])
# scatter_kwargs = dict(cmap = cmap,vmin = -0.1, vmax = 0.2)

# fig, ax = plt.subplots(figsize = (3,3), frameon = False)
# fig, ax, m, plot = plotmap(gt = gt, var = np.where(areas[0]==1, 1, np.nan),\
#                     map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, \
#                 marker_factor = .1, 
#                       fill = "white",background="white",fig = fig,ax=ax,
#                       shapefilepath = 'D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
#                   shapefilename ='states')
# scatter_kwargs=dict(cmap = ListedColormap(["#f2d600"]))
# fig, ax, m, plot = plotmap(gt = gt, var = np.where(areas[-1]==1, 1, np.nan),\
#             map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, \
#                 marker_factor = .1, 
#                       fill = "white",background="white",fig = fig,ax=ax,
#                       shapefilepath = 'D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
#                   shapefilename ='states')
# ax.axis('off')


# fig, ax = plt.subplots(figsize = (3,3), frameon = False)
# cmap = ListedColormap(["#1565c0"])
# ax.imshow(np.where(areas[0]==1, 1, np.nan), cmap = cmap)
# cmap = ListedColormap(["#f2d600"])
# ax.imshow(np.where(areas[-15]==1, 1, np.nan), cmap = cmap)
# ax.axis('off')

#%% map of all 15 areas (not just 2)
# for i in range(areas.shape[0]):
#     area = areas[i]
#     fig, ax = plt.subplots(figsize = (5,5))
#     ax.imshow(area,cmap = "Greys")
#     ax.set_title(i)
#     ax.axis('off')
    
# fig, ax = plt.subplots(figsize = (4.8,4.8),frameon=False)
# fig, ax, m, plot = plotmap(gt = gt, var = np.where(areas[0]==1, np.nan, np.nan),\
#                     map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, \
#                 marker_factor = .1, 
#                       fill = "white",background="white",fig = fig,ax=ax,
#                       shapefilepath = 'D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
#                   shapefilename ='states')
# ax.axis('off')
# plt.tight_layout()

#%% segregated plot
ctr=0
colors = ["seagreen","darkgoldenrod","limegreen"]
fig, axs = plt.subplots(1, 3,figsize = (9,3), sharey = True)
for quadrant in lc_dict.keys():
    # x, y = bootstrap(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], stderrors[nEcoregions[ctr]:nEcoregions[ctr+1]])
    
    x = df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]]
    y = df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]]
    print(quadrant)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    print(r_value**2)
    print(p_value)
    sns.regplot(x = x, y = y,ax=axs[ctr], color = colors[ctr],scatter_kws={'s':0}, ci = 99.5)
    axs[ctr].errorbar(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], yerr = stderrors[nEcoregions[ctr]:nEcoregions[ctr+1]], color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 2)
    axs[ctr].scatter(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], s = 50, edgecolor = "grey",label = quadrant, color = colors[ctr])
    
    # ax.set_xlabel(r"%s, %s"%(var, hr))
    # ax.set_xlabel(r"$\sum_{i=1}^5\frac{d(LFMC_t)}{d(DFMC_{t-i})}$")
    axs[ctr].set_xlabel(r"Plant-water sensitivity (PWS)")
    axs[ctr].set_ylabel("")
    # ax.set_title(norm)
    axs[ctr].set_xlim(0,2)
    axs[ctr].legend(loc = "upper left")
    
    
    # Hide the right and top spines
    axs[ctr].spines['right'].set_visible(False)
    axs[ctr].spines['top'].set_visible(False)
    
    # Only show ticks on the left and bottom spines
    # axs[ctr].yaxis.set_ticks_position('left')
    # axs[ctr].xaxis.set_ticks_position('bottom')

    ctr+=1
axs[0].set_ylim(0,550)
axs[0].set_ylabel(r"$\rm  \frac{d(Burned\ area)}{d(VPD)}$                ", fontsize = 16)
fig.text(x = 0.045 ,y = 0.6, s = r'(km$^2$ hPa$^{-1}$)',rotation = 90)

# axs[ctr].legend(bbox_to_anchor=[1, 1],loc = "upper left")
    

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
# axs[0].set_ylabel(r"PWS")

# ax.set_title(norm)
# plt.legend(bbox_to_anchor=[1, 1],loc = "upper left")


#%% only fire plot
# bas,stderrors = segregate_fire(areas)
# df = pd.DataFrame({"x":plantClimate_seg,"y":bas})

# ctr=0
# n=10

# fig, ax = plt.subplots(figsize = (3,3))
# sns.regplot(x=plantClimate_seg, y=bas,ax=ax, color = "darkgrey")

# # ax.errorbar(plantClimate_seg, bas, yerr = stderrors, color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 2)
# ax.scatter(plantClimate_seg, bas, s = 50, color = "k", edgecolor = "grey")
# ax.set_xlabel(r"%s, %s"%(var, hr))
# ax.set_xlabel(r"PWS")
# ax.set_xlim(0,2)
# # ax.set_xlabel(r"Mean VPD")
# ax.set_ylabel(r"BA (km$^2$)")
# # ax.set_title(norm)

# plt.show()



#%% only fire plot segregated

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
    
#%% map of plantclimate
# ds = gdal.Open(plantClimatePath)
# plantClimateMap = np.array(ds.GetRasterBand(1).ReadAsArray())

# plantClimateQuadrant = ecoregionalizeUSA(plantClimateMap, area = 1)
# fig, ax = plt.subplots(figsize = (3,3))
# ax.imshow(plantClimateQuadrant)
# plt.axis('off')

# ds = gdal.Open(r"D:\Krishna\projects\wildfire_from_lfmc\data\mean\ndvi_mean.tif")
# ndviMap = np.array(ds.GetRasterBand(1).ReadAsArray())

# ndviQuadrant = ecoregionalizeUSA(ndviMap, area = 1)
# fig, ax = plt.subplots(figsize = (3,3))
# ax.imshow(ndviQuadrant)
# plt.axis('off')


# fig, ax = plt.subplots(figsize = (3,3))
# ax.scatter(plantClimateQuadrant.flatten(),ndviQuadrant.flatten(),  s = 0.01, alpha = 0.1, color = "k")
# ax.set_ylabel("NDVI")
# # ax.set_xlim(0.6,1.2)
# ax.set_xlabel("PWS")



#%% plantclimate and landcover


# ds = gdal.Open(plantClimatePath)
# plantClimateMap = np.array(ds.GetRasterBand(1).ReadAsArray())

# plantClimateQuadrant = ecoregionalizeUSA(plantClimateMap, area = 8)
# fig, ax = plt.subplots(figsize = (3,3))
# ax.imshow(plantClimateQuadrant)
# plt.axis('off')

# ds = gdal.Open(r"D:\Krishna\projects\wildfire_from_lfmc\data\mean\landcover.tif")
# lcMap = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(float)

# mask = np.ones_like(lcMap, int)
# mask[np.isin(lcMap, 
#     [item for sublist in list(lc_dict.values()) for item in sublist])] = 0
# lcMap[mask] = np.nan

# lcQuadrant = ecoregionalizeUSA(lcMap, area = 8)
# fig, ax = plt.subplots(figsize = (3,3))
# plot = ax.imshow(lcQuadrant, cmap = "Accent")
# plt.axis('off')
# plt.colorbar(plot)


# # fig, ax = plt.subplots(figsize = (3,3))
# # ax.scatter(plantClimateQuadrant.flatten(),lcQuadrant.flatten(),  s = 0.01, alpha = 0.1, color = "k")
# # ax.set_ylabel("LC")

# # # ax.set_xlim(0.6,1.2)
# # ax.set_xlabel("PWS")

# ctr=20
# fig, ax = plt.subplots(figsize = (3,3))
# for area in areas[20:]:
#     lc = stats.mode(lcMap[area==True], axis = None).mode[0]
#     pc = plantClimate_seg[ctr]
#     ax.scatter(pc,lc,color = u'#2ca02c')
#     ctr+=1
# ax.set_xlim(0,2)
# ax.set_xlabel("plantClimate")
# ax.set_ylabel("Majority land cover")
# ax.set_yticks([130,140])
# ax.set_yticklabels(["shrubs","grass"])

    
#%% ndvi prior precip

# r2,coefs, stderrors = segregate_ndviPpt(areas)

# df = pd.DataFrame({"x":plantClimate_seg,"y":r2})

# ctr=0
# fig, ax = plt.subplots(figsize = (3,3))
# for quadrant in lc_dict.keys():

#     # ax.errorbar(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], yerr = stderrors[nEcoregions[ctr]:nEcoregions[ctr+1]], color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 2)
#     ax.scatter(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], s = 50, edgecolor = "grey",label = quadrant, color = colors[ctr])
#     ctr+=1
# ax.set_xlabel(r"Plant-water sensitivity (PWS)")
# ax.set_ylabel(r"r$^2$(PPT$_{t-1}$, NDVI$_t$)")

# plt.legend(bbox_to_anchor=[1, 1],loc = "upper left")


# df = pd.DataFrame({"x":plantClimate_seg,"y":coefs})

# ctr=0
# fig, ax = plt.subplots(figsize = (3,3))
# for quadrant in lc_dict.keys():
#     ax.errorbar(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], \
#                 df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], \
#                     yerr = stderrors[nEcoregions[ctr]:nEcoregions[ctr+1]], \
#                         color = "lightgrey", zorder = -1, linewidth = 0, \
#                             elinewidth = 2)
#     ax.scatter(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], \
#                 df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], s = 50, \
#                     edgecolor = "grey",label = quadrant, color = colors[ctr])
#     ctr+=1
# ax.set_xlabel(r"Plant-water sensitivity (PWS)")

# ax.set_ylabel(r"$\frac{d\ NDVI_{t}}{d\ \rm{precipitation}_{t-1}}$")

# plt.legend(bbox_to_anchor=[1, 1],loc = "upper left")

# plt.show()

#%% dNDVI/dPPT vs. PWS on precip gradients 

# r2,coefs, stderrors = segregate_ndviPpt(areas)

# df = pd.DataFrame({"x":plantClimate_seg,"y":coefs})
# colors =sns.color_palette("Blues", n_colors = 3).as_hex()
# ctr=0
# fig, ax = plt.subplots(figsize = (3,3))
# for quadrant in ppt_dict.keys():

#     ax.errorbar(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], \
#                 df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], \
#                     yerr = stderrors[nEcoregions[ctr]:nEcoregions[ctr+1]], \
#                         color = "lightgrey", zorder = -1, linewidth = 0, \
#                             elinewidth = 2)
#     ax.scatter(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], \
#                 df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], s = 50, \
#                     edgecolor = "grey",label = quadrant, color = colors[ctr])
#     ctr+=1
# ax.set_xlabel(r"PWS")
# ax.set_xlim(0,2)
# ax.set_ylabel(r"$\frac{dNDVI_{t}}{dPPT_{t-1}}$")

# plt.legend(bbox_to_anchor=[1, 1],loc = "upper left")

# plt.show()

#%% ndvi ba

# r2,coefs, stderrors = segregate_ndviBa(areas)
# df = pd.DataFrame({"x":plantClimate_seg,"y":r2})

# ctr=0
# fig, ax = plt.subplots(figsize = (3,3))
# for quadrant in ecoregions_dict.keys():

#     # ax.errorbar(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], yerr = stderrors[nEcoregions[ctr]:nEcoregions[ctr+1]], color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 2)
#     ax.scatter(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], s = 50, edgecolor = "grey",label = ecoregions_dict[quadrant])
#     ctr+=1
# ax.set_xlabel(r"PWS")

# ax.set_ylabel(r"r$^2$(NDVI, BA)")

# plt.legend(bbox_to_anchor=[1, 1],loc = "upper left")


# df = pd.DataFrame({"x":plantClimate_seg,"y":coefs})

# ctr=0
# fig, ax = plt.subplots(figsize = (3,3))
# for quadrant in ecoregions_dict.keys():

#     ax.errorbar(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], yerr = stderrors[nEcoregions[ctr]:nEcoregions[ctr+1]], color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 2)
#     ax.scatter(df.x.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], df.y.iloc[nEcoregions[ctr]:nEcoregions[ctr+1]], s = 50, edgecolor = "grey",label = ecoregions_dict[quadrant])
#     ctr+=1
# ax.set_xlabel(r"PWS")

# ax.set_ylabel(r"$\frac{dBA}{dNDVI}$")

# plt.legend(bbox_to_anchor=[1, 1],loc = "upper left")


#%% plot landcover

# plt.imshow(landcover)

