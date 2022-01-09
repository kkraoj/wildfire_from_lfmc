# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:41:53 2020

@author: kkrao
"""


import os
from init import dir_root, dir_data, dir_fig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import gdal
import matplotlib as mpl
import seaborn as sns
import scipy
from plotmap import plotmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.filters import gaussian_filter
import matplotlib.patches as patches

SAVEPLOT = True
sns.set(font_scale = 1., style = "ticks")
plt.style.use("pnas")


plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","PAS_6_jan_2021.tif")
ds = gdal.Open(plantClimatePath)
plantClimate = np.array(ds.GetRasterBand(1).ReadAsArray())

wuiNames = ["wui1990.tif","wui2010.tif"]
popNames = ["pop1990.tif","pop2010.tif"]
res = 3.5932611
plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","lfmc_dfmc_100hr_lag_6_lfmc_dfmc_norm_positive_coefSum.tif")
ds = gdal.Open(plantClimatePath)
plantClimate = np.array(ds.GetRasterBand(1).ReadAsArray())
gt = ds.GetGeoTransform()


#%% % absolute population timeseries split by pc quantiles

ctr = 0
wuiThresh = 0
for (wuiName, popName) in zip(wuiNames, popNames):
    
    fullfilename = os.path.join(dir_root, "data","WUI",wuiName)
    ds = gdal.Open(fullfilename)
    wui = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(float)
    wui[wui<0] = np.nan
    # print(np.nansum(wui))

    fullfilename = os.path.join(dir_root, "data","population","gee",popName)
    ds = gdal.Open(fullfilename)
    pop = np.array(ds.GetRasterBand(1).ReadAsArray())*res**2
    # ds = None
    if pop.shape[0]!=645:
        pop = pop[1:646]
    pop[pop<0] = 0
    # pop = subset_CA(pop)
    # wui = subset_CA(wui)
    # pop = pop*wui
    # plt.imshow(wui)
    # plt.show()
    print((pop*wui).sum()/pop.sum())
    wui = wui>wuiThresh

    if wuiName == wuiNames[0]:
        pop1990 = pop.copy()
    else:
        pop2010 = pop.copy()
    
    pop = pop[wui==1]
    pc = plantClimate[wui==1]
    df = pd.DataFrame({"pc":pc,"pop":pop})
    df.dropna(inplace = True)
    ctr+=1
    
    wui = wui*1.0
        
    if wuiName == wuiNames[0]:
        wui1990 = wui.copy()
    else:
        wui2010 = wui.copy()
    # break

wuiDiff = wui2010-wui1990
wuiDiff[wuiDiff<0] = 0
wuiDiff[wuiDiff<1] = np.nan


#%% plant climate map for blog
map_kwargs = dict(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
cmap = LinearSegmentedColormap.from_list('pinks', ['white','#cc0e74'], N=256)

fig, ax = plt.subplots(figsize = (3,3),frameon = False)
scatter_kwargs = dict(cmap = cmap,vmin = 0, vmax = 2.0,alpha = 1)
ax.axis("off")

fig, ax, m, plot = plotmap(gt = gt, var = plantClimate,map_kwargs=map_kwargs ,\
                           scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig = fig,ax=ax,
                      shapefilepath = 'D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                  shapefilename ='states')
    

scatter_kwargs = dict(cmap = "binary", vmax=1,vmin = 0)
cax = fig.add_axes([0.7, 0.45, 0.03, 0.3])
cb0 = fig.colorbar(plot,ax=ax,cax=cax,ticks = np.linspace(0,2,5))
fig, ax, m, plot = plotmap(gt = gt, var = wuiDiff,map_kwargs=map_kwargs ,\
                           scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig = fig,ax=ax,
                      shapefilepath = 'D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                  shapefilename ='states')
    

plt.tight_layout()
ax.axis("off")
fig.savefig("D:/Krishna/projects/wildfire_from_lfmc/the_conversation_blog/pws.jpg",dpi = 300)


plt.show()
