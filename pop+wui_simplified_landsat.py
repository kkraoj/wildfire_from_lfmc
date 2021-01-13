# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:31:21 2020

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
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from plotmap import plotmap
from scipy.ndimage.filters import gaussian_filter





def subset_CA(wui):
    wuiCA = wui[200:450,:300]
    return wuiCA
    
sns.set(font_scale = 1.1, style = "ticks")
wuiNames = ["urban2001NeighborsResampledGee.tif","urban2016NeighborsResampledGee.tif"]
# popNames = ["pop2000.tif","pop2010.tif"]
popNames = ["worldPopDensity2000.tif","worldPopDensity2015.tif"]

res = 3.5932611
plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","lfmc_dfmc_100hr_lag_6_lfmc_dfmc_norm_positive_coefSum.tif")
ds = gdal.Open(plantClimatePath)
plantClimate = np.array(ds.GetRasterBand(1).ReadAsArray())


#%% % absolute population timeseries split by pc quantiles

ctr = 0
wuiThresh = 0.1
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
    wui = wui>wuiThresh

   
    # # pc[pc<=np.nanquantile(pc,0.75)] = np.nan
    # plt.hist(toplot)
    # plt.show()
    # fig, ax = plt.subplots(figsize = (3,3))
    # vmin = 1
    # vmax = 20000
    # ax.imshow(toplot,cmap = "PuRd",vmin = vmin, vmax = vmax,  norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    
    # ax.tick_params(
    # axis='both',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    # left=False,      # ticks along the bottom edge are off
    # labelleft=False,
    # labelbottom=False) # labels along the bottom edge are off
    # plt.show()
    
    pop = pop[wui==1]
    pc = plantClimate[wui==1]
    df = pd.DataFrame({"pc":pc,"pop":pop})
    df.dropna(inplace = True)
    ctr+=1
    
    wui = wui*1.0
        
    if wuiName == "urban2001NeighborsResampledGee.tif":
        wui1990 = wui.copy()
    else:
        wui2010 = wui.copy()
    # break

wuiDiff = wui2010-wui1990
wuiDiff[wuiDiff<0] = 0
wuiDiff[wuiDiff<1] = np.nan
gt = ds.GetGeoTransform()
map_kwargs = dict(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

scatter_kwargs = dict(cmap = "Oranges",vmin = 0, vmax = 1,alpha = 0.05)
fig, ax, m, plot = plotmap(gt = gt, var = wuiDiff,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 0.2, 
                      fill = "white",background="white",
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

data = np.nan_to_num(plantClimate,nan = -9999)
data = gaussian_filter(data, sigma = 3,order = 0)
data[data<0] = np.nan
# plt.imshow(data,cmap = "viridis")

fig, ax, m, plot = plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig=fig, ax=ax,contour = True,contourLevel = np.nanquantile(data,0.9),
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')
plt.show()

scatter_kwargs = dict(cmap = "viridis",vmin = 0, vmax = 2,alpha = 1)
fig, ax, m, plot = plotmap(gt = gt, var = plantClimate,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 0.2, 
                      fill = "white",background="white",
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

data = np.nan_to_num(wuiDiff,nan = -9999)
data = gaussian_filter(data, sigma = 3,order = 0)
# data[data<0] = np.nan
# plt.imshow(data,cmap = "viridis")

fig, ax, m, plot = plotmap(gt = gt,  contourColor = "orange",var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig=fig, ax=ax,contour = True,contourLevel = np.nanquantile(data,0.9),
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')
plt.show()


    
#%% growth rates for 10 bins


nbins = 15
cmap = plt.get_cmap('viridis',nbins)    # PiYG
colors = [mpl.colors.rgb2hex(cmap(i))  for i in range(cmap.N)]
  
    # rgb2hex accepts rgb or rgba
_, vulLabels = pd.qcut(df['pc'],nbins, retbins = True)
vulLabels = np.round(vulLabels, 2)
# vulLabels = np.linspace(0,2,nbins+1)
ts = pd.DataFrame(columns = vulLabels[:-1], index = [1990,2010])
ctr = 0

for (wuiName, popName) in zip(wuiNames, popNames):
    
    fullfilename = os.path.join(dir_root, "data","WUI",wuiName)
    ds = gdal.Open(fullfilename)
    wui = np.array(ds.GetRasterBand(1).ReadAsArray())
    wui = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(float)
    wui[wui<0] = np.nan
    # plt.hist(wui.flatten(),bins = 100)
    # plt.show()
    
    fullfilename = os.path.join(dir_root, "data","population","gee",popName)
    ds = gdal.Open(fullfilename)
    pop = np.array(ds.GetRasterBand(1).ReadAsArray())*res**2
    ds = None
    if pop.shape[0]!=645:
        pop = pop[0:645]
    pop[pop<0] = 0
    # pop = pop*wui    
    wui = wui>wuiThresh
    # pop = subset_CA(pop)
    # wui = subset_CA(wui)
    pop = pop[wui==1]
    # pop = wui[wui==1].copy()
    # print(len(pop))
    pc = plantClimate[wui==1]
    df = pd.DataFrame({"pc":pc,"pop":pop})
    df.dropna(inplace = True)
    df['pcBin'] = pd.qcut(df.pc, nbins, labels = vulLabels[:-1])
    
    cum = df.groupby("pcBin").pop.sum()
    ts.loc[1990+ctr*20, :] = cum
    ctr+=1


density = gaussian_kde(df.pc)
xs = np.linspace(df.pc.min(),df.pc.max(),100)
density.covariance_factor = lambda : .25
density._compute_covariance()

fig, ax = plt.subplots(figsize = (1,1))
ax.plot(xs,density(xs), linewidth = 3, color = "grey")
# for q in [0.0,0.25,0.5,0.75]:
ctr=0
for q in vulLabels[:-1]:
    low = q 
    high = vulLabels[ctr+1]
    xsq = xs[(xs>=low)&(xs<=high)]
    densityq = density(xsq)
    ax.fill_between(xsq, 0, densityq, color = colors[ctr])
    ctr+=1
    
# ax.set_xlabel("Plant climate sensitivity")
ax.set_ylabel("Density")
ax.set_xticks([0,1,2])
ax.set_yticks([0,0.5,1])

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

fig, ax = plt.subplots(figsize = (3,3))

ax.bar(ts.columns,ts.diff().dropna().values.tolist()[0],align = "edge",color = colors,width = np.diff(vulLabels),edgecolor = "k")
ax.set_xlabel("PAS")
ax.set_ylabel("$\Delta$ WUI population")

ylabels = ['{:,.1f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)
ax.set_xlim(0,2.1)
# Only show ticks on the left and bottom spines
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax2 = ax.twinx()
ax2.plot(xs,density(xs), linewidth = 3, color = "grey")
# for q in [0.0,0.25,0.5,0.75]:
ctr=0
for q in vulLabels[:-1]:
    low = q 
    high = vulLabels[ctr+1]
    xsq = xs[(xs>=low)&(xs<=high)]
    densityq = density(xsq)
    ax2.fill_between(xsq, 0, densityq, facecolor = colors[ctr],  linewidth = 1.5,edgecolor = "grey")
    ctr+=1
ax2.set_ylabel("Density          ",ha ="right",color = "grey")
ax2.set_ylim(0,3.5)
ax2.set_yticks([0,0.5,1])
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax2.spines['bottom'].set_color('grey')
ax2.spines['top'].set_color('grey') 
ax2.spines['right'].set_color('grey')
ax2.spines['left'].set_color('grey')
ax2.tick_params(axis='y', colors='grey')

wui1990[wui1990<1] = np.nan
fig, ax, m, plot = plotmap(gt = gt,  var = wui1990,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 0.01, 
                      fill = "white",background="white",
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

wui2010[wui2010<1] = np.nan
fig, ax, m, plot = plotmap(gt = gt,  var = wui2010,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 0.01, 
                      fill = "white",background="white",
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

plt.show()

## 30m data m

# plantClimatePath = os.path.join(dir_root, "data","WUI","30m","urban2001mosaicNeighbors.tif")
# ds = gdal.Open(plantClimatePath)
# data = np.array(ds.GetRasterBand(1).ReadAsArray())
# print(data.sum()*30*30/1000/1000)

# plantClimatePath = os.path.join(dir_root, "data","WUI","30m","urban2016mosaicNeighbors.tif")
# ds = gdal.Open(plantClimatePath)
# data = np.array(ds.GetRasterBand(1).ReadAsArray())
# print(data.sum()*30*30/1000/1000)
