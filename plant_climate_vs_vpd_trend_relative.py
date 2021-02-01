# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:41:53 2020

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
import scipy
from plotmap import plotmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.filters import gaussian_filter



sns.set(font_scale = 1.1, style = "ticks")


fullfilename = os.path.join(dir_root, "data","mean","vpdtrend.tif")
ds = gdal.Open(fullfilename)
vpd = np.array(ds.GetRasterBand(1).ReadAsArray())

fullfilename = os.path.join(dir_root, "data","mean","vpd_mean.tif")
ds = gdal.Open(fullfilename)
vpd /= np.array(ds.GetRasterBand(1).ReadAsArray())/100

fullfilename = os.path.join(dir_root, "data","mean","landcover.tif")
ds = gdal.Open(fullfilename)
lc = np.array(ds.GetRasterBand(1).ReadAsArray())

vpd[lc==0] = np.nan

plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","PAS_6_jan_2021.tif")
ds = gdal.Open(plantClimatePath)
plantClimate = np.array(ds.GetRasterBand(1).ReadAsArray())

df = pd.DataFrame({"vpdTrend":vpd.flatten(), "sigma": plantClimate.flatten()}).dropna()

res = 100
fig, ax  = plt.subplots(figsize  = (3,3))
ax.hist2d(x = df.sigma, y = df.vpdTrend, bins=(res, res), vmax = 2e2, vmin = 0, cmap='mako')
ax.set_xlabel("PAS")
ax.set_ylabel("VPD Trend (%/yr)")
ax.set_xlim(0,2)
ax.set_ylim(-1,1)
ax.axvline(x = df.sigma.mean(), color = "lightgrey", linestyle = "--", linewidth = 2)
ax.axhline(y = df.vpdTrend.mean(), color = "lightgrey", linestyle = "--", linewidth = 2)


ax.annotate("%d %%"%(df[(df['vpdTrend']>=df['vpdTrend'].mean())&(df['sigma']>=df['sigma'].mean())].shape[0]/df.shape[0]*100),
            xy = (1,1), xycoords = "axes fraction",ha = 'right',va = 'top',color = "lightgrey")
ax.annotate("%d %%"%(df[(df['vpdTrend']<=df['vpdTrend'].mean())&(df['sigma']<=df['sigma'].mean())].shape[0]/df.shape[0]*100),
            xy = (0.1,0), xycoords = "axes fraction",ha = 'left',va = 'bottom',color = "lightgrey")

ax.annotate("%d %%"%(df[(df['vpdTrend']<=df['vpdTrend'].mean())&(df['sigma']>=df['sigma'].mean())].shape[0]/df.shape[0]*100),
            xy = (1,0), xycoords = "axes fraction",ha = 'right',va = 'bottom',color = "lightgrey")

ax.annotate("%d %%"%(df[(df['vpdTrend']>=df['vpdTrend'].mean())&(df['sigma']<=df['sigma'].mean())].shape[0]/df.shape[0]*100),
            xy = (.1,1), xycoords = "axes fraction",ha = 'left',va = 'top',color = "lightgrey")


# sns.kdeplot(data = df,
#     fill=True,cmap="mako",
# )

# geyser = sns.load_dataset("geyser")
# sns.kdeplot(data=geyser, x="waiting", y="duration")

cuts = [df.vpdTrend.min(),0,0.25,0.5,1]
bins = len(cuts)-1
colors = sns.diverging_palette(240, 10,n=(bins-1)*2).as_hex()
colors = [colors[1]]+sns.color_palette("dark:salmon_r",n_colors = bins).as_hex()[:-1]
sns.set(font_scale = 1.1, style = "ticks")
fig, ax = plt.subplots(figsize =(2,3))
n = []
for i in range(len(cuts)-1):
    minVal = cuts[i]
    maxVal = cuts[i+1]
    data = df.loc[(df.vpdTrend>=minVal)&(df.vpdTrend<maxVal)]
    n.append(len(data.sigma))
    sns.kdeplot(data = data,y="sigma",ax = ax, color = colors[i],linewidth = 3)
    
ax.annotate("n = 25k",
            xy = (0.7,0.4), ha = 'left',va = 'top',
            color = colors[0],weight = "bold",fontsize = 12)
ax.annotate("n = 89k",
            xy = (.94,1.15), ha = 'left',va = 'top',
            color = colors[1],weight = "bold",fontsize = 12)
ax.annotate("n = 155k",
            xy = (0.8,1.4), ha = 'left',va = 'top',
            color = colors[2],weight = "bold",fontsize = 12)
ax.annotate("n = 10k",
            xy = (0.35,1.8),ha = 'left',va = 'top',
            color = colors[3],weight = "bold",fontsize = 12)

ax.set_ylabel("PAS")
ax.set_xlabel("Density")
ax.set_ylim(0,2.5)
ax.set_xticks([0,0.5,1,1.5])
# ax.legend(bbox_to_anchor = [0.5,-0.2], loc = "upper center")

# %% VPD sigma box plot
ndf = pd.DataFrame(columns = range(len(cuts)-1), index = range(df.shape[0]))
for i in range(len(cuts)-1):
    minVal = cuts[i]
    maxVal = cuts[i+1]
    data = df.loc[(df.vpdTrend>=minVal)&(df.vpdTrend<maxVal)]
    ndf.loc[0:len(data.sigma)-1,i] = data.sigma.values
ndf.dropna(inplace = True,how = "all")

fig, ax = plt.subplots(figsize =(1,3))
sns.boxplot(data= ndf, ax = ax,palette = colors,saturation = 1,width = 0.8,fliersize = 0)

ax.set_ylabel("")
ax.set_xlabel("")
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim(0,2.5)




#%% VPD trend map
sns.set(font_scale = 0.8, style = "ticks")

gt = ds.GetGeoTransform()
map_kwargs = dict(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
# mycmap = sns.diverging_palette(240, 10, as_cmap=True)
colors = [colors[0]]+[colors[0]]+[colors[0]]+colors + [colors[-1]]
cmap = ListedColormap(colors)
scatter_kwargs = dict(cmap = cmap,vmin = -1, vmax = 1)
fig, ax = plt.subplots(figsize = (3,3))
fig, ax, m, plot = plotmap(gt = gt, var = vpd,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig = fig,ax=ax,
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\states",shapefilename ='states')

# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)
cax = fig.add_axes([0.68, 0.45, 0.02, 0.3])
cbar = fig.colorbar(plot, cax=cax, orientation='vertical')
cax.set_title("VPD trend\n(%/yr)")
cbar.set_ticks([-1,0,0.25,0.5,1])
scatter_kwargs = dict(cmap = "Greys",vmin = 0, vmax = 1,alpha = 0)
# data = scipy.ndimage.zoom(plantClimate,0.1)
data = np.nan_to_num(plantClimate,nan = -9999)
data = gaussian_filter(data, sigma = 3,order = 0)
data[data<0] = np.nan
# plt.imshow(data,cmap = "viridis")

fig2, ax2, m, plot = plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig=fig, ax=ax,contour = True,contourLevel = np.nanquantile(data,0.9),
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')
plt.show()
print(np.nanmean(vpd[plantClimate>=np.nanquantile(plantClimate,0.9)]))
print(np.mean(vpd))
print(ndf.mean()[:3].mean())
print(ndf.mean())