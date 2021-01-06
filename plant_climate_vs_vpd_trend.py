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
from plotmap import plotmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap


sns.set(font_scale = 1.1, style = "ticks")


fullfilename = os.path.join(dir_root, "data","mean","vpdtrend.tif")
ds = gdal.Open(fullfilename)
vpd = np.array(ds.GetRasterBand(1).ReadAsArray())

fullfilename = os.path.join(dir_root, "data","mean","vpd_mean.tif")
ds = gdal.Open(fullfilename)
vpd /= np.array(ds.GetRasterBand(1).ReadAsArray())/100


plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","lfmc_dfmc_100hr_lag_6_lfmc_dfmc_norm_positive_coefSum.tif")
ds = gdal.Open(plantClimatePath)
plantClimate = np.array(ds.GetRasterBand(1).ReadAsArray())

df = pd.DataFrame({"vpdTrend":vpd.flatten(), "sigma": plantClimate.flatten()}).dropna()

res = 100
fig, ax  = plt.subplots(figsize  = (3,3))
ax.hist2d(x = df.sigma, y = df.vpdTrend, bins=(res, res), vmax = 2e2, vmin = 0, cmap='mako')
ax.set_xlabel("$\sigma$")
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
mycmap = sns.diverging_palette(240, 10, as_cmap=True)
negThresh = df.loc[df['vpdTrend']<0,'vpdTrend'].mean()
posThresh = df.loc[df['vpdTrend']>0,'vpdTrend'].mean()

fig, ax = plt.subplots(figsize =(2,3))
colors = sns.diverging_palette(240, 10,n=4).as_hex()

# sns.kdeplot(data= df[(df['vpdTrend']<0)].sigma, ax = ax, color = colors[0], label = "-ve VPD trend")
# sns.kdeplot(data= df[(df['vpdTrend']<df['vpdTrend'].mean())&(df['vpdTrend']>=0)].sigma,  ax = ax, color = colors[1], alpha = 0.5, label = "Below average +ve VPD trend")
# sns.kdeplot(data= df[(df['vpdTrend']>=df['vpdTrend'].mean())].sigma, ax = ax, color = colors[1], label = "Above average +ve VPD trend")

sns.kdeplot(data= df[(df['vpdTrend']<negThresh)], y = "sigma",ax = ax, color = colors[0])
sns.kdeplot(data= df[(df['vpdTrend']>=negThresh)&(df['vpdTrend']<0)], y = "sigma",  ax = ax, color = colors[1])
sns.kdeplot(data= df[(df['vpdTrend']<posThresh)&(df['vpdTrend']>=0)], y = "sigma",  ax = ax, color = colors[2])
sns.kdeplot(data= df[df['vpdTrend']>=posThresh], y = "sigma", ax = ax, color = colors[3])

ax.set_ylabel("PAS")
ax.set_xlabel("Density")
ax.set_ylim(0,2.5)
ax.set_xticks([0,0.5,1,1.5])
# ax.legend(bbox_to_anchor = [0.5,-0.2], loc = "upper center")

# %% VPD sigma box plot
fig, ax = plt.subplots(figsize =(1,3))
colors = sns.diverging_palette(240, 10,n=4).as_hex()

# sns.kdeplot(data= df[(df['vpdTrend']<0)].sigma, ax = ax, color = colors[0], label = "-ve VPD trend")
# sns.kdeplot(data= df[(df['vpdTrend']<df['vpdTrend'].mean())&(df['vpdTrend']>=0)].sigma,  ax = ax, color = colors[1], alpha = 0.5, label = "Below average +ve VPD trend")
# sns.kdeplot(data= df[(df['vpdTrend']>=df['vpdTrend'].mean())].sigma, ax = ax, color = colors[1], label = "Above average +ve VPD trend")


ndf = pd.DataFrame({-3:df[(df['vpdTrend']<negThresh)].sigma,
                    -1: df[(df['vpdTrend']>=negThresh)&(df['vpdTrend']<0)].sigma,
                    1:df[(df['vpdTrend']<posThresh)&(df['vpdTrend']>=0)].sigma,
                    3:df[df['vpdTrend']>=posThresh].sigma})
# sns.boxplot(data= df[(df['vpdTrend']<negThresh)], y = "sigma", ax = ax, color = colors[0])
# sns.boxplot(data= df[(df['vpdTrend']>=negThresh)&(df['vpdTrend']<0)], y = "sigma",  ax = ax, color = colors[1])
# sns.boxplot(data= df[(df['vpdTrend']<posThresh)&(df['vpdTrend']>=0)], y = "sigma",  ax = ax, color = colors[2])
# sns.boxplot(data= df[df['vpdTrend']>=posThresh], y = "sigma", ax = ax, color = colors[3])

sns.boxplot(data= ndf, ax = ax,palette = colors,saturation = 1,width = 0.8,fliersize = 0)

ax.set_ylabel("")
ax.set_xlabel("")
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim(0,2.5)




#%% VPD trend map

gt = ds.GetGeoTransform()
from plotmap import plotmap
map_kwargs = dict(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
mycmap = sns.diverging_palette(240, 10, as_cmap=True)
scatter_kwargs = dict(cmap = mycmap,vmin = -1, vmax = 1)
fig, ax, m, plot = plotmap(gt = gt, var = vpd,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\states",shapefilename ='states')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(plot, cax=cax, orientation='vertical')
cax.set_title("VPD trend (%/yr)")
scatter_kwargs = dict(cmap = "Greys",vmin = 0, vmax = 1,alpha = 0)

fig, ax, m, plot = plotmap(gt = gt, var = plantClimate,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig=fig, ax=ax,contour = True,
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\states",shapefilename ='states')


plt.show()
