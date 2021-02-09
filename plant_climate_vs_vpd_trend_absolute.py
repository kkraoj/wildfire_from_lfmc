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
import matplotlib.patches as patches




sns.set(font_scale = 1.1, style = "ticks")


fullfilename = os.path.join(dir_root, "data","mean","vpdtrend.tif")
ds = gdal.Open(fullfilename)
vpd = np.array(ds.GetRasterBand(1).ReadAsArray())

# fullfilename = os.path.join(dir_root, "data","mean","vpd_mean.tif")
# ds = gdal.Open(fullfilename)
# vpd /= np.array(ds.GetRasterBand(1).ReadAsArray())/100

fullfilename = os.path.join(dir_root, "data","mean","landcover.tif")
ds = gdal.Open(fullfilename)
lc = np.array(ds.GetRasterBand(1).ReadAsArray())

vpd[lc==0] = np.nan

plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","PAS_6_jan_2021.tif")
ds = gdal.Open(plantClimatePath)
plantClimate = np.array(ds.GetRasterBand(1).ReadAsArray())

df = pd.DataFrame({"vpdTrend":vpd.flatten(), "sigma": plantClimate.flatten()}).dropna()

# res = 100
# fig, ax  = plt.subplots(figsize  = (3,3))
# ax.hist2d(x = df.sigma, y = df.vpdTrend, bins=(res, res), vmax = 2e2, vmin = 0, cmap='mako')
# ax.set_xlabel("PAS")
# ax.set_ylabel("VPD Trend (hPa/yr)")
# ax.set_xlim(0,2)
# ax.set_ylim(-0.05,0.2)
# ax.axvline(x = df.sigma.mean(), color = "lightgrey", linestyle = "--", linewidth = 2)
# ax.axhline(y = df.vpdTrend.mean(), color = "lightgrey", linestyle = "--", linewidth = 2)


# ax.annotate("%d %%"%(df[(df['vpdTrend']>=df['vpdTrend'].mean())&(df['sigma']>=df['sigma'].mean())].shape[0]/df.shape[0]*100),
#             xy = (1,1), xycoords = "axes fraction",ha = 'right',va = 'top',color = "lightgrey")
# ax.annotate("%d %%"%(df[(df['vpdTrend']<=df['vpdTrend'].mean())&(df['sigma']<=df['sigma'].mean())].shape[0]/df.shape[0]*100),
#             xy = (0.1,0), xycoords = "axes fraction",ha = 'left',va = 'bottom',color = "lightgrey")

# ax.annotate("%d %%"%(df[(df['vpdTrend']<=df['vpdTrend'].mean())&(df['sigma']>=df['sigma'].mean())].shape[0]/df.shape[0]*100),
#             xy = (1,0), xycoords = "axes fraction",ha = 'right',va = 'bottom',color = "lightgrey")

# ax.annotate("%d %%"%(df[(df['vpdTrend']>=df['vpdTrend'].mean())&(df['sigma']<=df['sigma'].mean())].shape[0]/df.shape[0]*100),
#             xy = (.1,1), xycoords = "axes fraction",ha = 'left',va = 'top',color = "lightgrey")


# geyser = sns.load_dataset("geyser")
# sns.kdeplot(data=geyser, x="waiting", y="duration")

# cuts = [df.vpdTrend.min(),0,0.025,0.05,df.vpdTrend.max()]
cuts = [df.vpdTrend.min(),0,0.03,0.06,df.vpdTrend.max()]
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
print(n)
# ax.annotate("n = 25k",
#             xy = (0.7,0.4), ha = 'left',va = 'top',
#             color = colors[0],weight = "bold",fontsize = 12)
# ax.annotate("n = 89k",
#             xy = (.94,1.15), ha = 'left',va = 'top',
#             color = colors[1],weight = "bold",fontsize = 12)
# ax.annotate("n = 155k",
#             xy = (0.8,1.4), ha = 'left',va = 'top',
#             color = colors[2],weight = "bold",fontsize = 12)
# ax.annotate("n = 10k",
#             xy = (0.35,1.8),ha = 'left',va = 'top',
#             color = colors[3],weight = "bold",fontsize = 12)

ax.set_ylabel("PWS")
ax.set_xlabel("Density")
ax.set_ylim(0,2.5)
ax.set_xlim(0,1.6)
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
colors = list(np.repeat([colors[0]], 3)) + colors + list(np.repeat([colors[-1]], 5))
cmap = ListedColormap(colors)
scatter_kwargs = dict(cmap = cmap,vmin = -0.1, vmax = 0.2)
fig, ax = plt.subplots(figsize = (3,3))
fig, ax, m, plot = plotmap(gt = gt, var = vpd,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig = fig,ax=ax,
                      shapefilepath = 'D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                  shapefilename ='states')
ax.axis('off')

# cax = fig.add_axes([0.68, 0.45, 0.02, 0.3])
# cbar = fig.colorbar(plot, cax=cax, orientation='vertical')
# cax.set_title("       VPD trend\n       (hPa/yr)",ha = "center")
# cbar.set_ticks([-0.2,-0.1,0,0.1,0.2,])
# cbar.ax.tick_params(labelsize=8) 
scatter_kwargs = dict(cmap = "Greys",vmin = 0, vmax = 1,alpha = 0)
# data = scipy.ndimage.zoom(plantClimate,0.1)
data = np.nan_to_num(plantClimate,nan = -9999)
data = gaussian_filter(data, sigma = 3,order = 0)
data[data<0] = np.nan
# plt.imshow(data,cmap = "viridis")

# fig2, ax2, m, plot = plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
#                       fill = "white",background="white",fig=fig, ax=ax,contour = True,contourLevel = np.nanquantile(data,0.9),contourColor = "lightgrey",
#                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')
plt.show()
print(np.nanmean(vpd[plantClimate>=np.nanquantile(plantClimate,0.9)]))
print(np.nanmean(vpd))
print(ndf.mean()[:3].mean())
print(ndf.mean())

#%% 2d density plots
sns.set(font_scale = 1.1, style = "ticks")

fig = plt.figure(constrained_layout=True, figsize = (3.3,3))
ns = 6
gs = fig.add_gridspec(6, 6, wspace=0.01, hspace=0.01)
ax = fig.add_subplot(gs[1:, :ns-1])
axh = fig.add_subplot(gs[0, :ns-1])
axv = fig.add_subplot(gs[1:, -1])
axh.set_xticks([])
axv.set_yticks([])

sns.kdeplot(data = df.sample(int(1e4), random_state =1), x = "sigma",y = "vpdTrend",\
    fill=True,cmap="mako_r",levels = 10, ax = ax)
lw = 2
risk = np.where((vpd<0)&(plantClimate<1), plantClimate, np.nan).flatten()
sns.kdeplot(data =risk,linewidth = lw,\
    fill=True, ax = axh,color = "#FEC5E5" )

risk = np.where((vpd>=0)&(vpd<0.05)&(plantClimate>=1)&(plantClimate<1.5), plantClimate, np.nan).flatten()
sns.kdeplot(data = risk.flatten(),linewidth = lw,\
    fill=True, ax = axh,color = "#FA86C4" )
risk = np.where((vpd>=0.05)&(plantClimate>=1.5), plantClimate, np.nan).flatten()
sns.kdeplot(data = risk,linewidth = lw,\
    fill=True, ax = axh,color = "#FF1694" )

    
risk = np.where((vpd<0)&(plantClimate<1), vpd, np.nan).flatten()
sns.kdeplot(data =risk,y = risk, linewidth = lw,\
    fill=True, ax = axv,color = "#FEC5E5" )

risk = np.where((vpd>=0)&(vpd<0.05)&(plantClimate>=1)&(plantClimate<1.5), vpd, np.nan).flatten()
sns.kdeplot(data = risk, y=risk,linewidth = lw,\
    fill=True, ax = axv,color = "#FA86C4" )
risk = np.where((vpd>=0.05)&(plantClimate>=1.5), vpd, np.nan).flatten()
sns.kdeplot(data = risk,y = risk, linewidth = lw,\
    fill=True, ax = axv,color = "#FF1694" )
    
# ax.axhline(y = 0, color = "darkgrey", linestyle = "-", linewidth = 0.5)
rect = patches.Rectangle((0,-0.05),1,0.05,linewidth=3,edgecolor='#FEC5E5',facecolor='none', clip_on = False, zorder = 99)
ax.add_patch(rect)
rect = patches.Rectangle((1,0),0.5,0.05,linewidth=3,edgecolor='#FA86C4',facecolor='none', clip_on = False, zorder = 100)
ax.add_patch(rect)
rect = patches.Rectangle((1.5,0.05),0.49,0.05,linewidth=3,edgecolor='#FF1694',facecolor='none', clip_on = False, zorder = 101)

ax.add_patch(rect)
ax.set_ylabel("VPD Trend (hPa/yr)")
ax.set_xlabel("PWS")
axv.set_xlabel("")
axh.set_ylabel("")
ax.set_xlim(0,2)
axh.set_yticks([])
axv.set_xticks([])
axh.set_xlim(0,2)
axv.set_ylim(-0.05,0.15)

ax.set_ylim(-0.05,0.15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


axv.spines['right'].set_visible(False)
axv.spines['top'].set_visible(False)
axv.spines['bottom'].set_visible(False)

axh.spines['right'].set_visible(False)
axh.spines['top'].set_visible(False)
axh.spines['left'].set_visible(False)
# plt.tight_layout()

#%%high risk contour plot 

fig, ax = plt.subplots(figsize = (3,3))
risk = np.where((vpd<0)&(plantClimate<1), 1, np.nan)
scatter_kwargs=dict(cmap = ListedColormap(["#FEC5E5"]))
marker_factor = 0.2
fig, ax, m, plot = plotmap(gt = gt, var = risk,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = marker_factor, 
                      fill = "white",background="white",fig=fig, ax=ax,
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

scatter_kwargs=dict(cmap = ListedColormap(["#FA86C4"]))
risk = np.where((vpd>=0)&(vpd<0.05)&(plantClimate>=1)&(plantClimate<1.5), 1, np.nan)
fig, ax, m, plot = plotmap(gt = gt, var = risk,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = marker_factor, 
                      fill = "white",background="white",fig=fig, ax=ax,
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

scatter_kwargs=dict(cmap = ListedColormap(["#FF1694"]))
risk = np.where((vpd>=0.05)&(plantClimate>=1.5), 1, np.nan)
fig, ax, m, plot = plotmap(gt = gt, var = risk,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = marker_factor, 
                      fill = "white",background="white",fig=fig, ax=ax,
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

ax.axis('off')



