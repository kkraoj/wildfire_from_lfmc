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
import plotly.express as px
import squarify
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


def subset_CA(wui):
    wuiCA = wui[200:450,:300]
    return wuiCA
    
sns.set(font_scale = 1.1, style = "ticks")
wuiNames = ["urban2001NeighborsResampledGee.tif","urban2016NeighborsResampledGee.tif"]
popNames = ["pop2000.tif","pop2010.tif"]
# popNames = ["worldPopDensity2005.tif","worldPopDensity2010.tif"]

res = 3.5932611
plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","lfmc_dfmc_100hr_lag_6_lfmc_dfmc_norm_positive_coefSum.tif")
ds = gdal.Open(plantClimatePath)
plantClimate = np.array(ds.GetRasterBand(1).ReadAsArray())


#%% % absolute population timeseries split by pc quantiles

ctr = 0
wuiThresh = 0.0
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
    plt.imshow(wui)
    plt.show()
    print((pop*wui).sum()/pop.sum())
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
gt = ds.GetGeoTransform()
map_kwargs = dict(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

scatter_kwargs = dict(cmap = "Oranges",vmin = 0, vmax = 1,alpha = 0.05)
# fig, ax, m, plot = plotmap(gt = gt, var = wuiDiff,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 0.2, 
#                       fill = "white",background="white",
#                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

# data = np.nan_to_num(plantClimate,nan = -9999)
# data = gaussian_filter(data, sigma = 3,order = 0)
# data[data<0] = np.nan
# # plt.imshow(data,cmap = "viridis")

# fig, ax, m, plot = plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
#                       fill = "white",background="white",fig=fig, ax=ax,contour = True,contourLevel = np.nanquantile(data,0.9),
#                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')
# plt.show()

# scatter_kwargs = dict(cmap = "viridis",vmin = 0, vmax = 2,alpha = 1)
# fig, ax, m, plot = plotmap(gt = gt, var = plantClimate,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 0.2, 
#                       fill = "white",background="white",
#                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

# data = np.nan_to_num(wuiDiff,nan = -9999)
# data = gaussian_filter(data, sigma = 3,order = 0)
# # data[data<0] = np.nan
# # plt.imshow(data,cmap = "viridis")

# fig, ax, m, plot = plotmap(gt = gt,  contourColor = "orange",var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
#                       fill = "white",background="white",fig=fig, ax=ax,contour = True,contourLevel = np.nanquantile(data,0.9),
#                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')
# plt.show()


    
#%% growth rates for 10 bins


nbins = 10
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
low = xs.min()
ctr = 0
for q in vulLabels[1:]:
    high = vulLabels[ctr+1]
    xsq = xs[(xs>=low)&(xs<=high)]
    densityq = density(xsq)
    ax.fill_between(xsq, 0, densityq, facecolor = colors[ctr],  linewidth = 1.5,edgecolor = "grey")
    ctr+=1
    low = xsq.max()
    
# ax.set_xlabel("Plant climate sensitivity")
ax.set_ylabel("Density")
ax.set_xticks([0,1,2])
ax.set_yticks([0,0.5,1])

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

fig, ax = plt.subplots(figsize = (3,3))

ax.bar(ts.columns,ts.diff().dropna().values.tolist()[0],align = "edge",color = colors,width = np.diff(vulLabels),edgecolor = "k")
ax.set_xlabel("PWS")
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
ctr=1
low = xs.min()
for q in vulLabels[1:]:
    high = vulLabels[ctr]
    xsq = xs[(xs>=low)&(xs<=high)]
    densityq = density(xsq)
    ax2.fill_between(xsq, 0, densityq, facecolor = colors[ctr-1],  linewidth = 1.5,edgecolor = "grey")
    ctr+=1
    low = xsq.max()
    
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

# wui1990[wui1990<1] = np.nan
# fig, ax, m, plot = plotmap(gt = gt,  var = wui1990,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 0.01, 
#                       fill = "white",background="white",
#                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

# wui2010[wui2010<1] = np.nan
# fig, ax, m, plot = plotmap(gt = gt,  var = wui2010,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 0.01, 
#                       fill = "white",background="white",
#                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

# plt.show()

## 30m data m

# plantClimatePath = os.path.join(dir_root, "data","WUI","30m","urban2001mosaicNeighbors.tif")
# ds = gdal.Open(plantClimatePath)
# data = np.array(ds.GetRasterBand(1).ReadAsArray())
# print(data.sum()*30*30/1000/1000)

# plantClimatePath = os.path.join(dir_root, "data","WUI","30m","urban2016mosaicNeighbors.tif")
# ds = gdal.Open(plantClimatePath)
# data = np.array(ds.GetRasterBand(1).ReadAsArray())
# print(data.sum()*30*30/1000/1000)


#%% bubble plot for wui pop expansion
popDiff = ts.columns,ts.diff().dropna().values.tolist()[0]
popDiff = pd.Series(index = popDiff[0], data = popDiff[1])
xinds = []
for i in range(len(vulLabels)-1):
    xinds.append(np.mean([vulLabels[i],vulLabels[i+1]]))
    
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (3, 3), sharex = True)

ax2.plot(xs,density(xs), linewidth = 2, color = "grey")
# for q in [0.0,0.25,0.5,0.75]:
low = xs.min()
ctr = 0
for q in vulLabels[1:]:
    high = vulLabels[ctr+1]
    xsq = xs[(xs>=low)&(xs<=high)]
    densityq = density(xsq)
    ax2.fill_between(xsq, 0, densityq, facecolor = colors[ctr],  linewidth = 1.5,edgecolor = "grey")
    ctr+=1
    low = xsq.max()
    
ax2.set_ylabel("Density",ha ="center")
ax2.set_ylim(0,1.1)
ax2.set_yticks([0,0.5,1])
ax2.set_xticks([0,0.5,1,1.5,2])
ax2.set_xlim(0,2.1)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_xlabel("PWS")

maxY = 2e6
ax1.set_ylim(0,maxY)
ctr = 0
for (x,y) in zip(xinds, popDiff.values/maxY):
    ax1.axvline(x = x,ymin = 0, ymax = y, color ="grey", linewidth = 1,zorder =-1)
    t1 = plt.Polygon([[vulLabels[ctr],0],[vulLabels[ctr+1],0],[x,0.2e6]],color="grey")
    ax1.add_patch(t1)
    ctr+=1
ax1.scatter(xinds, popDiff.values, s = 30, c = colors,edgecolor = "grey",linewidth = 1)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_ylabel("$\Delta$ WUI population")

ylabels = ['{:,.1f}'.format(x) + ' M' for x in ax1.get_yticks()/1e6]
ax1.set_yticklabels(ylabels)

#%% bar chart with quantiles
width = 10

xticks = np.linspace(0,100-width,nbins)
fig, ax = plt.subplots(figsize = (3,3))
ax.bar(xticks,ts.diff().dropna().values.tolist()[0],align = "edge",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)
    

ax.hlines(y = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
           xmin = xticks[0], xmax = xticks[int(nbins/2)],linestyle = "--", \
               color = "k")
    
# ax.fill_between(x = xticks[:int(nbins/2)],\
#     y1 = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]) - \
#         np.std(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
#     y2 = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]) + \
#         np.std(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
#     color = "k",alpha = 0.2)
# ax.fill_between(x = xticks[int(nbins/2):],\
#     y1 = np.mean(ts.diff().dropna().values.tolist()[0][int(nbins/2):]) - \
#         np.std(ts.diff().dropna().values.tolist()[0][int(nbins/2):]), \
#     y2 = np.mean(ts.diff().dropna().values.tolist()[0][int(nbins/2):]) + \
#         np.std(ts.diff().dropna().values.tolist()[0][int(nbins/2):]), \
#     color = "k",alpha = 0.2)
ax.hlines(y = np.mean(ts.diff().dropna().values.tolist()[0][int(nbins/2):]), \
           xmin =xticks[int(nbins/2)], xmax = xticks[-1]+width,linestyle = "--", \
               color = "k")
    
ax.set_xlabel("PWS percentile")
ax.set_ylabel("WUI population rise")
ylabels = ['{:,.1f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.xaxis.set_major_locator(MultipleLocator(2*width))
ax.set_yticklabels(ylabels)
# ax.set_xticks(np.linspace(0,1.1, nbins+1))
ax.xaxis.set_minor_locator(MultipleLocator(width))

#%% bar chart with quantiles percentage
width = 10

xticks = np.linspace(0,100-width,nbins)
fig, ax = plt.subplots(figsize = (3,3))
ax.bar(xticks,(ts.diff().dropna()/ts.iloc[0,:]*100).values.tolist()[0],align = "edge",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)

ax.set_xlabel("PWS percentile")
ax.set_ylabel("% WUI population rise")
# ylabels = ['{:,.1f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
ax.xaxis.set_major_locator(MultipleLocator(2*width))
# ax.set_yticklabels(ylabels)
# ax.set_xticks(np.linspace(0,1.1, nbins+1))
ax.xaxis.set_minor_locator(MultipleLocator(width))

#%% time series of WUI pop growth

reds = sns.color_palette("dark:salmon",n_colors = nbins).as_hex()
fig, ax = plt.subplots(figsize = (2,2))
plot = ts.plot(ax = ax, legend = False, \
               cmap = ListedColormap(reds), linewidth = 2)
# ax.plot(ts.iloc[:,0],colors[0], linewidth = 3)
# ax.plot(ts.iloc[:,5],colors[5],marker = "o",markeredgecolor = "k")
# ax.plot(ts.iloc[:,-1],colors[-1], linewidth = 3)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.2)

cax.set_title("PWS percentile",ha = "center")
# cbar.set_ticks([-0.2,-0.1,0,0.1,0.2,])
# cbar.ax.tick_params(labelsize=8) 

ax.set_xticks([1990,2010])
ax.set_xticklabels([2001,2016])
ax.set_ylabel("WUI population")
ylabels = ['{:,.0f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)



#%% treemap for wui pop expansion
# sns.set(font_scale = 0.9, style = "ticks")

# df = pd.DataFrame({"popDiff":popDiff.values, "colors":colors},index = popDiff.index)
# df = df.sort_values("popDiff",ascending = False)

# label = ((np.round(df.popDiff*1e-6*10)/10).astype(str) + " M").values
# label[-3:] = [x.replace(" ","\n") for x in label[-3:]]
# fig, ax = plt.subplots(figsize = (3,2))
# squarify.plot(df.popDiff, color = df.colors,ax = ax,label=label)
# for child in ax.get_children():
#     if isinstance(child, mpl.text.Text):
#         if child.get_text() in ["1.1 M","0.7 M","0.6 M","0.4\nM"]:
#             child.set(color = "white")
# plt.axis('off')
# plt.gca().invert_yaxis()
# ax.set_title("$\Delta$ WUI population")
# fig, ax2 = plt.subplots(figsize = (3,1))
# ax2.plot(xs,density(xs), linewidth = 2, color = "grey")
# # for q in [0.0,0.25,0.5,0.75]:
# low = xs.min()
# ctr = 0
# for q in vulLabels[1:]:
#     high = vulLabels[ctr+1]
#     xsq = xs[(xs>=low)&(xs<=high)]
#     densityq = density(xsq)
#     ax2.fill_between(xsq, 0, densityq, facecolor = colors[ctr],  linewidth = 1.5,edgecolor = "grey")
#     ctr+=1
#     low = xsq.max()
    
# ax2.set_ylabel("Density",ha ="center")
# ax2.set_ylim(0,1.1)
# ax2.set_yticks([0,0.5,1])
# ax2.set_xticks([0,0.5,1,1.5,2])
# ax2.set_xlim(0,2.1)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.set_xlabel("PWS")
# plt.show()

#%% PDFs for wui expansion

fig, ax = plt.subplots(figsize = (3,3))

# sns.kdeplot(plantClimate.flatten(),ax = ax,label = "Overall")
sns.kdeplot(np.where(wui1990==1, plantClimate, np.nan).flatten(),weights = pop1990.flatten(), ax = ax, label = "weighted by 2001 WUI pop", bw_adjust = 1)
sns.kdeplot(np.where(wui2010==1, plantClimate, np.nan).flatten(),weights = np.where((pop2010-pop1990).flatten()>=0,(pop2010-pop1990).flatten(), 0) , ax = ax,label = "weighted by $\Delta$ WUI population")
ax.set_xlabel("PWS")
ax.set_ylabel("Density")
ax.legend(bbox_to_anchor=(1,1))

#%% CDFs for wui expansion

fig, ax = plt.subplots(figsize = (3,3))

# sns.ecdfplot(plantClimate.flatten(),ax = ax,label = "Overall")
sns.ecdfplot(np.where(wui1990==1, plantClimate, np.nan).flatten(),ax = ax, label = "Overall")
sns.ecdfplot(data = pd.DataFrame({"data":np.where(wui2010==1, plantClimate, np.nan).flatten()}) , \
         x = "data",ax = ax,label = "weighted by $\Delta$ WUI population",\
        weights = np.where((pop2010-pop1990).flatten()>=0,(pop2010-pop1990).flatten(), 0))
ax.set_xlabel("PWS")
ax.set_ylabel("CDF")
ax.legend(bbox_to_anchor=(1,1))




