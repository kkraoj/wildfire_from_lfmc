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
# wuiNames = ["urban2001NeighborsResampledGee.tif","urban2016NeighborsResampledGee.tif"]
wuiNames = ["wui1990.tif","wui2010.tif"]
# popNames = ["pop2000.tif","pop2010.tif"]
popNames = ["pop1990.tif","pop2010.tif"]
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
    
    pop = pop[wui==0]
    pc = plantClimate[wui==0]
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
    pop = pop[wui==0]
    # pop = wui[wui==1].copy()
    # print(len(pop))
    pc = plantClimate[wui==0]
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
ax.set_ylabel("$\Delta$ Non-Wui population")

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

# 30m data m

# plantClimatePath = os.path.join(dir_root, "data","WUI","30m","urban2001mosaicNeighbors.tif")
# ds = gdal.Open(plantClimatePath)
# data = np.array(ds.GetRasterBand(1).ReadAsArray())
# print(data.sum()*30*30/1000/1000)
# fig, ax =plt.subplots(figsize = (3,3))
# plt.axis("off")
# ax.imshow(data, vmin = 0, vmax = 0.3)

# plantClimatePath = os.path.join(dir_root, "data","WUI","30m","urban2016mosaicNeighbors.tif")
# ds = gdal.Open(plantClimatePath)
# data = np.array(ds.GetRasterBand(1).ReadAsArray())
# print(data.sum()*30*30/1000/1000)
# fig, ax =plt.subplots(figsize = (3,3))
# plt.axis("off")
# ax.imshow(data, vmin = 0, vmax = 0.3)

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
ax1.set_ylabel("$\Delta$ Non-Wui population")

ylabels = ['{:,.1f}'.format(x) + ' M' for x in ax1.get_yticks()/1e6]
ax1.set_yticklabels(ylabels)

#%% bar chart with quantiles with big bars
width = 10

xticks = np.linspace(0,100-width,nbins)
fig, ax = plt.subplots(figsize = (3,3))
ax.bar(xticks,ts.diff().dropna().values.tolist()[0],align = "edge",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)
    

ax.hlines(y = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
            xmin = xticks[0], xmax = xticks[int(nbins/2)],linestyle = "--", \
                color = "black")
    
ax.fill_between(x = xticks[:int(nbins/2)+1],\
    y1 = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]) - \
        np.std(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
    y2 = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]) + \
        np.std(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
    color = "black",alpha = 0.6, zorder = 100)
ax.fill_between(x = np.linspace(50,100,6),\
    y1 = np.mean(ts.diff().dropna().values.tolist()[0][int(nbins/2):]) - \
        np.std(ts.diff().dropna().values.tolist()[0][int(nbins/2):]), \
    y2 = np.mean(ts.diff().dropna().values.tolist()[0][int(nbins/2):]) + \
        np.std(ts.diff().dropna().values.tolist()[0][int(nbins/2):]), \
    color = "grey",alpha = 0.4, zorder = 100)
ax.hlines(y = np.mean(ts.diff().dropna().values.tolist()[0][int(nbins/2):]), \
            xmin =xticks[int(nbins/2)], xmax = xticks[-1]+width,linestyle = "--", \
                color = "grey")
    
ax.set_xlabel("PWS percentile")
ax.set_ylabel("Non-Wui population rise")
ylabels = ['{:,.1f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.xaxis.set_major_locator(MultipleLocator(2*width))
ax.set_yticklabels(ylabels)
# ax.set_xticks(np.linspace(0,1.1, nbins+1))
ax.xaxis.set_minor_locator(MultipleLocator(width))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#%% bar chart with quantiles y axis normalized by total WUI expansion
width = 10

xticks = np.linspace(0,100-width,nbins)
fig, ax = plt.subplots(figsize = (3,3))
y = ts.diff().dropna()
y = (y/y.sum(axis = 1).values[0]*100).values.tolist()[0]

ax.bar(xticks,y,align = "edge",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)
    

ax.hlines(y = np.mean(y[:int(nbins/2)]), \
            xmin = xticks[0], xmax = xticks[int(nbins/2)],linestyle = "--", \
                color = "k")
    
ax.fill_between(x = xticks[:int(nbins/2)+1],\
    y1 = np.mean(y[:int(nbins/2)]) - \
        np.std(y[:int(nbins/2)]), \
    y2 = np.mean(y[:int(nbins/2)]) + \
        np.std(y[:int(nbins/2)]), \
    color = "k",alpha = 0.6, zorder = 100)
ax.fill_between(x = np.linspace(50,100,6),\
    y1 = np.mean(y[int(nbins/2):]) - \
        np.std(y[int(nbins/2):]), \
    y2 = np.mean(y[int(nbins/2):]) + \
        np.std(y[int(nbins/2):]), \
    color = "grey",alpha = 0.5, zorder = 100)
ax.hlines(y = np.mean(y[int(nbins/2):]), \
            xmin =xticks[int(nbins/2)], xmax = xticks[-1]+width,linestyle = "--", \
                color = "grey")
    
ax.set_xlabel("PWS percentile")
ax.set_ylabel("% Non-Wui population rise")
ax.xaxis.set_major_locator(MultipleLocator(2*width))
ax.xaxis.set_minor_locator(MultipleLocator(width))


#%% bar chart with quantiles y axis normalized by total WUI expansion, 2 big bars
width = 10

xticks = np.linspace(0,100-width,nbins)
fig, ax = plt.subplots(figsize = (3,3))
y = ts.diff().dropna()
y = (y/y.sum(axis = 1).values[0]*100).values.tolist()[0]

ax.bar(xticks,y,align = "edge",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)

ax.bar([0,50],[np.mean(y[:int(nbins/2)]), np.mean(y[int(nbins/2):])],\
       yerr =[np.std(y[:int(nbins/2)]), np.std(y[int(nbins/2):])], align = "edge",\
       color = ["k","grey"],width = 50,edgecolor = "k",linewidth = 1.5, \
           capsize = 10, alpha = 0.7)

ax.set_xlabel("PWS percentile")
ax.set_ylabel("% Non-Wui population rise")
ax.xaxis.set_major_locator(MultipleLocator(2*width))
ax.xaxis.set_minor_locator(MultipleLocator(width))
#%% bar chart with quantiles without big bars
width = 10

xticks = np.linspace(0,100-width,nbins)
fig, ax = plt.subplots(figsize = (3,3))
ax.bar(xticks,ts.diff().dropna().values.tolist()[0],align = "edge",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)
    

# ax.hlines(y = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
#            xmin = xticks[0], xmax = xticks[int(nbins/2)],linestyle = "--", \
#                color = "k")
    
# ax.fill_between(x = xticks[:int(nbins/2)+1],\
#     y1 = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]) - \
#         np.std(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
#     y2 = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]) + \
#         np.std(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
#     color = "k",alpha = 0.2)
# ax.fill_between(x = np.linspace(50,100,6),\
#     y1 = np.mean(ts.diff().dropna().values.tolist()[0][int(nbins/2):]) - \
#         np.std(ts.diff().dropna().values.tolist()[0][int(nbins/2):]), \
#     y2 = np.mean(ts.diff().dropna().values.tolist()[0][int(nbins/2):]) + \
#         np.std(ts.diff().dropna().values.tolist()[0][int(nbins/2):]), \
#     color = "k",alpha = 0.2)
# ax.hlines(y = np.mean(ts.diff().dropna().values.tolist()[0][int(nbins/2):]), \
#            xmin =xticks[int(nbins/2)], xmax = xticks[-1]+width,linestyle = "--", \
#                color = "k")
    
ax.set_xlabel("PWS percentile")
ax.set_ylabel("Non-Wui population rise")
ylabels = ['{:,.1f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.xaxis.set_major_locator(MultipleLocator(2*width))
ax.set_yticklabels(ylabels)
# ax.set_xticks(np.linspace(0,1.1, nbins+1))
ax.xaxis.set_minor_locator(MultipleLocator(width))

#%% bar chart with just two bars

fig, ax = plt.subplots(figsize = (1.5,1.5))

y = [np.sum(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]),np.sum(ts.diff().dropna().values.tolist()[0][int(nbins/2):])]
yerr = [np.std(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), np.std(ts.diff().dropna().values.tolist()[0][int(nbins/2):])]
x = [0,1]
ax.set_ylim(0,10e6)
ax.bar(x = x, height=y, yerr=yerr, color = "grey", capsize = 5,ecolor = "k", \
       edgecolor ="k", linewidth = 1, width = 0.7)
ylabels = ['{:,.0f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)
ax.set_xticks(x)
ax.set_xticklabels(["0-50", '50-100'],rotation = 0, ha = "center")
ax.set_ylabel("Non-Wui population\nrise")
ax.set_xlabel("PWS percentile")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)



#%% bar chart with quantiles percentage
width = 10

xticks = np.linspace(0,100-width,nbins)
fig, ax = plt.subplots(figsize = (3,3))
ax.bar(xticks,(ts.diff().dropna()/ts.iloc[0,:]*100).values.tolist()[0],align = "edge",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)

ax.set_xlabel("PWS percentile")
ax.set_ylabel("% Non-Wui population rise")
# ylabels = ['{:,.1f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
ax.xaxis.set_major_locator(MultipleLocator(2*width))
# ax.set_yticklabels(ylabels)
# ax.set_xticks(np.linspace(0,1.1, nbins+1))
ax.xaxis.set_minor_locator(MultipleLocator(width))

#%% time series of WUI pop growth

# reds = sns.color_palette("dark:salmon",n_colors = nbins).as_hex()
fig, ax = plt.subplots(figsize = (2.2,2))
plot = ts.plot(ax = ax, legend = False, \
               cmap = ListedColormap(colors), linewidth = 2)
plot = ax.scatter(x = np.repeat(2002,10),y = np.repeat(3e6,10), \
                  c = np.linspace(0,90,10),\
                  s = 0,cmap = ListedColormap(colors) )
# ax.plot(ts.iloc[:,0],colors[0], linewidth = 3)
# ax.plot(ts.iloc[:,5],colors[5],marker = "o",markeredgecolor = "k")
# ax.plot(ts.iloc[:,-1],colors[-1], linewidth = 3)

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.2)
cax.set_title("PWS\npercentile",ha = "center")
cbar = fig.colorbar(plot, cax=cax, orientation='vertical')

# cbar.set_ticks(np.linspace(0,100,11))
cax.yaxis.set_minor_locator(MultipleLocator(width*0.9))
cax.yaxis.set_ticks(np.linspace(0,90,6))
cax.yaxis.set_ticklabels(np.linspace(0,100,6).astype(int))
# cbar.ax.tick_params(labelsize=8) 

ax.set_xticks([1990,2010])
# ax.set_xticklabels([2001,2016])
ax.set_ylabel("Non-Wui population")
ylabels = ['{:,.0f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')


#%% time series of WUI pop growth

# reds = sns.color_palette("dark:salmon",n_colors = nbins).as_hex()
fig, ax = plt.subplots(figsize = (1.5,1.5))
# plot = ax.scatter(x = np.repeat(2002,10),y = np.repeat(5e6,10), \
#                   c = np.linspace(0,90,10),\
#                   s = 0,cmap = ListedColormap(colors) )
ax.plot(ts.iloc[:,0],colors[0], linewidth = 3)
# ax.plot(ts.iloc[:,5],colors[5],marker = "o",markeredgecolor = "k")
ax.plot(ts.iloc[:,-1],colors[-1], linewidth = 3)
ax.set_xticks([1990,2010])

# ax.set_xticklabels([1990,2016])
# ax.set_ylim(6e6,12e6)
# ax.set_yticks([8e6, 10e6, 12e6])
ax.set_ylabel("Non-Wui population")
ylabels = ['{:,.0f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)



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
# ax.set_title("$\Delta$ Non-Wui population")
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

# fig, ax = plt.subplots(figsize = (3,3))

# # sns.kdeplot(plantClimate.flatten(),ax = ax,label = "Overall")
# sns.kdeplot(np.where(wui1990==1, plantClimate, np.nan).flatten(),weights = pop1990.flatten(), ax = ax, label = "weighted by 2001 WUI pop", bw_adjust = 0.5)
# # sns.kdeplot(np.where(wui2010==1, plantClimate, np.nan).flatten(),weights = np.where((pop2010-pop1990).flatten()>=0,(pop2010-pop1990).flatten(), 0) , ax = ax,bw_adjust = 0.1, label = "weighted by $\Delta$ Non-Wui population")
# sns.kdeplot(np.where(wui2010==1, plantClimate, np.nan).flatten(),weights = (pop2010-pop1990).flatten()-(pop2010-pop1990).min(), ax = ax,bw_adjust = 0.5, label = "weighted by $\Delta$ Non-Wui population")
# ax.set_xlabel("PWS")
# ax.set_ylabel("Density")
# ax.legend(bbox_to_anchor=(1,1))

#%% CDFs for wui expansion

# fig, ax = plt.subplots(figsize = (3,3))

# # sns.ecdfplot(plantClimate.flatten(),ax = ax,label = "Overall")
# sns.ecdfplot(np.where(wui1990==1, plantClimate, np.nan).flatten(),ax = ax, label = "Overall")
# sns.ecdfplot(data = pd.DataFrame({"data":np.where(wui2010==1, plantClimate, np.nan).flatten()}) , \
#           x = "data",ax = ax,label = "weighted by $\Delta$ Non-Wui population",\
#         weights = np.where((pop2010-pop1990).flatten()>=0,(pop2010-pop1990).flatten(), 0))
# ax.set_xlabel("PWS")
# ax.set_ylabel("CDF")
# ax.legend(bbox_to_anchor=(1,1))

# %% stacked bar plot
cumsum = ts.diff().dropna().cumsum(axis = 1)
fig, ax = plt.subplots(figsize =(3,3))

ax.bar(xticks,height = ts.diff().dropna().values[0], bottom= [0]+list(cumsum.values[0][:-1]), align = "edge",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)

ax.set_xlabel("PWS percentile")
ax.set_ylabel("Non-Wui population rise")
ylabels = ['{:,.0f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)

ax.set_xticks(np.linspace(0,100,6))

#%% before after total WUI pop plots
fig, ax = plt.subplots(figsize =(3,3))

ax.bar(xticks,height = ts.iloc[0], align = "edge",\
       color = "dodgerblue",width = width,edgecolor = "k",linewidth = 1.5, label = "1990")
ax.bar(xticks,height =ts.diff().dropna().values[0], bottom = ts.iloc[0], align = "edge",\
       color = "darkorange",width = width,edgecolor = "k",linewidth = 1.5, label = "2010")

ax.legend(frameon = False, bbox_to_anchor = (1.05,1.05), loc = "upper right")    
ax.set_xlabel("PWS percentile")
ax.set_ylabel("Non-Wui population")
# ax.set_ylim(0,2e6)
ylabels = ['{:,.0f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)

ax.set_xticks(np.linspace(0,100,6))


#%% only 1 bar
fig, ax = plt.subplots(figsize =(3,1))

ts.sort_index(ascending = False).plot(kind = "barh", stacked = True, color = colors, legend = False, ax = ax, edgecolor = "darkgrey")

xlabels = ['{:,.0f}'.format(x) + ' M' for x in ax.get_xticks()/1e6]
ax.set_xticklabels(xlabels)
ax.set_yticklabels([2010,1990])
ax.set_xlabel("Non-Wui population")