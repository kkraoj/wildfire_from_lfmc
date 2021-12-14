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
colors = [colors[1]]+sns.color_palette("YlOrBr",n_colors = bins+1).as_hex()[:-2]
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

ndf = ndf[[3,2,1,0]]
fig, ax = plt.subplots(figsize =(2.5,0.7))
sns.boxplot(data= ndf, ax = ax,palette = colors[::-1],orient = "h",\
            saturation = 1,width = 0.7,fliersize = 0)

ax.set_ylabel("")
ax.set_xlabel("")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,2.17)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

if SAVEPLOT:
    fig.savefig(os.path.join(dir_fig, "vpd box plot mini.eps"), format = "eps")


fig, ax = plt.subplots(figsize =(3.5,3))
sns.boxplot(data= ndf, ax = ax,palette = colors[::-1],orient = "h",\
            saturation = 1,width = 0.7,fliersize = 0)

ax.set_xlabel("Plant-water sensitivity (PWS)")
ax.set_ylabel("VPD trend (hPa/yr)")
# ax.set_xticks([])
# ax.set_yticks([])
labels = [r"$\in$[-0.2,0)",r"$\in$[0,0.03)",r"$\in$[0.03,0.06)",r"$\in$[0.06,0.20]"]
ax.set_yticklabels(labels[::-1])
ax.set_xlim(-0.02,2.17)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

if SAVEPLOT:
    fig.savefig(os.path.join(dir_fig, "vpd box plot big.eps"), format = "eps")

# %% PWS box plot

colorsPWS = ["#f1d4d4","#e6739f","#cc0e74"]
# cuts = [0, 1.0,1.5,4]
# ndf = pd.DataFrame(columns = range(len(cuts)-1), index = range(df.shape[0]))
# for i in range(len(cuts)-1):
#     minVal = cuts[i]
#     maxVal = cuts[i+1]
#     data = df.loc[(df.sigma>=minVal)&(df.sigma<maxVal)]
#     ndf.loc[0:len(data.vpdTrend)-1,i] = data.vpdTrend.values
# ndf.dropna(inplace = True,how = "all")

# # ndf = ndf[[3,2,1,0]]
# fig, ax = plt.subplots(figsize =(0.4, 2.6))
# sns.boxplot(data= ndf, ax = acontourx,palette = colorsPWS,orient = "v",\
#             saturation = 1,width = 0.7,fliersize = 0)

# ax.set_ylabel("")
# ax.set_xlabel("")
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_ylim(-0.055,0.15)

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

#%% VPD trend map

gt = ds.GetGeoTransform()
map_kwargs = dict(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
# mycmap = sns.diverging_palette(240, 10, as_cmap=True)
colors = list(np.repeat([colors[0]], 3)) + colors + list(np.repeat([colors[-1]], 5))
cmap = ListedColormap(colors)
scatter_kwargs = dict(cmap = cmap,vmin = -0.1, vmax = 0.2)


fig, ax = plt.subplots(figsize = (3,3), frameon = False)
fig, ax, m, plot = plotmap(gt = gt, var = vpd,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig = fig,ax=ax,
                      shapefilepath = 'D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                  shapefilename ='states')
scatter_kwargs=dict(cmap = ListedColormap([colorsPWS[-1]]))
risk = np.where((vpd>=0.05)&(plantClimate>=1.5), np.ones(plantClimate.shape), 0)

data = gaussian_filter(risk, sigma = 10,order = 0)
# fig, _, m, _ = plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      # fill = "white",background="white",fig=fig, ax=ax,contour = True,contourLevel = 0.1,contourColor = "aqua",
                      # shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')
ax.axis('off')

fig, _, m, _ = plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig=fig, ax=ax,contour = True,contourLevel =0.1,contourColor = colorsPWS[-1],contourWidth = 2,
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

# cax = fig.add_axes([0.7, 0.45, 0.03, 0.3])
# cbar = fig.colorbar(plot, cax=cax, orientation='vertical')
# cax.set_title("       VPD trend\n       (hPa/yr)",ha = "center")
# cbar.set_ticks([-0.2,-0.1,0,0.1,0.2,])
# cbar.ax.tick_params(labelsize=8) 
scatter_kwargs = dict(cmap = "Greys",vmin = 0, vmax = 1,alpha = 0)
plt.tight_layout()

if SAVEPLOT:
    fig.savefig(os.path.join(dir_fig, "vpd trend map absolute.tiff"), format = "tiff", dpi = 300)
    fig.savefig(r"C:/Users/kkrao/Dropbox/meetingsKrishna/Figures/wildfire_from_lfmc/fig3d.tiff", dpi = 300)

plt.show()

# fig, ax = plt.subplots(figsize = (3,3), frameon = False)
# fig, _, m, _ = plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
#                       fill = "white",background="white",fig=fig, ax=ax,contour = True,contourLevel =0.1,contourColor = colorsPWS[-1],contourWidth = 2,
#                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')
# ax.axis('off')
# plt.tight_layout()
# if SAVEPLOT:
#     fig.savefig(os.path.join(dir_fig, "vpd trend map absolute contours.eps"), format = "eps")
# plt.show()


print(np.nanmean(vpd[plantClimate>=np.nanquantile(plantClimate,0.9)]))
print(np.nanmean(vpd))
print(ndf.mean()[:3].mean())
print(ndf.mean())

### just colorbar separately
colorsUnique = list(dict.fromkeys(colors))
cmap = ListedColormap(colorsUnique)
scatter_kwargs = dict(cmap = cmap,vmin = -0.2, vmax = 0.2)

fig, ax = plt.subplots(figsize = (3,3), frameon = False)
fig, ax, m, plot = plotmap(gt = gt, var = vpd,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig = fig,ax=ax,
                      shapefilepath = 'D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                  shapefilename ='states')
cax = fig.add_axes([0.7, 0.45, 0.03, 0.3])
cbar = fig.colorbar(plot, cax=cax, orientation='vertical')
# cax.set_title("       VPD trend\n       (hPa/yr)",ha = "center")
cbar.set_ticks([-0.2,-0.1,0,0.1,0.2,])
cbar.set_ticklabels([])

if SAVEPLOT:
    # fig.savefig(os.path.join(dir_fig, "vpd trend map legend.tiff"), format = "tiff", dpi = 300)
    fig.savefig(r"C:/Users/kkrao/Dropbox/meetingsKrishna/Figures/wildfire_from_lfmc/fig3dlegend.tiff", dpi = 300)


#%% plant climate map
fig, ax = plt.subplots(figsize = (3,3),frameon = False)
scatter_kwargs = dict(cmap = "PiYG_r",vmin = 0, vmax = 2,alpha = 1)
ax.axis("off")

# plantClimate[plantClimate<1.9] = np.nan
fig, _, m, plot = plotmap(gt = gt, var = plantClimate,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig = fig,ax=ax,
                      shapefilepath = 'D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                  shapefilename ='states')
risk = np.where((vpd>=0.05)&(plantClimate>=1.5), np.ones(plantClimate.shape), 0)

data = gaussian_filter(risk, sigma = 10,order = 0)
fig, _, m, _ = plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                       fill = "white",background="white",fig=fig, ax=ax,contour = True,contourLevel = 0.1,contourColor = colorsPWS[-1],contourWidth = 2,
                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')
cax = fig.add_axes([0.7, 0.45, 0.03, 0.3])
    
# cax.annotate('Plant  (%) \n', xy = (0.,0.94), ha = 'left', va = 'bottom', color = "w")
cb0 = fig.colorbar(plot,ax=ax,cax=cax,ticks = np.linspace(0,2,5))
plt.tight_layout()
ax.axis("off")
# plt.savefig(os.path.join(dir_fig, "pws_with_contour.jpg"), dpi=300, facecolor='w', edgecolor='w',
        # frameon = False)

if SAVEPLOT:
    # fig.savefig(os.path.join(dir_fig, "PWS map.tiff"), format = "tiff", dpi = 300)
    fig.savefig(r"C:/Users/kkrao/Dropbox/meetingsKrishna/Figures/wildfire_from_lfmc/fig3b.tiff",dpi =300)
    
plt.show()


#%% plant climate map for blog

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('PiYG_r')
new_cmap = truncate_colormap(cmap, 0.6, 0.9)

fig, ax = plt.subplots(figsize = (3,3),frameon = False)
scatter_kwargs = dict(cmap = new_cmap,vmin = 0, vmax = 2,alpha = 1)
ax.axis("off")

# plantClimate[plantClimate<1.9] = np.nan
fig, _, m, plot = plotmap(gt = gt, var = plantClimate,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig = fig,ax=ax,
                      shapefilepath = 'D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                  shapefilename ='states')
# risk = np.where((vpd>=0.05)&(plantClimate>=1.5), np.ones(plantClimate.shape), 0)

# data = gaussian_filter(risk, sigma = 10,order = 0)
# fig, _, m, _ = plotmap(gt = gt, var = data,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      # fill = "white",background="white",fig=fig, ax=ax,contour = True,contourLevel = 0.1,contourColor = colorsPWS[-1],contourWidth = 2,
                      # shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')
cax = fig.add_axes([0.7, 0.45, 0.03, 0.3])
    
# cax.annotate('Plant  (%) \n', xy = (0.,0.94), ha = 'left', va = 'bottom', color = "w")
cb0 = fig.colorbar(plot,ax=ax,cax=cax,ticks = np.linspace(0,2,5))
plt.tight_layout()
ax.axis("off")
# plt.savefig(os.path.join(dir_fig, "pws_with_contour.jpg"), dpi=300, facecolor='w', edgecolor='w',
        # frameon = False)

# if SAVEPLOT:
    # fig.savefig(os.path.join(dir_fig, "PWS map.eps"), format = "eps")
    # fig.savefig(os.path.join(dir_fig, "PWS map.tiff"), format = "tiff", dpi = 300)
    
plt.show()

#%% 2d density plot of PWS vs. VPD trend


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

fig, ax =  plt.subplots(figsize = (3,3))


sns.kdeplot(data = df.sample(int(1e4), random_state =1), x = "sigma",y = "vpdTrend",\
    fill=True,cmap="gray_r",levels = 10, ax = ax)
lw = 2

rect = patches.Rectangle((1.5,0.05),0.5,0.1,linewidth=3,edgecolor=colorsPWS[-1],facecolor='none', clip_on = False, zorder = 101)
ax.set_xlim(0,2)
ax.set_ylim(-0.05,0.15)
ax.add_patch(rect)
ax.hlines(y = df.vpdTrend.median(), xmin = 0, xmax =2,  linewidth = 0.5, \
          linestyle = "--", color = "dimgrey")

ax.vlines(x = df.sigma.median(), ymin = -0.05, ymax =0.15,  linewidth = 0.5, \
          linestyle = "--", color = "dimgrey", label = "per axis\nmedians")
ax.set_ylabel("VPD trend (hPa/yr)")
ax.set_xlabel("Plant-water sensitivity (PWS)")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(loc = "lower right", frameon = False, bbox_to_anchor = (1.05,-0.05), fontsize = 9)
plt.tight_layout()

df.loc[(df.vpdTrend>=df.vpdTrend.median())&(df.sigma>=df.sigma.median())].shape[0]/df.shape[0]
if SAVEPLOT:
    fig.savefig(os.path.join(dir_fig, "vpd PWS 2d density tall.eps"), format = "eps")

plt.show()


gg = vpd[plantClimate>=1.5]
gg = gg[~np.isnan(gg)]
print((gg>=np.nanmean(vpd)).sum()/len(gg))

#%% 2d density plots VPD trend versus PWS
# colorsPc = ["#f1d4d4","#e6739f","#cc0e74","#790c5a"]


# fig = plt.figure(constrained_layout=True, figsize = (3.3,3))
# ns = 6
# gs = fig.add_gridspec(6, 6, wspace=0.01, hspace=0.01)
# ax = fig.add_subplot(gs[1:, :ns-1])
# axh = fig.add_subplot(gs[0, :ns-1])
# axv = fig.add_subplot(gs[1:, -1])
# axh.set_xticks([])
# axv.set_yticks([])

# sns.kdeplot(data = df.sample(int(1e4), random_state =1), x = "sigma",y = "vpdTrend",\
#     fill=True,cmap="mako_r",levels = 10, ax = ax)
# lw = 2
# risk = np.where((plantClimate<0.64), vpd, np.nan).flatten()
# sns.kdeplot(data =risk,y = risk,linewidth = lw,\
#     fill=False, ax = axv,color = colorsPc[0] )

# risk = np.where((plantClimate>=0.64)&(plantClimate<0.92), vpd, np.nan).flatten()
# sns.kdeplot(data = risk.flatten(),y = risk,linewidth = lw,\
#     fill=False, ax = axv,color = colorsPc[1] )
# risk = np.where((plantClimate>=0.92)&(plantClimate<1.2), vpd, np.nan).flatten()
# sns.kdeplot(data = risk,y = risk, linewidth = lw,\
#     fill=False, ax = axv,color = colorsPc[2] )
    
# risk = np.where((plantClimate>=1.2), vpd, np.nan).flatten()
# sns.kdeplot(data = risk,y = risk,linewidth = lw,\
#     fill=False, ax = axv,color = colorsPc[3] )
    
# colorsVpd = sns.diverging_palette(240, 10,n=(bins-1)*2).as_hex()
# colorsVpd = [colorsVpd[1]]+sns.color_palette("dark:salmon_r",n_colors = bins).as_hex()[:-1]

# risk = np.where((vpd<0), plantClimate, np.nan).flatten()
# sns.kdeplot(data =risk,x = risk, linewidth = lw,\
#     fill=False, ax = axh,color =  colorsVpd[0])

# risk = np.where((vpd>=0)&(vpd<0.025), plantClimate, np.nan).flatten()
# sns.kdeplot(data =risk,x = risk, linewidth = lw,\
#     fill=False, ax = axh,color =  colorsVpd[1])
# risk = np.where((vpd>=0.025)&(vpd<0.05), plantClimate, np.nan).flatten()
# sns.kdeplot(data =risk,x = risk, linewidth = lw,\
#     fill=False, ax = axh,color =  colorsVpd[2])
# risk = np.where((vpd>=0.05), plantClimate, np.nan).flatten()
# sns.kdeplot(data =risk,x = risk, linewidth = lw,\
#     fill=False, ax = axh,color =  colorsVpd[3])    

# rect = patches.Rectangle((1.5,0.05),0.5,0.1,linewidth=3,edgecolor='aqua',facecolor='none', clip_on = False, zorder = 101)

# ax.add_patch(rect)
# ax.set_ylabel("VPD trend (hPa/yr)")
# ax.set_xlabel("Plant-water sensitivity")
# axv.set_xlabel("")
# axh.set_ylabel("")
# ax.set_xlim(0,2)
# axh.set_yticks([])
# axv.set_xticks([])
# axh.set_xlim(0,2)
# ax.set_xticks([0,0.5,1.0,1.5,2])
# axv.set_ylim(-0.05,0.15)

# ax.set_ylim(-0.05,0.15)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)


# axv.spines['right'].set_visible(False)
# axv.spines['top'].set_visible(False)
# axv.spines['bottom'].set_visible(False)

# axh.spines['right'].set_visible(False)
# axh.spines['top'].set_visible(False)
# axh.spines['left'].set_visible(False)
# # plt.tight_layout()

#%%high risk contour plot 

# fig, ax = plt.subplots(figsize = (3,3))
# risk = np.where((vpd<0)&(plantClimate<1), 1, np.nan)
# scatter_kwargs=dict(cmap = ListedColormap(["#FEC5E5"]))
# marker_factor = 0.2
# fig, ax, m, plot = plotmap(gt = gt, var = risk,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = marker_factor, 
#                       fill = "white",background="white",fig=fig, ax=ax,
#                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

# scatter_kwargs=dict(cmap = ListedColormap(["#FA86C4"]))
# risk = np.where((vpd>=0)&(vpd<0.05)&(plantClimate>=1)&(plantClimate<1.5), 1, np.nan)
# fig, ax, m, plot = plotmap(gt = gt, var = risk,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = marker_factor, 
#                       fill = "white",background="white",fig=fig, ax=ax,
#                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

# scatter_kwargs=dict(cmap = ListedColormap(["#FF1694"]))
# risk = np.where((vpd>=0.05)&(plantClimate>=1.5), 1, np.nan)
# fig, ax, m, plot = plotmap(gt = gt, var = risk,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = marker_factor, 
#                       fill = "white",background="white",fig=fig, ax=ax,
#                       shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

# ax.axis('off')



