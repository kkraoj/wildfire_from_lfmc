# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:31:21 2020

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

SAVEPLOT = False

def subset_CA(wui):
    wuiCA = wui[200:450,:300]
    return wuiCA
    
sns.set(font_scale = 1., style = "ticks")
plt.style.use("pnas")
# wuiNames = ["wui1990.tif","wui2000.tif","wui2010.tif"]
# popNames = ["pop1990.tif","pop2000.tif","pop2010.tif"]

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
gt = ds.GetGeoTransform()
  
#%% growth rates for 10 bins


nbins = [0,1.0, 1.5, 4]
colors = ["#f1d4d4","#e6739f","#cc0e74"]
# colors = ["#FEC5E5","#FF1694"]
cmap = ListedColormap(colors)
    # rgb2hex accepts rgb or rgba
_, vulLabels = pd.cut(df['pc'],nbins, retbins = True)
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
    df['pcBin'] = pd.cut(df.pc, nbins, labels = vulLabels[:-1])
    
    cum = df.groupby("pcBin").pop.sum()
    ts.loc[1990+ctr*20, :] = cum
    ctr+=1
ts = ts.sort_index()
#%% bar chart with quantiles with big bars
width = 1
nbins = 3
xticks = np.linspace(0,nbins-1, nbins)
fig, ax = plt.subplots(figsize = (3,3))
ax.bar(xticks,ts.diff().dropna().values.tolist()[0],align = "center",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)
    
ax.set_xlabel("Risk")
ax.set_ylabel("WUI population rise")
ylabels = ['{:,.1f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.xaxis.set_major_locator(MultipleLocator(2*width))
ax.set_yticklabels(ylabels)
ax.set_xticks(xticks)
ax.set_xticklabels(["Low","Medium","High"])
# ax.set_xticks(np.linspace(0,1.1, nbins+1))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
# if SAVEPLOT:
#     fig.savefig(os.path.join(dir_fig, "WUI","wui pop rise bar plot absolute.eps"), format = "eps")


#%% bar chart with quantiles y axis normalized by total WUI expansion
fig, ax = plt.subplots(figsize = (3.2,3))
y = ts.diff().dropna()
y = (y/y.sum(axis = 1).values[0]*100).values.tolist()[0]

ax.bar(xticks,y,align = "center",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)
    
ax.set_xticks(xticks)
ax.set_xticklabels(["Low","Medium","High"])
ax.set_xlabel("Risk")

ax.set_ylabel("% WUI population rise\n(wrt to total WUI pop. rise)")

plt.tight_layout()
# if SAVEPLOT:
#     fig.savefig(os.path.join(dir_fig, "WUI", "WUI pop rise bar plot relative.eps"), format = "eps")


#%% bar chart with quantiles y axis normalized by 1990 population
fig, ax = plt.subplots(figsize = (3.2,3))
y = ts.diff().dropna()
y = (y/ts.iloc[0]*100).values[0]

ax.bar(xticks,y,align = "center",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)
    
ax.set_xticks(xticks)
ax.set_xticklabels(["Low","Medium","High"])
ax.set_xlabel("Risk")

ax.set_ylabel("% WUI population rise\n(wrt to 1990 population)")

plt.tight_layout()
# if SAVEPLOT:
#     fig.savefig(os.path.join(dir_fig, "WUI", "WUI pop rise bar plot relative wrt 1990.eps"), format = "eps")


#%% time series of WUI pop growth

# reds = sns.color_palette("dark:salmon",n_colors = nbins).as_hex()
fig, ax = plt.subplots(figsize = (2.2,2))
plot = ts.plot(ax = ax, legend = False, \
               cmap = ListedColormap(colors), linewidth = 4)
plot = ax.scatter(x = np.repeat(2002,nbins),y = np.repeat(5e6,nbins), \
                  c = np.linspace(0,max(xticks),nbins),\
                  s = 0,cmap = ListedColormap(colors) )
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='10%', pad=0.2)
cax.set_title("Risk",ha = "center")
cbar = fig.colorbar(plot, cax=cax, orientation='vertical')

cax.yaxis.set_ticks(xticks)
cax.yaxis.set_ticklabels(["Low","Medium","High"])
# cbar.ax.tick_params(labelsize=8) 

# ax.set_xticks([1990,2010])
# ax.set_xticklabels([2001,2016])
ax.set_ylabel("WUI population")
ylabels = ['{:,.0f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')



#%% time series of normalized WUI pop growth

# reds = sns.color_palette("dark:salmon",n_colors = nbins).as_hex()
fig, ax = plt.subplots(figsize = (3.2,2))
plot = (ts.divide(ts.iloc[0])*100).plot(ax = ax, legend = False, \
               cmap = ListedColormap(colors), linewidth = 4)
plot = ax.scatter(x = np.repeat(2002,nbins),y = np.repeat(150,nbins), \
                  c = np.linspace(0,max(xticks),nbins),\
                  s = 0,cmap = ListedColormap(colors) )
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='10%', pad=0.2)
cax.set_title("Risk",ha = "center")
cbar = fig.colorbar(plot, cax=cax, orientation='vertical')

cax.yaxis.set_ticks(xticks)
cax.yaxis.set_ticklabels(["Low","Medium","High"])
# cbar.ax.tick_params(labelsize=8) 

# ax.set_xticks([1990,2010])
# ax.set_xticklabels([2001,2016])
ax.set_ylabel("Normalized\nWUI population")
# ylabels = ['{:,.0f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
# ax.set_yticklabels(ylabels)
# Hide the right and top spines
ax.set_yticks([100,150,200,250])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
# if SAVEPLOT:
#     fig.savefig(os.path.join(dir_fig, "WUI", "WUI timeseries normalized.eps"), format = "eps")



# %% stacked bar plot
cumsum = ts.diff().dropna().cumsum(axis = 1)
fig, ax = plt.subplots(figsize =(3.2,3))

ax.bar(xticks,height = ts.diff().dropna().values[0], bottom= [0]+list(cumsum.values[0][:-1]), align = "center",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)

ax.set_xlabel("Risk")
ax.set_ylabel("WUI population rise")
ylabels = ['{:,.0f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)

ax.set_xticks(xticks)
ax.set_xticklabels(["Low","Medium","High"])
ax.set_xlabel("Risk")

plt.tight_layout()
# if SAVEPLOT:
#     fig.savefig(os.path.join(dir_fig, "WUI", "WUI pop rise bar plot absolute stacked.eps"), format = "eps")


#%% before after total WUI pop plots
fig, ax = plt.subplots(figsize =(3.2,3))
thinWidth = 0.6
ax.bar(xticks,height = ts.iloc[0], align = "center",bottom = [0]+list(ts.iloc[-1].cumsum())[:-1],\
       color = colors,width = thinWidth,edgecolor = "k",linewidth = 1.5, label = "1990")
bars =ax.bar(xticks,height =ts.diff().dropna().values[0], \
             bottom = [ts.iloc[0,0]]+\
                 [ts.iloc[1,0]+ts.iloc[0,1]]+\
                     [ts.iloc[1,:2].sum()+ts.iloc[0,2]],\
                             align = "center",\
        color = colors,width = thinWidth,edgecolor = "k",linewidth = 1.5, \
            label = "$\Delta$ 1990-2010")
for bar in bars:
    bar.set_hatch("////")
ax.legend(frameon = False, loc = "lower right")
# ax.legend(frameon = False, bbox_to_anchor = (1.05,1.05), loc = "upper right")    
ax.set_xlabel("Risk")
ax.set_ylabel("WUI population")
# ax.set_ylim(0,14e6)
ylabels = ['{:,.0f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)

ax.set_xticks(xticks)
ax.set_xticklabels(["Low","Medium","High"])
ax.set_xlabel("Risk")

plt.tight_layout()
# if SAVEPLOT:
#     fig.savefig(os.path.join(dir_fig, "WUI", "WUI pop rise before after bar plot stacked.eps"), format = "eps")


#%% only 1 stacked bar per year
fig, ax = plt.subplots(figsize =(88/25.4,1.5))

ts.sort_index(ascending = False).plot(kind = "barh", stacked = True, color = colors, legend = False, ax = ax, edgecolor = "k", linewidth = 0.7)

# xlabels = ['{:,.0f}'.format(x) + ' million' for x in ax.get_xticks()/1e6]
ax.set_xticks([0,1e7,2e7])
ax.set_xticklabels(['0 million', '10 million' , '20 million'])
# ax.set_yticklabels([2016,2001])
ax.set_xlabel("")

ax.set_title("WUI population", weight = "bold")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
if SAVEPLOT:
    fig.savefig(os.path.join(r"C:\Users\kkrao\Dropbox\meetingsKrishna\Figures\wildfire_from_lfmc\fig4a.eps"), format = "eps")

#%% only 1 bar proportionaized to 100
fig, ax = plt.subplots(figsize =(3.5,1.5))

(ts.divide(ts.sum(axis = 1), axis = 0)*100).sort_index(ascending = False).plot(kind = "barh", stacked = True, color = colors, legend = False, ax = ax, edgecolor = "k")
ax.set_xlabel("% WUI population")

plt.tight_layout()
# if SAVEPLOT:
#     fig.savefig(os.path.join(dir_fig, "WUI", "WUI pop stacked 1 bar relative.eps"), format = "eps")


#%% 4 bars starting with shortened y axis

fig, ax = plt.subplots(figsize = (3.5,4))
y = ts.diff().dropna()
y = (y/ts.iloc[0]*100).values[0]

y = np.insert(y,0, ts.sum(axis=1).diff().dropna().iloc[0]/ts.sum(axis=1).iloc[0]*100)

rects = ax.bar([0,1,2,3],y,align = "center",\
       color = ['grey']+colors,width = 0.7,edgecolor = "k",linewidth = 1)
    
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(["All\nWUI","Low\nhazard","Medium\nhazard","High\nhazard"])
ax.set_xlabel("")
ax.set_title("% growth in WUI population", weight = "bold")
ax.set_ylabel("% change in WUI population (1990-2010)")
ax.set_ylim(90,170)
ax.set_yticks([100,120,140,160])

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = int(rect.get_height())
        ax.annotate('+{}%'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize = 11)

# ax.annotate('1990 to 2010',
#                     xy=(0.38,0.7), xycoords = "axes fraction",
#                     xytext=(0.38,0.7),
#                     textcoords="axes fraction",
#                     ha='center', va='bottom', fontsize = 11)

autolabel(rects)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
# if SAVEPLOT:
#     fig.savefig(os.path.join(dir_fig, "WUI", "WUI pop rise bar plot relative wrt 1990 4 bars.eps"), format = "eps")





#%% non wui + WUI


# wuiNonWui = pd.DataFrame({"WUI":[2.53788e+06,2.64769e+06 ,2.34899e+06 ,3.21948e+06],\
#               "Non-WUI":[7.3954e+06, 7.11049e+06, 7.377e+06 , 7.26205e+06], \
#                   }, index =["Low","Medium","High"]).T
    

# fig, ax = plt.subplots(figsize =(3,3))
# thinWidth = 0.6
# ax.bar(xticks,height = wuiNonWui.iloc[0], align = "center",bottom = [0]+list(wuiNonWui.iloc[-1].cumsum())[:-1],\
#        color = colors,width = thinWidth,edgecolor = "k",linewidth = 1.5, label = "WUI")
# bars =ax.bar(xticks,height =wuiNonWui.iloc[-1], \
#              bottom = [wuiNonWui.iloc[0,0]]+\
#                  [wuiNonWui.iloc[1,0]+wuiNonWui.iloc[0,1]]+\
#                      [wuiNonWui.iloc[1,:2].sum()+wuiNonWui.iloc[0,2]]+\
#                          [wuiNonWui.iloc[1,:3].sum()+wuiNonWui.iloc[0,3]],\
#                              align = "center",\
#         color = colors,width = thinWidth,edgecolor = "k",linewidth = 1.5, \
#             label = "Non-WUI")
# for bar in bars:
#     bar.set_hatch("////")
# ax.legend(frameon = False)
# # ax.legend(frameon = False, bbox_to_anchor = (1.05,1.05), loc = "upper right")    
# ax.set_xlabel("Risk")
# ax.set_ylabel("Population")
# # ax.set_ylim(0,14e6)
# ylabels = ['{:,.0f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
# ax.set_yticklabels(ylabels)

# ax.set_xticks(xticks)
# ax.set_xticklabels(["Low","Medium","High"])
# ax.set_xlabel("Risk")
map_kwargs = dict(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
scatter_kwargs=dict(cmap = ListedColormap(["grey"]))
fig, ax = plt.subplots(figsize=(3,3))
fig, ax, m, plot = plotmap(gt = gt, var = np.where(wui1990>0, wui1990, np.nan),map_kwargs=map_kwargs \
                           ,scatter_kwargs=scatter_kwargs, marker_factor = 0.5, 
                     fill = "white",background="white",fig=fig, ax=ax,
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

fig, ax = plt.subplots(figsize=(3,3))
fig, ax, m, plot = plotmap(gt = gt, var = np.where(wui2010>0, wui2010, np.nan),map_kwargs=map_kwargs \
                           ,scatter_kwargs=scatter_kwargs, marker_factor = 0.5, 
                     fill = "white",background="white",fig=fig, ax=ax,
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')

fig, ax = plt.subplots(figsize=(3,3))
fig, ax, m, plot = plotmap(gt = gt, var = np.where(wuiDiff>0, wui2010, np.nan),map_kwargs=map_kwargs \
                           ,scatter_kwargs=scatter_kwargs, marker_factor = 0.5, 
                     fill = "white",background="white",fig=fig, ax=ax,
                      shapefilepath = r"D:\Krishna\projects\vwc_from_radar\data\usa_shapefile\west_usa\cb_2017_us_state_500k",shapefilename ='states')


#%%  reviewers request of dBA/dVPD %/hPA vs. WUI delta

fig, ax = plt.subplots(figsize = (88/25.4,88/25.4))
x = [108, 107 , 95, 160] # with all
y = [124, 95, 107, 170]
yerr = [29, 22, 24, 37]
ax.set_xlim(85,170)
ax.set_ylim(65,210)

# x = [107 , 95, 160]
# y = [95, 107, 170]
# yerr = [22, 24, 37]
ax.errorbar(x, y, yerr = yerr, color = "lightgrey", zorder = -1, linewidth = 0, elinewidth = 1,capsize = 3)
ax.scatter(x, y, s = 80, c =["grey","#f1d4d4","#e6739f","#cc0e74"], edgecolor = "grey")
ax.set_xlabel(r"% growth in WUI population")
ax.set_ylabel(r"$\rm \frac{d(Burned\ area)}{d(VPD)}$ (%/hPa)")
ax.set_title(r"Relative $\rm \frac{dBA}{dVPD}$ and $\Delta$ WUI population", weight = "bold")

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
if SAVEPLOT:
    fig.savefig(os.path.join(r"C:\Users\kkrao\Dropbox\meetingsKrishna\Figures\wildfire_from_lfmc\fig4b.eps"), format = "eps")
