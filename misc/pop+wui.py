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



def subset_CA(wui):
    wuiCA = wui[200:450,:300]
    return wuiCA
    
sns.set(font_scale = 1.1, style = "ticks")
wuiNames = ["wui1990.tif","wui2000.tif","wui2010.tif"]
popNames = ["pop1990.tif","pop2000.tif","pop2010.tif"]

res = 3.5932611
plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","lfmc_dfmc_100hr_lag_6_lfmc_dfmc_norm_positive_coefSum.tif")
ds = gdal.Open(plantClimatePath)
plantClimate = np.array(ds.GetRasterBand(1).ReadAsArray())
# plantClimate = subset_CA(plantClimate)
#%% absolute population
# colors = ['#1E88E5','#FFC107','#D81B60']
# fig, ax = plt.subplots(figsize = (3,3))
# ctr = 0
# for (wuiName, popName) in zip(wuiNames, popNames):
    
#     fullfilename = os.path.join(dir_root, "data","WUI",wuiName)
#     ds = gdal.Open(fullfilename)
#     wui = np.array(ds.GetRasterBand(1).ReadAsArray())
    
#     fullfilename = os.path.join(dir_root, "data","population","gee",popName)
#     ds = gdal.Open(fullfilename)
#     pop = np.array(ds.GetRasterBand(1).ReadAsArray())*res**2
#     if pop.shape[0]!=645:
#         pop = pop[1:646]
#     pop[pop<0] = 0
#     pop = subset_CA(pop)
#     wui = subset_CA(wui)
#     print(pop.sum())
#     pop = pop[wui>0]
#     pc = plantClimate[wui>0]
#     df = pd.DataFrame({"pc":pc,"pop":pop})
#     df.dropna(inplace = True)
#     nbins = 10
#     df['pcBin'] = pd.cut(df.pc, nbins, labels = np.linspace(df.pc.min(), df.pc.max(), nbins))
#     cum = df.groupby("pcBin").pop.sum()
#     cum = cum.cumsum()
#     width = 0.2
#     ax.plot(np.round(cum.index.categories-width/2, 2), cum, color = colors[ctr], linewidth = 3, label = 1990+ctr*10)
#     ctr+=1
    
#     # checking if density multiplication is working
#     # filename = "pop1990_project_4km.tif"
#     # fullfilename = os.path.join(dir_root, "data","population",filename)
#     # ds = gdal.Open(fullfilename)
#     # arr = np.array(ds.GetRasterBand(1).ReadAsArray())
#     # arr[arr<0] = 0
#     # arr.sum()*res**2 # 248311744 = 250 Mn population of USA in 1990
# ax.set_xlabel("Plant climate sensitivity")
# ylabels = ['{:,.0f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
# ax.set_yticklabels(ylabels)
# ax.set_ylabel('WUI Population')
# ax.legend(frameon = False)

#%% % population relative to total west USA

# fig, axs = plt.subplots(1,3,figsize = (9,3), sharey = True)
# ctr = 0
# for (wuiName, popName) in zip(wuiNames, popNames):
    
#     fullfilename = os.path.join(dir_root, "data","WUI",wuiName)
#     ds = gdal.Open(fullfilename)
#     wui = np.array(ds.GetRasterBand(1).ReadAsArray())
    
#     fullfilename = os.path.join(dir_root, "data","population","gee",popName)
#     ds = gdal.Open(fullfilename)
#     pop = np.array(ds.GetRasterBand(1).ReadAsArray())[:645]*res**2
#     pop[pop<0] = 0
#     total = pop.sum()
#     pop = pop[wui>0]
#     pc = plantClimate[wui>0]
#     df = pd.DataFrame({"pc":pc,"pop":pop})
#     df.dropna(inplace = True)
#     df.shape
#     nbins = 10
#     df['pcBin'] = pd.cut(df.pc, nbins, labels = np.linspace(df.pc.min(), df.pc.max(), nbins))
#     df.head()
#     cum = df.groupby("pcBin").pop.sum()/total*100
#     ax = axs[ctr]
#     width = 0.2
#     ax.bar(np.round(cum.index.categories-width/2, 2), cum, width = width)
#     ax.set_xlabel("Plant climate sensitivity")
#     ctr+=1
# axs[0].set_ylabel('Population (% total)')

#%% % population relative to WUI population of west USA

# fig, ax = plt.subplots(figsize = (3,3))
# ctr = 0
# for (wuiName, popName) in zip(wuiNames, popNames):
    
#     fullfilename = os.path.join(dir_root, "data","WUI",wuiName)
#     ds = gdal.Open(fullfilename)
#     wui = np.array(ds.GetRasterBand(1).ReadAsArray())
    
#     fullfilename = os.path.join(dir_root, "data","population","gee",popName)
#     ds = gdal.Open(fullfilename)
#     pop = np.array(ds.GetRasterBand(1).ReadAsArray())*res**2
#     if pop.shape[0]!=645:
#         pop = pop[1:646]
#     pop[pop<0] = 0
#     pop = subset_CA(pop)
#     wui = subset_CA(wui)
#     pop = pop[wui>0]
#     pc = plantClimate[wui>0]
#     df = pd.DataFrame({"pc":pc,"pop":pop})
#     df.dropna(inplace = True)
#     df.shape
#     nbins = 10
#     df['pcBin'] = pd.cut(df.pc, nbins, labels = np.linspace(df.pc.min(), df.pc.max(), nbins))
#     df.head()
#     cum = df.groupby("pcBin").pop.sum()
#     cum = cum/cum.sum()*100
#     cum = cum.cumsum()
#     ax.plot(np.round(cum.index.categories-width/2, 2), cum, color = colors[ctr], linewidth = 3, label = 1990+ctr*10)
#     ctr+=1
# ax.set_ylabel('Population (% WUI)')
# ax.set_xlabel("Plant climate sensitivity")
# ax.legend(frameon = False)

#%% % population relative to WUI population of west USA bubbble plot

# colors = ['#1E88E5','#FFC107','#D81B60']
# fig, ax = plt.subplots(figsize = (4,3))
# ctr = 0
# for (wuiName, popName) in zip(wuiNames, popNames):
    
#     fullfilename = os.path.join(dir_root, "data","WUI",wuiName)
#     ds = gdal.Open(fullfilename)
#     wui = np.array(ds.GetRasterBand(1).ReadAsArray())
    
#     fullfilename = os.path.join(dir_root, "data","population","gee",popName)
#     ds = gdal.Open(fullfilename)
#     pop = np.array(ds.GetRasterBand(1).ReadAsArray())*res**2
#     if pop.shape[0]!=645:
#         pop = pop[:645]
#     pop[pop<0] = 0

#     pop = pop[wui>0]
#     pc = plantClimate[wui>0]
#     df = pd.DataFrame({"pc":pc,"pop":pop})
#     df.dropna(inplace = True)
#     df.shape
#     nbins = 10
#     df['pcBin'] = pd.cut(df.pc, nbins, labels = np.linspace(df.pc.min(), df.pc.max(), nbins))
#     df.head()
#     cum = df.groupby("pcBin").pop.sum()
#     cum = cum/cum.sum()*100

#     high = cum.iloc[int(nbins*0.8):].sum()
#     low = cum.iloc[:int(nbins*0.8)].sum()
#     print(low)
#     ax.scatter(1,1,s = high*100, alpha = 0.5, color = colors[ctr])
#     ax.scatter(-1,1,s = low*100, alpha = 0.5, color = colors[ctr])
#     ctr+=1
# ax.set_xlim(-2,2)

#%% % absolute population timeseries split by pc 

# colors = ['#1E88E5','#FFC107']
# ts = pd.DataFrame(columns = ['High Vulnerability','Low Vulnerability'], index = [1990,2000,2010])
# thresh =0.4
# ctr = 0
# for (wuiName, popName) in zip(wuiNames, popNames):
    
#     fullfilename = os.path.join(dir_root, "data","WUI",wuiName)
#     ds = gdal.Open(fullfilename)
#     wui = np.array(ds.GetRasterBand(1).ReadAsArray())
    
#     fullfilename = os.path.join(dir_root, "data","population","gee",popName)
#     ds = gdal.Open(fullfilename)
#     pop = np.array(ds.GetRasterBand(1).ReadAsArray())*res**2
#     if pop.shape[0]!=645:
#         pop = pop[1:646]
#     pop[pop<0] = 0
#     pop = subset_CA(pop)
#     wui = subset_CA(wui)
#     pop = pop[wui>0]
#     pc = plantClimate[wui>0]
#     df = pd.DataFrame({"pc":pc,"pop":pop})
#     df.dropna(inplace = True)
#     df.shape
#     nbins = 10
#     df['pcBin'] = pd.cut(df.pc, nbins, labels = np.linspace(df.pc.min(), df.pc.max(), nbins))
#     df.head()
#     cum = df.groupby("pcBin").pop.sum()
#     ts.loc[1990+ctr*10, 'High Vulnerability'] = cum.iloc[int(nbins*thresh):].sum()
#     ts.loc[1990+ctr*10, 'Low Vulnerability'] = cum.iloc[:int(nbins*thresh)].sum()
#     ctr+=1
    
# sns.set(font_scale = 1.1, style = "ticks")
# fig, ax = plt.subplots(figsize = (3,3))
# ts.plot(ax = ax, marker = "o", color = colors, linewidth = 2.5)
# ylabels = ['{:,.0f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
# ax.set_yticklabels(ylabels)
# ax.set_ylabel('WUI Population')
# # ax.set_ylim(0, 16e6)
# ax.legend(frameon=False)

#%% % absolute population timeseries split by pc quantiles

vulLabels = ["low","medium","high","extreme"]
vulLabels = ["low","medium","high"]
ts = pd.DataFrame(columns = vulLabels, index = [1990,2000,2010])
thresh =0.4
ctr = 0
for (wuiName, popName) in zip(wuiNames, popNames):
    
    fullfilename = os.path.join(dir_root, "data","WUI",wuiName)
    ds = gdal.Open(fullfilename)
    wui = np.array(ds.GetRasterBand(1).ReadAsArray())
    
    fullfilename = os.path.join(dir_root, "data","population","gee",popName)
    ds = gdal.Open(fullfilename)
    pop = np.array(ds.GetRasterBand(1).ReadAsArray())*res**2
    if pop.shape[0]!=645:
        pop = pop[1:646]
    pop[pop<0] = 0
    # pop = subset_CA(pop)
    # wui = subset_CA(wui)
    pop = pop[wui==1]
    pc = plantClimate[wui==1]
    df = pd.DataFrame({"pc":pc,"pop":pop})
    df.dropna(inplace = True)
    df['pcBin'] = pd.qcut(df.pc, len(vulLabels), labels = vulLabels)
    df.head()
    cum = df.groupby("pcBin").pop.sum()
    ts.loc[1990+ctr*10, :] = cum
    ctr+=1
    
sns.set(font_scale = 1.1, style = "ticks")
fig, ax = plt.subplots(figsize = (3,3))
ctr=0
for level in vulLabels:
    ts[level].plot(ax = ax, marker = "o", color = "darkred", alpha = 0.33+ctr*0.33, linewidth = 2.5,markeredgewidth = 0)
    ctr+=1
ylabels = ['{:,.1f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)
ax.set_ylabel('WUI Population')
# ax.set_ylim(0, 16e6)
ax.legend(frameon=False)


density = gaussian_kde(df.pc)
xs = np.linspace(df.pc.min(),df.pc.max(),100)
density.covariance_factor = lambda : .25
density._compute_covariance()

# fig, ax = plt.subplots(figsize = (3,3))
# ax.plot(xs,density(xs), linewidth = 3, color = "darkred")
# # for q in [0.0,0.25,0.5,0.75]:
# for q in [0.0,0.33,0.67]:
#     low = np.quantile(df.pc, q)
#     high = np.quantile(df.pc, q+0.33)
#     xsq = xs[(xs>=low)&(xs<=high)]
#     densityq = density(xsq)
#     ax.fill_between(xsq, 0, densityq, color = "darkred",alpha = q+0.25 )
    
# ax.set_xlabel("Plant climate sentivity")
# ax.set_ylabel("Density")
# ax.set_ylim(0)

#%% growth rates for 10 bins


nbins = 15
cmap = plt.get_cmap('viridis',nbins)    # PiYG
colors = [mpl.colors.rgb2hex(cmap(i))  for i in range(cmap.N)]
  
    # rgb2hex accepts rgb or rgba
_, vulLabels = pd.qcut(df['pc'],nbins, retbins = True)
vulLabels = np.round(vulLabels, 2)
ts = pd.DataFrame(columns = vulLabels[:-1], index = [1990,2000,2010])
thresh =0.4
ctr = 0
for (wuiName, popName) in zip(wuiNames, popNames):
    
    fullfilename = os.path.join(dir_root, "data","WUI",wuiName)
    ds = gdal.Open(fullfilename)
    wui = np.array(ds.GetRasterBand(1).ReadAsArray())
    
    fullfilename = os.path.join(dir_root, "data","population","gee",popName)
    ds = gdal.Open(fullfilename)
    pop = np.array(ds.GetRasterBand(1).ReadAsArray())*res**2
    if pop.shape[0]!=645:
        pop = pop[0:645]
    pop[pop<0] = 0
    # pop = subset_CA(pop)
    # wui = subset_CA(wui)
    pop = pop[wui==1]
    pc = plantClimate[wui==1]
    df = pd.DataFrame({"pc":pc,"pop":pop})
    df.dropna(inplace = True)
    df['pcBin'] = pd.qcut(df.pc, nbins, labels = vulLabels[:-1])
    
    cum = df.groupby("pcBin").pop.sum()
    ts.loc[1990+ctr*10, :] = cum
    ctr+=1
    
sns.set(font_scale = 1.1, style = "ticks")
fig, ax = plt.subplots(figsize = (3,3))
ctr=0
for level in vulLabels[:-1]:
    ts[level].plot(ax = ax, color = colors[ctr], linewidth = 2.5,markeredgewidth = 0)
    ctr+=1
ax.set_ylim(0)
ylabels = ['{:,.1f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)
ax.set_ylabel('Intermix Population')
# ax.set_ylim(0, 16e6)
# ax.legend(frameon=False)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

density = gaussian_kde(df.pc)
xs = np.linspace(df.pc.min(),df.pc.max(),100)
density.covariance_factor = lambda : .25
density._compute_covariance()

fig, ax = plt.subplots(figsize = (3,3))
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
    
ax.set_xlabel("Plant climate sensitivity")
ax.set_ylabel("Density")

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

##Average growth rates 
width = 0.08
fig, ax = plt.subplots(figsize = (3,3))
ax.bar(np.linspace(0,1,nbins), ts.diff().dropna().loc[2000]/ts.loc[1990], width = width, color = colors, linewidth  = 0)
ax.bar(np.linspace(2,3,nbins), ts.diff().dropna().loc[2010]/ts.loc[2000], width = width, color = colors, linewidth  = 0)
    
# set the x-spine (see below for more info on `set_position`)
ax.spines['left'].set_position('zero')

# turn off the right spine/ticks
ax.spines['right'].set_color('none')
ax.yaxis.tick_left()

# set the y-spine
ax.spines['bottom'].set_position('zero')

# turn off the top spine/ticks
ax.spines['top'].set_color('none')
ax.xaxis.tick_bottom()
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylabel("Relative growth")
ax.set_xticks([0.5,2.5])
ax.set_xticklabels(["1990-2000",'2000-2010'])