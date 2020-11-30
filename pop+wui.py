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
plantClimate = subset_CA(plantClimate)
#%% absolute population
colors = ['#1E88E5','#FFC107','#D81B60']
fig, ax = plt.subplots(figsize = (3,3))
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
    pop = subset_CA(pop)
    wui = subset_CA(wui)
    print(pop.sum())
    pop = pop[wui>0]
    pc = plantClimate[wui>0]
    df = pd.DataFrame({"pc":pc,"pop":pop})
    df.dropna(inplace = True)
    nbins = 10
    df['pcBin'] = pd.cut(df.pc, nbins, labels = np.linspace(df.pc.min(), df.pc.max(), nbins))
    cum = df.groupby("pcBin").pop.sum()
    cum = cum.cumsum()
    width = 0.2
    ax.plot(np.round(cum.index.categories-width/2, 2), cum, color = colors[ctr], linewidth = 3, label = 1990+ctr*10)
    ctr+=1
    
    # checking if density multiplication is working
    # filename = "pop1990_project_4km.tif"
    # fullfilename = os.path.join(dir_root, "data","population",filename)
    # ds = gdal.Open(fullfilename)
    # arr = np.array(ds.GetRasterBand(1).ReadAsArray())
    # arr[arr<0] = 0
    # arr.sum()*res**2 # 248311744 = 250 Mn population of USA in 1990
ax.set_xlabel("Plant climate sensitivity")
ylabels = ['{:,.0f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)
ax.set_ylabel('WUI Population')
ax.legend(frameon = False)

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

fig, ax = plt.subplots(figsize = (3,3))
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
    pop = subset_CA(pop)
    wui = subset_CA(wui)
    pop = pop[wui>0]
    pc = plantClimate[wui>0]
    df = pd.DataFrame({"pc":pc,"pop":pop})
    df.dropna(inplace = True)
    df.shape
    nbins = 10
    df['pcBin'] = pd.cut(df.pc, nbins, labels = np.linspace(df.pc.min(), df.pc.max(), nbins))
    df.head()
    cum = df.groupby("pcBin").pop.sum()
    cum = cum/cum.sum()*100
    cum = cum.cumsum()
    ax.plot(np.round(cum.index.categories-width/2, 2), cum, color = colors[ctr], linewidth = 3, label = 1990+ctr*10)
    ctr+=1
ax.set_ylabel('Population (% WUI)')
ax.set_xlabel("Plant climate sensitivity")
ax.legend(frameon = False)

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

colors = ['#1E88E5','#FFC107']
ts = pd.DataFrame(columns = ['High Vulnerability','Low Vulnerability'], index = [1990,2000,2010])
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
    pop = subset_CA(pop)
    wui = subset_CA(wui)
    pop = pop[wui>0]
    pc = plantClimate[wui>0]
    df = pd.DataFrame({"pc":pc,"pop":pop})
    df.dropna(inplace = True)
    df.shape
    nbins = 10
    df['pcBin'] = pd.cut(df.pc, nbins, labels = np.linspace(df.pc.min(), df.pc.max(), nbins))
    df.head()
    cum = df.groupby("pcBin").pop.sum()
    ts.loc[1990+ctr*10, 'High Vulnerability'] = cum.iloc[int(nbins*thresh):].sum()
    ts.loc[1990+ctr*10, 'Low Vulnerability'] = cum.iloc[:int(nbins*thresh)].sum()
    ctr+=1
    
sns.set(font_scale = 1.1, style = "ticks")
fig, ax = plt.subplots(figsize = (3,3))
ts.plot(ax = ax, marker = "o", color = colors, linewidth = 2.5)
ylabels = ['{:,.0f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)
ax.set_ylabel('WUI Population')
# ax.set_ylim(0, 16e6)
ax.legend(frameon=False)