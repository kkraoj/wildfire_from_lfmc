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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)



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
    

#%% growth rates for 10 bins


nbins = 10
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
 
ts = ts.drop(2000)

fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(ts.loc[1990],ts.loc[2010],c = ts.columns, cmap = "viridis",s = 80, edgecolor  = "grey")
ax.set_xlabel("1990 WUI population")
ax.set_ylabel("2010 WUI population")
# ax.set_xlim(xmin = 0)
# ax.set_ylim(ymin = 0)
ylabels = ['{:,.1f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)

xlabels = ['{:,.1f}'.format(x) + 'M' for x in ax.get_xticks()/1e6]
ax.set_xticklabels(xlabels)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')



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
    
ax.set_xlabel("PWS")
ax.set_ylabel("Density")

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

fig, ax = plt.subplots(figsize = (3,3))

ax.bar(ts.columns,ts.diff().dropna().values.tolist()[0],align = "edge",color = colors,width = np.diff(vulLabels))
ax.set_xlabel("PWS")
ax.set_ylabel("WUI population rise")

ylabels = ['{:,.1f}'.format(x) + ' M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)
ax.set_xlim(0,2.1)
# Only show ticks on the left and bottom spines
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

width = 10

xticks = np.linspace(0,100-width,nbins)
fig, ax = plt.subplots(figsize = (3,3))
ax.bar(xticks,ts.diff().dropna().values.tolist()[0],align = "edge",\
       color = colors,width = width,edgecolor = "k",linewidth = 1.5)
    

ax.hlines(y = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
           xmin = xticks[0], xmax = xticks[int(nbins/2)],linestyle = "--", \
               color = "k")
    
ax.fill_between(x = xticks[:int(nbins/2)+1],\
    y1 = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]) - \
        np.std(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
    y2 = np.mean(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]) + \
        np.std(ts.diff().dropna().values.tolist()[0][:int(nbins/2)]), \
    color = "k",alpha = 0.2)
ax.fill_between(x = np.linspace(50,100,6),\
    y1 = np.mean(ts.diff().dropna().values.tolist()[0][int(nbins/2):]) - \
        np.std(ts.diff().dropna().values.tolist()[0][int(nbins/2):]), \
    y2 = np.mean(ts.diff().dropna().values.tolist()[0][int(nbins/2):]) + \
        np.std(ts.diff().dropna().values.tolist()[0][int(nbins/2):]), \
    color = "k",alpha = 0.2)
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
