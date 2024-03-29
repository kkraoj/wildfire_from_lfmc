import os
from init import dir_data, dir_root
import gdal
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sns.set(font_scale = 1.1, style = "ticks")
#%%
def calc_wui_pc(wuiBefore,wuiAfter,wuiThresh):
        
    wuiNames = [wuiBefore, wuiAfter]
    popNames = ["pop2000.tif","pop2010.tif"]
    
    res = 3.5932611
    plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","lfmc_dfmc_100hr_lag_6_lfmc_dfmc_norm_positive_coefSum.tif")
    ds = gdal.Open(plantClimatePath)
    plantClimate = np.array(ds.GetRasterBand(1).ReadAsArray())
    
    
    #%% % absolute population timeseries split by pc quantiles
    
    ctr = 0
    for (wuiName, popName) in zip(wuiNames, popNames):
        
        fullfilename = wuiName
        # print(fullfilename)
        ds = gdal.Open(fullfilename)
        wui = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(float)
        wui[wui<0] = np.nan
        # print(wui[602,0])
        # plt.imshow(wui)
        # plt.show()
        wui[wui>1e6] = np.nan

        # plt.imshow(wui)
        # plt.show()
        # print(wui[602,0])
        wui = wui>=wuiThresh
        # print(np.nansum(wui))
    
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
        ctr+=1
        
    
    #%% growth rates for 10 bins
    
    
    nbins = 15
    cmap = plt.get_cmap('viridis',nbins)    # PiYG
    colors = [mpl.colors.rgb2hex(cmap(i))  for i in range(cmap.N)]
      
        # rgb2hex accepts rgb or rgba
    _, vulLabels = pd.qcut(df['pc'],nbins, retbins = True)
    vulLabels = np.round(vulLabels, 2)
    ts = pd.DataFrame(columns = vulLabels[:-1], index = [1990,2010])
    ctr = 0
    
    for (wuiName, popName) in zip(wuiNames, popNames):
        
        fullfilename = os.path.join(dir_root, "data","WUI","arc_export",wuiName)
        ds = gdal.Open(fullfilename)
        wui = np.array(ds.GetRasterBand(1).ReadAsArray())
        wui = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(float)
        wui[wui<0] = np.nan

        wui = wui>=wuiThresh
        
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
        ts.loc[1990+ctr*20, :] = cum
        ctr+=1
     
    # ts = ts.drop(2000)
    
    return ts.diff(), colors, vulLabels
    
    # fig, ax = plt.subplots(figsize = (3,3))
    
    # ax.bar(ts.columns,ts.diff().dropna().values.tolist()[0],align = "edge",color = colors,width = np.diff(vulLabels))
    # ax.set_xlabel("Plant climate sensitvity")
    # ax.set_ylabel("$\Delta$ WUI population")
    
    # ylabels = ['{:,.1f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
    # ax.set_yticklabels(ylabels)
    # ax.set_xlim(0,2.1)
    # # Only show ticks on the left and bottom spines
    # # Hide the right and top spines
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    
urbanThreshs = np.linspace(0.0,0.03,10)
df = pd.DataFrame(index = urbanThreshs, columns = range(15))

for urbanThresh in urbanThreshs:
    print(urbanThresh)
    df_ ,colors, vulLabels= calc_wui_pc(os.path.join(dir_root, "data","WUI","urban2001NeighborsResampledGee.tif"),\
                    os.path.join(dir_root, "data","WUI","urban2016NeighborsResampledGee.tif"),\
                    urbanThresh)
    df.loc[urbanThresh, :] = df_.dropna().values
          


          
df.columns = df_.columns
fig, ax = plt.subplots(figsize = (3,3))

ax.bar(df.columns,df.mean().values.tolist()[0],align = "edge",color = colors,width = np.diff(vulLabels), yerr = df.std())
ax.set_xlabel("Plant climate sensitvity")
ax.set_ylabel("$\Delta$ WUI population")

ylabels = ['{:,.1f}'.format(x) + 'M' for x in ax.get_yticks()/1e6]
ax.set_yticklabels(ylabels)
ax.set_xlim(0,2.1)
# Only show ticks on the left and bottom spines
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
