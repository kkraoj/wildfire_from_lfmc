# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:30:10 2020

@author: kkrao
"""

import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from init import dir_root, dir_data
import fiona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import mapping
from pandas.tseries.offsets import DateOffset
from sklearn.linear_model import LinearRegression
import sys
import gdal
import matplotlib as mpl


def get_dates(date):
    subsetDates = []
    for delta in range(1, 6):
        shiftedDate = pd.to_datetime(date,format = "%Y-%m-%d") - DateOffset(months = delta)
        shiftedDate = shiftedDate.date().strftime(format = "%Y-%m-%d")                    
        for day in [1,15]:
            subsetDates+= [shiftedDate[:-2]+"%02d"%day]
    
    subsetDates = pd.to_datetime(subsetDates,format = "%Y-%m-%d")
    subsetDates = list(subsetDates.sort_values(ascending = False).strftime(date_format = "%Y-%m-%d") )
    
    return subsetDates        
    
def create_time_df():
    date = "2016-01-01"
    filename = os.path.join(dir_root, "data","lfmc_vpd_raw","lfmc_map_%s.tif"%date)
    ds = gdal.Open(filename)
    array = np.array(ds.GetRasterBand(1).ReadAsArray())
    
    
    x_loc, y_loc = np.meshgrid(range(array.shape[1]),range(array.shape[0]) )
    
    master = pd.DataFrame()
    for date in dates:
        df = pd.DataFrame()
        filename = os.path.join(dir_root, "data","lfmc_vpd_raw","lfmc_map_%s.tif"%date)
        ds = gdal.Open(filename)
        lfmc = np.array(ds.GetRasterBand(1).ReadAsArray())
        vpd = np.array(ds.GetRasterBand(2).ReadAsArray())
        # out_image[:,cluster_image[0,:,:]!=zone] = np.nan
        
        
        
        
        df['lfmc(t)'] = lfmc.flatten()
        df['vpd(t)'] = vpd.flatten()
        df['x_loc'] = x_loc.flatten()
        df['y_loc'] = y_loc.flatten()
        df['pixel_index'] = df.index
        
        df['date'] = date
        subsetDates = get_dates(date)
        ctr = 1
        sys.stdout.write('\r'+'[INFO] Time step %s'%date)
        sys.stdout.flush()
        # print(date)

        for t in subsetDates:
            shiftedFile = os.path.join(dir_root, "data","lfmc_vpd_raw","lfmc_map_%s.tif"%t)
            filename = os.path.join(dir_root, "data","lfmc_vpd_raw","lfmc_map_%s.tif"%date)
            ds = gdal.Open(filename)
            df['vpd(t-%d)'%ctr] = np.array(ds.GetRasterBand(2).ReadAsArray()).flatten()
            ctr+=1
        # df.dropna(inplace = True)
        master = master.append(df,ignore_index = True) 
    master.to_pickle(os.path.join(dir_root, "data","arr_pixels_raw","arr_pixels_time_wise"))
    return master
    

def regress(df):                    
    cols = [col for col in df.columns if "vpd" in col]
    X = df.loc[:,cols]
    y = df.iloc[:,0]

    reg = LinearRegression().fit(X, y)

    r2 = reg.score(X, y)
    coefs = [reg.intercept_]+list(reg.coef_)
    
    return r2, coefs, df['x_loc'].iloc[0],df['y_loc'].iloc[0]
    
#%%

years = range(2016, 2021)
months = range(1,13)
days = [1,15]

dates = []
for year in years:
    for month in months:
        for day in days:
            dates+=["%s-%02d-%02d"%(year, month, day)]

dates = dates[12:-11]
    

# create_time_df()
master = pd.read_pickle(os.path.join(dir_root, "data","arr_pixels_raw","arr_pixels_time_wise"))

master = master.dropna()
master.date = pd.to_datetime(master.date)
master = master.loc[master.date.dt.month>=6]

out = master.groupby('pixel_index').apply(regress)
r2 = [x[0] for x in out]
x_loc = [x[2] for x in out]
y_loc = [x[3] for x in out]

fig, ax = plt.subplots(figsize = (3,3))
ax.hist(r2,bins = 100, histtype=u'step', color = "magenta", linewidth = 2)
ax.set_xlim(0,1)
ax.set_xlabel("$R^2$")
ax.set_ylabel("Frequency")


fig, ax = plt.subplots(figsize = (3,3))

cmap = plt.cm.viridis  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds = np.linspace(0, 1, 21)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

sc = ax.scatter(x_loc,np.max(y_loc) - y_loc,c = r2, vmin = 0, vmax = 1, cmap = cmap,marker = "s", s = 0.05,norm = norm)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

# create a second axes for the colorbar
ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
    spacing='proportional', boundaries=bounds, format='%0.2f')

ax2.set_ylabel('$R^2$', size=12)

plt.show()




#%%

# filename = os.path.join(dir_root, "data","mean","vpd_mean.tif")
# ds = gdal.Open(filename)
# vpd = np.array(ds.GetRasterBand(1).ReadAsArray())


# filename = os.path.join(dir_root, "data","mean","ndvi_mean.tif")
# ds = gdal.Open(filename)
# ndvi = np.array(ds.GetRasterBand(1).ReadAsArray())



# plantClimate = ndvi.copy()
# plantClimate[:,:] = np.nan
# plantClimate[y_loc, x_loc] = r2
# plt.imshow(plantClimate,vmin = 0, vmax = 1)

# fig, ax = plt.subplots(figsize = (3,3))
# ax.scatter(vpd, plantClimate, alpha = 0.3, s = 0.001, color = "k")
# # ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_xlabel("VPD (Hpa)")
# ax.set_ylabel("$R^2(LFMC, DFMC_{100hr})$")

# fig, ax = plt.subplots(figsize = (3,3))
# ax.scatter(ndvi, plantClimate, alpha = 0.3, s = 0.001, color = "k")
# # ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_xlabel("NDVI")
# ax.set_ylabel("$R^2(LFMC, DFMC_{100hr})$")

# np.corrcoef(vpd.flatten(), plantClimate.flatten())
# data = pd.DataFrame({ "plantClimate":plantClimate.flatten(),"ndvi":ndvi.flatten(),"vpd":vpd.flatten()})
# (data.corr()**2).round(2)


# df = pd.DataFrame(columns = ["pixel", "plantClimateR2","plantClimateCoefSum","plantClimateCoefDiff"])
# # df["plantClimateR2"] = np.nan
# # df["plantClimateCoefSum"] = np.nan
# # df["plantClimateCoefDiff"] = np.nan

# for pixel in master.pixel_index.unique():
#     sub = master.loc[master.pixel_index==pixel]
#     sub = sub.dropna()
    
#     if sub.shape[0]>30:
#         print('[INFO] Processing pixel = %d'%pixel)
#         r2, coefs = regress(sub.dropna())
#         row = pd.Series()
#         row['pixel'] = pixel
#         row["plantClimateR2"] = r2
        
#         coefs = coefs[1:]
#         minCoef = np.min(coefs)
#         maxCoef = np.max(coefs)
#         coefs = (coefs - minCoef) /(maxCoef - minCoef)
    
#         row["plantClimateCoefSum"] = np.sum(coefs)
#         row["plantClimateCoefDiff"] = np.sum(coefs[:4]) - np.sum(coefs[-4:])
        
#         df = df.append(row,ignore_index = True)
        
#     else:
#         print('[INFO] Skipping pixel = %d'%pixel)
# df.to_csv(os.path.join(dir_data, "arr_ecoregions_L3_kmeans_fire_climate_plant.csv"))
    

    
    