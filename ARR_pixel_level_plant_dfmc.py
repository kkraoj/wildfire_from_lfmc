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
from osgeo import gdal, osr


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
    filename = os.path.join(dir_root, "data","lfmc_dfmc_raw","lfmc_map_%s.tif"%date)
    ds = gdal.Open(filename)
    array = np.array(ds.GetRasterBand(1).ReadAsArray())
    
    
    x_loc, y_loc = np.meshgrid(range(array.shape[1]),range(array.shape[0]) )
    
    master = pd.DataFrame()
    for date in dates:
        df = pd.DataFrame()
        filename = os.path.join(dir_root, "data","lfmc_dfmc_raw","lfmc_map_%s.tif"%date)
        ds = gdal.Open(filename)
        lfmc = np.array(ds.GetRasterBand(1).ReadAsArray())
        dfmc = np.array(ds.GetRasterBand(2).ReadAsArray())
        # out_image[:,cluster_image[0,:,:]!=zone] = np.nan
        
        
        
        
        df['lfmc(t)'] = lfmc.flatten()
        df['dfmc(t)'] = dfmc.flatten()
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
            shiftedFile = os.path.join(dir_root, "data","lfmc_dfmc_raw","lfmc_map_%s.tif"%t)
            filename = os.path.join(dir_root, "data","lfmc_dfmc_raw","lfmc_map_%s.tif"%date)
            ds = gdal.Open(filename)
            df['dfmc(t-%d)'%ctr] = np.array(ds.GetRasterBand(2).ReadAsArray()).flatten()
            ctr+=1
        # df.dropna(inplace = True)
        master = master.append(df,ignore_index = True) 
    master.to_pickle(os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_raw","arr_pixels_time_wise_100hr"))
    return master
    

def regress(df):            
    cols = [col for col in df.columns if "dfmc" in col]        
    X = df.loc[:,cols]
    y = df.iloc[:,0]
    y = (y - np.mean(y))/np.std(y)
    reg = LinearRegression().fit(X, y)

    r2 = reg.score(X, y)
    coefs = [reg.intercept_]+list(reg.coef_)
    
    return r2, coefs, df['x_loc'].iloc[0],df['y_loc'].iloc[0]

def save_tif(data, geotransform, savepath = None):
    
    nrows, ncols = data.shape
    
    output_raster = gdal.GetDriverByName('GTiff').Create(savepath,ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
    srs = osr.SpatialReference()                 # Establish its coordinate encoding
    srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                                # Anyone know how to specify the 
                                                 # IAU2000:49900 Mars encoding?
    output_raster.SetProjection(srs.ExportToWkt() )   # Exports the coordinate system 
                                                       # to the file
    output_raster.GetRasterBand(1).WriteArray(data)   # Writes my array to the raster
    
    output_raster.FlushCache()
    output_raster = None 
    print("output_saved")
    
    
    
    
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
# master = pd.read_pickle(os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_raw","arr_pixels_time_wise_100hr"))

# master = master.dropna()
# master.date = pd.to_datetime(master.date)
# master = master.loc[master.date.dt.month>=6]

# out = master.groupby('pixel_index').apply(regress)
# out.to_pickle(os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_raw","plant_climate_regressed_normalized_100hr"))
out = pd.read_pickle(os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_raw","plant_climate_regressed_normalized_1000hr"))
 

#%% r2
r2 = [x[0] for x in out]
x_loc = [x[2] for x in out]
y_loc = [x[3] for x in out]


filename = os.path.join(dir_root, "data","mean","vpd_mean.tif")
ds = gdal.Open(filename)
vpd = np.array(ds.GetRasterBand(1).ReadAsArray())


filename = os.path.join(dir_root, "data","mean","ndvi_mean.tif")
ds = gdal.Open(filename)
ndvi = np.array(ds.GetRasterBand(1).ReadAsArray())
geotransform = ds.GetGeoTransform()

plantClimate = ndvi.copy()
plantClimate[:,:] = np.nan
plantClimate[y_loc, x_loc] = r2

# plantClimate[np.isnan(plantClimate)] = -9999
savepath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_raw","lfmc_dfmc_100hr_normalized_r2.tif")
# save_tif(plantClimate, geotransform, savepath)

#%% variants with coefs
coefs = [x[1] for x in out]
coefSum = [np.sum(x[1:]) for x in coefs]
coefAbsSum = [np.sum(np.abs(x[1:])) for x in coefs]

coefDiff = [np.sum(x[1:5]) - np.sum(x[8:12]) for x in coefs]
coefAbsDiff = [np.sum(np.abs(x[1:5])) - np.sum(np.abs(x[8:12])) for x in coefs]

plantClimate = ndvi.copy()
plantClimate[:,:] = np.nan
plantClimate[y_loc, x_loc] = coefAbsSum
plantClimate = np.clip(plantClimate,np.nanquantile(plantClimate, q = 0.01),np.nanquantile(plantClimate, q = 0.99)) 
savepath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_raw","lfmc_dfmc_100hr_normalized_coefAbsSum.tif")
# save_tif(plantClimate, geotransform, savepath)

plantClimate = ndvi.copy()
plantClimate[:,:] = np.nan
plantClimate[y_loc, x_loc] = coefAbsDiff
plantClimate = np.clip(plantClimate,np.nanquantile(plantClimate, q = 0.01),np.nanquantile(plantClimate, q = 0.99)) 
savepath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_raw","lfmc_dfmc_1000hr_normalized_coefAbsSum.tif")
# save_tif(plantClimate, geotransform, savepath)


#%%plots

fig, ax = plt.subplots(figsize = (3,3))
ax.hist([x for x in r2 if x>=0],bins = 100, histtype=u'step', color = "darkgoldenrod", linewidth = 2)
ax.set_xlim(0,1.0)
ax.set_ylim(0,17500)
ax.set_xlabel("$R^2$")
ax.set_ylabel("Frequency")
###############################################################################

fig, ax = plt.subplots(figsize = (3,3))
cmap = plt.cm.viridis  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

# define the bins and normalize

vmin = np.nanquantile(plantClimate, q = 0.05)
vmax = np.nanquantile(plantClimate, q = 0.95)
bounds = np.linspace(vmin, vmax, 15)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

sc = ax.imshow(plantClimate,vmin = vmin, vmax = vmax, cmap = cmap)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
# create a second axes for the colorbar
ax2 = fig.add_axes([0.95, 0.21, 0.05, 0.58])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
    spacing='proportional', boundaries=bounds)

ax2.set_ylabel('$R^2$', size=12)

plt.show()
###############################################################################

fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(vpd, plantClimate, alpha = 0.3, s = 0.001, color = "k")
# ax.set_xlim(0,1)
ax.set_ylim(vmin, vmax)
ax.set_xlabel("VPD (Hpa)")
ax.set_ylabel("$R^2(LFMC, DFMC_{100hr})$")
###############################################################################
fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(ndvi, plantClimate, alpha = 0.3, s = 0.001, color = "k")
# ax.set_xlim(0,1)
ax.set_ylim(vmin, vmax)
ax.set_xlabel("NDVI")
ax.set_ylabel("$R^2(LFMC, DFMC_{100hr})$")
    

data = pd.DataFrame({ "plantClimate":plantClimate.flatten(),"ndvi":ndvi.flatten(),"vpd":vpd.flatten()})
print((data.corr()**2).round(2))
###############################################################################