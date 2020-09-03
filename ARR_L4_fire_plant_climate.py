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



def import_ecoregions():
    shapePath = os.path.join(dir_root, "data","ecoregions_from_gee","ecoregions_L4.shp")
    
    with fiona.open(shapePath, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        props = [feature["properties"] for feature in shapefile]
    
    return shapes, props

# for ind in range(127):
#     with rasterio.open(rasterPath) as src:
#         out_image, out_transform = rasterio.mask.mask(src, [shapes[ind]], crop=True)
        


def crop_raster(rasterPath, shape):
    with rasterio.open(rasterPath) as src:
        out_image, _ = rasterio.mask.mask(src, shape, crop=True, nodata = np.nan)
    
    return out_image
    
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
    
# filenames = os.listdir()

def create_df(shape, prop):
    
    master = pd.DataFrame()
    for date in dates:
        df = pd.DataFrame()
        filename = os.path.join(dir_root, "data","lfmc_vpd_anomalies","lfmc_map_%s.tif"%date)
        
        out_image = crop_raster(filename, shape)
        df['lfmc(t)'] = out_image[0].flatten()
        df['vpd(t)'] = out_image[1].flatten()
        subsetDates = get_dates(date)
        ctr = 1
        sys.stdout.write('\r'+'[INFO] Time step %s'%date)
        sys.stdout.flush()
        # print(date)

        for t in subsetDates:
            shiftedFile = os.path.join(dir_root, "data","lfmc_vpd_anomalies","lfmc_map_%s.tif"%t)
            out_image = crop_raster(shiftedFile, shape)
            df['vpd(t-%d)'%ctr] = out_image[1].flatten()
            ctr+=1
        df.dropna(inplace = True)
        master = master.append(df)
        # except:
            # continue
        
    shapeArea = round(prop['shape_area'])
    
    master.to_csv(os.path.join(dir_root, "data","arr_ecoregions_L4","%s.csv"%shapeArea))
    return master
    

def regress(master):
    X = master.iloc[:,1:]
    y = master.iloc[:,0]
    
    reg = LinearRegression().fit(X, y)

    r2 = reg.score(X, y)
    coefs = [reg.intercept_]+list(reg.coef_)
    
    return r2, coefs
    
#%%


shapes, props = import_ecoregions()

years = range(2016, 2021)
months = range(1,13)
days = [1,15]

dates = []
for year in years:
    for month in months:
        for day in days:
            dates+=["%s-%02d-%02d"%(year, month, day)]

dates = dates[12:-11]
    
# shape = [shapes[0]]    

for shape, prop in zip(shapes, props):    
    shapeArea = prop['shape_area']
    if shapeArea>1e8:
        print('\r'+'[INFO] Processing ecoregion with area = %d'%shapeArea)
        create_df([shape], prop)
    

#%% run regression 

# df = pd.read_csv(os.path.join(dir_data, "ecoregions_fire_2001_2019_no_geo.csv"))
# dfr = pd.read_csv(os.path.join(dir_data, "ecoregions_plantClimate.csv"))
# df = df.join(dfr['plantClimateSensitivity'])
# dfr = pd.read_csv(os.path.join(dir_data, "ecoregions_fire_vpd_ndvi_2001_2019_no_geo.csv"))
# cols = [col for col in dfr.columns if 'vpd' in col]# select vpd
# cols = cols + ['ndviMean']
# dfr = dfr[cols]
# df = df.join(dfr[cols])

# df['shape_area'] = df['shape_area'].astype(np.int64)
# df = df.loc[df['shape_area']>1e8]
# df["plantClimateR2"] = np.nan
# df["plantClimateCoefSum"] = np.nan
# df["plantClimateCoefDiff"] = np.nan
   
# for filename in os.listdir(os.path.join(dir_root, "data","arr_ecoregions")):
#     sub = pd.read_csv(os.path.join(dir_root, "data","arr_ecoregions", filename))
#     if sub.shape[0] < 100:
#         continue
#     r2, coefs = regress(sub)
    
#     df.loc[df['shape_area']==int(filename[:-4]), "plantClimateR2"] = r2
    
#     coefs = coefs[1:]
#     minCoef = np.min(coefs)
#     maxCoef = np.max(coefs)
#     coefs = (coefs - minCoef) /(maxCoef - minCoef)
    
#     df.loc[df['shape_area']==int(filename[:-4]), "plantClimateCoefSum"] = np.sum(coefs)
#     df.loc[df['shape_area']==int(filename[:-4]), "plantClimateCoefDiff"] = np.sum(coefs[:4]) - np.sum(coefs[-4:])
    
#     print('[INFO] Processing ecoregion with file = %s'%filename)
# df.to_csv(os.path.join(dir_data, "arr_ecoregions_fire_climate_plant.csv"))
    

    
    