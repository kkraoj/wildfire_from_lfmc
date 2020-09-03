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
    shapePath = os.path.join(dir_root, "data","ecoregions_from_gee","ecoregions.shp")
    
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
    with rasterio.open(clusterPath) as src:
        cluster_image, _ = rasterio.mask.mask(src, shape, crop=True, nodata = -999)
    cluster_image = cluster_image.astype(float)
    cluster_image[cluster_image == -999] = np.nan
    return out_image, cluster_image
    
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
    shapeArea = round(prop['shape_area'])
    for zone in range(5):
        master = pd.DataFrame()
        for date in dates:
            df = pd.DataFrame()
            filename = os.path.join(dir_root, "data","lfmc_vpd_anomalies","lfmc_map_%s.tif"%date)
            out_image, cluster_image = crop_raster(filename, shape)
            # print(out_image.shape)
            # print(cluster_image.shape)
            if out_image[0,:,:].shape!=cluster_image[0,:,:].shape:
                print('\r'+'[INFO] Skipping ecoregion with area = %d, zone = %d because cropped shapes dont match'%(shapeArea,zone))
                return master
            out_image[:,cluster_image[0,:,:]!=zone] = np.nan
        
            df['lfmc(t)'] = out_image[0].flatten()
            df['vpd(t)'] = out_image[1].flatten()
            subsetDates = get_dates(date)
            ctr = 1
            sys.stdout.write('\r'+'[INFO] Time step %s'%date)
            sys.stdout.flush()
            # print(date)
    
            for t in subsetDates:
                shiftedFile = os.path.join(dir_root, "data","lfmc_vpd_anomalies","lfmc_map_%s.tif"%t)
                out_image, cluster_image = crop_raster(shiftedFile, shape)
                
                out_image[:,cluster_image[0,:,:]!=zone] = np.nan
                df['vpd(t-%d)'%ctr] = out_image[1].flatten()
                ctr+=1
            df.dropna(inplace = True)
            master = master.append(df) 
        master.to_csv(os.path.join(dir_root, "data","arr_ecoregions_L3_kmeans","%s_%1d.csv"%(shapeArea,zone)))
    return master
    

def regress(master):
    X = master.iloc[:,1:]
    y = master.iloc[:,0]
    
    reg = LinearRegression().fit(X, y)

    r2 = reg.score(X, y)
    coefs = [reg.intercept_]+list(reg.coef_)
    
    return r2, coefs
    
#%%


# shapes, props = import_ecoregions()

# years = range(2016, 2021)
# months = range(1,13)
# days = [1,15]

# dates = []
# for year in years:
#     for month in months:
#         for day in days:
#             dates+=["%s-%02d-%02d"%(year, month, day)]

# dates = dates[12:-11]
    
# # shape = [shapes[0]]    
# clusterPath = os.path.join(dir_root, "data","ecoregions_from_gee","ecoregions split by k means.tif")

# for shape, prop in zip(shapes, props):    
#     shapeArea = prop['shape_area']
#     # if int(shapeArea) == 44847075568:
#     if shapeArea>1e8:
#         print('\r'+'[INFO] Processing ecoregion with area = %d'%shapeArea)
#         create_df([shape], prop)
#         # break
    

#%% run regression 

# df = pd.read_csv(os.path.join(dir_data, "ecoregions_fire_2001_2019_no_geo.csv"))
# dfr = pd.read_csv(os.path.join(dir_data, "ecoregions_plantClimate.csv"))
# df = df.join(dfr['plantClimateSensitivity'])
# dfr = pd.read_csv(os.path.join(dir_data, "ecoregions_fire_vpd_ndvi_2001_2019_no_geo.csv"))

df = pd.read_csv(os.path.join(dir_data, "ecoregions_L3_Kmeans_fire_vpd_ndvi_2001_2019_no_geo.csv"))
# cols = [col for col in dfr.columns if 'vpd' in col]# select vpd
# cols = cols + ['ndviMean']
# dfr = dfr[cols]
# df = df.join(dfr[cols])

df['shape_area'] = df['shape_area'].astype(np.int64)
df = df.loc[df['shape_area']>1e8]
# df["plantClimateR2"] = np.nan
# df["plantClimateCoefSum"] = np.nan
# df["plantClimateCoefDiff"] = np.nan

#%% unmelt the df with each row as sub-region
variables = ["ba","vpd","ndviMean"]
ndf = pd.DataFrame()
for i in range(5):
    cols = [col for col in df.columns if col.split('_')[0] in variables]
    cols = [col for col in cols if int(col.split('_')[1])==i]
    # cols+=['shape_area']
    temp = df[cols]
    # new_col_names = 
    temp.columns = [x.split('_')[0]+ '_' + x.split('_')[-1] for x in cols]
    temp = temp.join(df['shape_area'])
    
    temp.rename(columns={"ndviMean_%1d"%i: "ndviMean"}, inplace = True)
    temp['zone'] = i
    ndf = ndf.append(temp, ignore_index = True, sort = False)

df = ndf.copy()
df["plantClimateR2"] = np.nan
df["plantClimateCoefSum"] = np.nan
df["plantClimateCoefDiff"] = np.nan

df.shape
for filename in os.listdir(os.path.join(dir_root, "data","arr_ecoregions_L3_kmeans")):
    sub = pd.read_csv(os.path.join(dir_root, "data","arr_ecoregions_L3_kmeans", filename))

    sub = sub.loc[sub['lfmc(t)']<0] #only summer time LFMC
    if sub.shape[0] < 10:
        continue
    
    r2, coefs = regress(sub)
    row = (df['shape_area']==int(filename[:-6]))&(df['zone']==int(filename[-5]))
    df.loc[row, "plantClimateR2"] = r2
    
    coefs = coefs[1:]
    minCoef = np.min(coefs)
    maxCoef = np.max(coefs)
    coefs = (coefs - minCoef) /(maxCoef - minCoef)
    
    df.loc[row, "plantClimateCoefSum"] = np.sum(coefs)
    df.loc[row, "plantClimateCoefDiff"] = np.sum(coefs[:4]) - np.sum(coefs[-4:])
    
    print('[INFO] Processing ecoregion with file = %s'%filename)
df.to_csv(os.path.join(dir_data, "arr_ecoregions_L3_kmeans_fire_climate_plant.csv"))
    

    
    