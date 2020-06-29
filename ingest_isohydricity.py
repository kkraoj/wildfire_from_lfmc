# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:44:54 2020

@author: kkrao
"""


import gdal
import pandas as pd
import numpy as np
import os
import osr
from init import dir_root, lc_dict, color_dict
import matplotlib.pyplot as plt


# for datafile in ["CWM_KL_10Deg.nc"]:

FILENAME = "isohydricityAMSRE_US.nc"
FEATURE = "isohydricity"
SAVENAME = "%s.tif"%FEATURE

datafile= os.path.join(dir_root,"data", "traits",FILENAME)

ds = gdal.Open('NETCDF:"'+datafile+'":%s'%FEATURE)
gt = ds.GetGeoTransform()
data = ds.ReadAsArray()
data[data>1] = 9999
data = np.rot90(data, k = 3)
# data = np.flip(data, axis = 0).astype(np.float32)


data.max()
data.min()
plt.imshow(data, vmin = -1, vmax = 1)

# data.g
data.shape
# Read full data from netcdf
# data[data < 0] = 0


driver = gdal.GetDriverByName( 'GTiff' )
dst_filename = os.path.join(dir_root,"data", "traits",SAVENAME)
dst_ds=driver.Create(dst_filename,data.shape[1],data.shape[0],1,gdal.GDT_Float32)
# print(gt)

dst_ds.SetGeoTransform([-125, 0.25, 0, 50, 0, -.25 ])
srs = osr.SpatialReference()
# >>> srs.SetUTM( 11, 1 )
srs.SetWellKnownGeogCS( 'WGS84' )
dst_ds.SetProjection( srs.ExportToWkt() )
# raster = numpy.zeros( (512, 512) )
dst_ds.GetRasterBand(1).WriteArray( data )

dst_ds = None
driver  = None
