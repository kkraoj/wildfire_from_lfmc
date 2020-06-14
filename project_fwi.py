# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:17:50 2020

@author: kkrao
"""

import os
import gdal
from init import dir_root

os.chdir(os.path.join(dir_root,"data", "FWI","python_export"))
for filename in os.listdir(os.path.join(dir_root,"data", "FWI","python_export")):
    # print(filename)
    # filename = r"C:\path\to\input\raster
    input_raster = gdal.Open(filename)
    output_raster = os.path.join(dir_root,"data", "FWI","python_export_projected",filename)
    print(output_raster)
    gdal.Warp(output_raster,input_raster,dstSRS='EPSG:4326')
    
input_raster = None
