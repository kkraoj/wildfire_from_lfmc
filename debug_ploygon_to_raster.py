# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 04:11:44 2020

@author: kkrao
"""

import os
from init import dir_data, dir_root
import arcpy 

from arcpy import env
from arcpy.sa import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput=True
save_path = os.path.join(dir_root, "data","WUI","arc_export","urban_thresh_cal","playground")

arcpy.env.workspace = save_path

inFeature = r"D:\Krishna\projects\wildfire_from_lfmc\data\WUI\arc_export\urban_thresh_cal\identity2016EraseDissolveIntersectProject.shp"
fieldName = 'area'
outRaster = r"D:\Krishna\projects\wildfire_from_lfmc\data\WUI\arc_export\urban_thresh_cal\playground\trial"
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("WGS 1984")


arcpy.PolygonToRaster_conversion(inFeature, fieldName, outRaster, "MAXIMUM_AREA",fieldName,0.035932611)
#arcpy.PolygonToRaster_conversion(inFeature, fieldName, outRaster, "MAXIMUM_AREA",fieldName)
#toSave =Raster(outRaster)
#toSave.save(outRaster+"_%02d_%02d.tif"%(forestThresh*100,urbanThresh*100))
