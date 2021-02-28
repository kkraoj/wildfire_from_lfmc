# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 04:29:08 2020

@author: kkrao
"""

#import arcpy
import os
import glob
from init import dir_root
import arcpy
from datetime import datetime

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput=True


files =""
for file in glob.glob(os.path.join(dir_root, "data","WUI","30m","urban2011*.tif")):
    files = files+file+";"
files = files[:-1]
arcpy.env.workspace = os.path.join(dir_root, "data","WUI","30m")

##Mosaic several TIFF images to a new TIFF image
print("[INFO] Mosaic started at %s"%datetime.now().strftime("%H:%M:%S"))
arcpy.MosaicToNewRaster_management(files,os.path.join(dir_root, "data","WUI","30m"), "urban2011mosaic.tif", number_of_bands = 1)

print("[INFO] Mosaic done at %s"%datetime.now().strftime("%H:%M:%S"))
