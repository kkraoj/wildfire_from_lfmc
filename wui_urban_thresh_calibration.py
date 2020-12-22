# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 03:12:06 2020

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

sns.set(font_scale = 1.1, style = "ticks")

#%%
def create_wui(forest,urban,year,save_path, forestThresh = 0, urbanThresh = 0):
    
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.overwriteOutput=True
    
    
#    year = 2016
#    forest = os.path.join(dir_root, r"data\WUI\forest%04d.tif"%year)
#    urban = os.path.join(dir_root, r"data\WUI\urban%04d.tif"%year)
    
#    forestThresh = 0
#    urbanThresh = 0.5
    if not(os.path.exists(save_path)):
        os.mkdir(savepath)
    #to int
    arcpy.env.workspace = save_path
    outRas = Raster(forest)>forestThresh
    forestName = "forest%04dInt"%year
    outRas.save(forestName+"_%02d.tif"%int(urbanThresh*100))
    
    outRas = Raster(urban)>urbanThresh
    urbanName = "urban%04dInt"%year
    
    outRas.save(urbanName+"_%02d.tif"%int(urbanThresh*100))
    
#    raster to polygon
    arcpy.RasterToPolygon_conversion(forestName+"_%02d.tif"%int(urbanThresh*100), "%s.shp"%forestName)
    arcpy.RasterToPolygon_conversion(urbanName+"_%02d.tif"%int(urbanThresh*100), "%s.shp"%urbanName)
    
#    identity
    outName = "identity%04d.shp"%year
    arcpy.Identity_analysis("%s.shp"%forestName, "%s.shp"%urbanName, outName)
    
#    buffer + 50, buffer -50 , erase
    
    arcpy.Buffer_analysis(outName, outName[:-4]+"Buffer50.shp", "50 Feet")
    arcpy.Buffer_analysis(outName, outName[:-4]+"BufferMinus50.shp", "-50 Feet")
    #erase
    arcpy.Erase_analysis(outName[:-4]+"Buffer50.shp", outName[:-4]+"BufferMinus50.shp", outName[:-4]+"Erase.shp")
    
    inFeature =  outName[:-4]+"Erase.shp"
    outFeature =  outName[:-4]+"EraseDissolve.shp"
    
    arcpy.Dissolve_management(inFeature, outFeature)
    #################################
    inFeature = [outFeature,os.path.join(dir_root,"data","WUI","arc_export","westUsa4km.shp")]
    outFeature = outName[:-4]+"EraseDissolveIntersect.shp"
    arcpy.Intersect_analysis(inFeature, outFeature)
    #project
    inFeature = outFeature
    outFeature = inFeature[:-4]+"Project.shp"
    out_coor_system = arcpy.SpatialReference(3395)#'WGS 1984 World Mercator'
    
    arcpy.management.Project(inFeature, outFeature, out_coor_system)
    
    ###########################
    inFeature = outFeature
    fieldName = "area"
    fieldPrecision = 4
    fieldScale = 2
    arcpy.AddField_management(inFeature, fieldName, "LONG")#,field_precision=fieldPrecision,field_scale=fieldScale
    #arcpy.AddField_management(inFeature, fieldName, "float",field_precision=fieldPrecision,field_scale=fieldScale)
    expression="math.ceil(!SHAPE.AREA!)"
    #expression="!SHAPE.AREA!"
    arcpy.CalculateField_management(inFeature, fieldName, expression,"PYTHON_9.3")
    ##    polygon to raster
    #outRaster = "wui_%s_%s_%04d.tif"%(forestThresh*100,urbanThresh*100,year)
    outRaster = "wui%04d"%year
    
#    print("begun")
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference("WGS 1984")
    arcpy.PolygonToRaster_conversion(inFeature, fieldName, outRaster, "MAXIMUM_AREA",fieldName,0.035932611)
#    print("done")
    toSave =Raster(outRaster)
    toSave.save(outRaster+"_%02d_%02d.tif"%(forestThresh*100,urbanThresh*100))
    
save_path = os.path.join(dir_root, "data","WUI","arc_export","urban_thresh_cal")
#
for urbanThresh in np.linspace(0.4,0.1,4):
    for year in [1992,2016]:        
        print("[INFO] Creating WUI map for year:%04d,\turbanThresh:%s"%(year,urbanThresh))
        forest = os.path.join(dir_root, r"data\WUI\forest%04d.tif"%year)
        urban = os.path.join(dir_root, r"data\WUI\urban%04d.tif"%year)
        create_wui(forest,urban,year,save_path, urbanThresh = urbanThresh)
