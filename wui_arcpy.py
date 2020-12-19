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

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput=True


year = 1992
forest = os.path.join(dir_root, r"data\WUI\forest%04d.tif"%year)
urban = os.path.join(dir_root, r"data\WUI\urban%04d.tif"%year)

forestThresh = 0
urbanThresh = 0


#to int
arcpy.env.workspace = os.path.join(dir_root, r"data\WUI\arc_export")
outRas = Raster(forest)>forestThresh
forestName = "forest%04dInt"%year
outRas.save(forestName+".tif")

outRas = Raster(urban)>urbanThresh
urbanName = "urban%04dInt"%year

outRas.save(urbanName+".tif")

#raster to polygon
arcpy.RasterToPolygon_conversion("%s.tif"%forestName, "%s.shp"%forestName)
arcpy.RasterToPolygon_conversion("%s.tif"%urbanName, "%s.shp"%urbanName)

#identity
outName = "identity%04d.shp"%year
arcpy.Identity_analysis("%s.shp"%forestName, "%s.shp"%urbanName, outName)

#buffer + 50, buffer -50 , erase

arcpy.Buffer_analysis(outName, outName[:-4]+"Buffer50.shp", "50 Feet")
arcpy.Buffer_analysis(outName, outName[:-4]+"BufferMinus50.shp", "-50 Feet")
#erase
arcpy.Erase_analysis(outName[:-4]+"Buffer50.shp", outName[:-4]+"BufferMinus50.shp", outName[:-4]+"Erase.shp")

inFeature =  outName[:-4]+"Erase.shp"
outFeature =  outName[:-4]+"EraseDissolve.shp"

arcpy.Dissolve_management(inFeature, outFeature)
#################################
inFeature = [outFeature,"westUsa4km.shp"]
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

print("begun")
arcpy.conversion.PolygonToRaster(inFeature, fieldName, outRaster, "MAXIMUM_AREA",fieldName,0.035932611,False)
print("done")
toSave =Raster(outRaster)
toSave.save(outRaster+"_%s_%s.tif"%(forestThresh*100,urbanThresh*100))