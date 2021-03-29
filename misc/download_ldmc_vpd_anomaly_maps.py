# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:14:09 2020

@author: kkrao
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 08:30:32 2018

@author: kkrao
"""
import ee
from ee import batch
from pandas.tseries.offsets import DateOffset
import pandas as pd
import folium


## Initialize (a ee python thing)

ee.Initialize()

####################################################

prismMean = ee.ImageCollection("OREGONSTATE/PRISM/Norm81m")
roi = ee.FeatureCollection("users/kkraoj/west_usa")
lfmc =  ee.ImageCollection("users/kkraoj/lfm-mapper/lfmc_col_24_jul_2020")
prism =  ee.ImageCollection("OREGONSTATE/PRISM/AN81d")

# start_date = '2019-06-30'; 
# end_date = '2019-07-14';

lfmc = lfmc.select(['b1'],['lfmc'])


nImages = lfmc.count().select("lfmc")

threshCount = 50

lfmcMin = ee.Image(lfmc.min())
lfmcMax = ee.Image(lfmc.max())

def minMax(image):
    start = image.get("system:time_start")
    end = image.get("system:time_end")

    image = image.subtract(lfmcMin).divide(lfmcMax.subtract(lfmcMin))
    
    image = image.set("system:time_start",start)
    image = image.set("system:time_end", end)
    return image
lfmc = lfmc.map(minMax) #should be performed before seasonal cycle computation

def countMask(image):
  image = image.updateMask(nImages.gte(threshCount))
  return image

months = ee.List.sequence(1,12);
def lfmcSeasonalCycle(m):
    return lfmc.filter(ee.Filter.calendarRange(m, m, 'month')).mean().set('month', m)

lfmcMean = ee.ImageCollection.fromImages(\
                months.map(lfmcSeasonalCycle))

# print(lfmcMean.size().getInfo())
def calcLfmcAnom(image):
  start = ee.Date(image.get("system:time_start"))
  end = ee.Date(image.get("system:time_end"))
  month = start.get('month')
  image = image.subtract(lfmcMean.filter(ee.Filter.eq('month',month)).mean()) #.mean(0 is required to convert imagecollection of 1 image to image)
  image = image.set("system:time_start",start)
  image = image.set("system:time_end", end)
  
  return image


# def shiftYear(image):
#   start = ee.Date(image.get("system:time_start")).advance(-1,"year")
#   end = ee.Date(image.get("system:time_end")).advance(-1,"year")
  
#   image = image.set("system:time_start",start.millis())
#   image = image.set("system:time_end", end.millis())
  
#   return image

# prismMeanRepeat = prismMean.map(shiftYear)

# prismMean = prismMean.merge(prismMeanRepeat)

# def aggregate(image):
#   start = ee.Date(image.get("system:time_start"))
#   month = start.get("month")
#   startAdvanced = start.advance(-6,"month")
#   end = ee.Date(image.get("system:time_end"))
#   image = prismMean.filterDate(startAdvanced, end).mean()
  
#   image = image.set("system:time_start",start)
#   image = image.set("system:time_end", end)
#   image = image.set("month",month)
  
#   return image

# prismMeanAggregated = prismMean.map(aggregate).filterDate('1981-01-01', '1981-12-31')

def addMonth(image):
    month = ee.Date(image.get("system:time_start")).get("month")
    image = image.set('month', month)
    return image
    
    

prismMean = prismMean.map(addMonth)

def addPrismAnom(image):
  
  start = ee.Date(image.get("system:time_start"))
  month = start.get('month')
  # startAdvanced = start.advance(-6,"month")
  end = ee.Date(image.get("system:time_end"))
  prism_ = prism.filterDate(start, end).mean()
  
  prismAnom = prism_.subtract(prismMean.filter(ee.Filter.eq('month',month)).mean()).select("vpdmax")
  
  # image = image.multiply(ee.Image.constant(-1))
  image = image.set("system:time_start",start)
  image = image.set("system:time_end", end)
  image = image.addBands(prismAnom)
  return image

def convert_to_float(image):
    return image.toFloat()

lfmcAnom = lfmc.map(calcLfmcAnom).map(addPrismAnom)
# lfmcAnom = lfmcAnom.map(countMask)
# print(lfmcAnom.first().getInfo())
# lfmcAnom = lfmcAnom.cast({'lfmc':'float','vpdmax':'float'}, ['lfmc','vpdmax'])
lfmcAnom = lfmcAnom.map(convert_to_float)

# # Define a method for displaying Earth Engine image tiles to folium map.
# def add_ee_layer(self, eeImageObject, visParams, name):
#   mapID = ee.Image(eeImageObject).getMapId(visParams)
#   folium.raster_layers.TileLayer(
#     tiles = "https://earthengine.googleapis.com/map/"+mapID['mapid']+
#       "/{z}/{x}/{y}?token="+mapID['token'],
#     attr = "Map Data © Google Earth Engine",
#     name = name,
#     overlay = True,
#     control = True
#     ).add_to(self)
  
# folium.Map.add_ee_layer = add_ee_layer


# Map = folium.Map(location=[38.594405,  -109.976566], zoom_start=8)

# visParams = {'min': -1, 'max': 1, 'palette': ['blue', 'white', 'green']}
# Map.add_ee_layer(lfmcAnom.first().select('lfmc'), visParams, 'LFMC Anoms')
# # Map.setControlVisibility(layerControl=True, fullscreenControl=True, latLngPopup=True)
# Map


# export from here onwartds
# lfmcAnom = lfmcAnom.filterDate(start_date, end_date)
print(lfmcAnom.size().getInfo())

n = lfmcAnom.size().getInfo() # number of images to download
    
colList = lfmcAnom.toList(n)

folder_name = "lfmc_vpd_anomalies"  
scale = 4000

for i in range(n):
    image = ee.Image(colList.get(i));
    id = image.id().getInfo() or 'image_'+i.toString();

    out = batch.Export.image.toDrive(
      image=image,
      folder=folder_name,
      description = id,
      scale= scale,
      maxPixels=1e11,
      region = roi.geometry()
    );
    batch.Task.start(out)    
## process the image

out.status()
print("process sent to cloud")


