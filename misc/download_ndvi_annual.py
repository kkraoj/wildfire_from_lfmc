# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:14:09 2020

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

roi = ee.FeatureCollection("users/kkraoj/west_usa")
ndviCol = ee.ImageCollection("MODIS/006/MOD13A2")


ndviCol = ndviCol.filter(ee.Filter.calendarRange(3, 9, 'month'))


def addMonthYear(image):
  month = ee.Date(image.get("system:time_start")).get("month")
  year = ee.Date(image.get("system:time_start")).get("year")
  image = image.set("month", month)
  image = image.set("year", year)
  
  return image

ndviCol = ndviCol.map(addMonthYear)
# ndviCol.select("NDVI").getInfo()

years = ee.List.sequence(2001,2020);


def ndviAnnualMean(y):
    # yr = ee.Number(y)
    ndvi_ = ndviCol.filter(ee.Filter.eq('year',y)).mean().divide(1e4).select("NDVI")
    
    start =ee.Date.fromYMD(year = y, month = 1, day = 1)
    end =ee.Date.fromYMD(year = y, month = 12, day = 31)
    ndvi_ = ndvi_.set("system:time_start", start)
    ndvi_ = ndvi_.set("system:time_end", end)
    
    return ndvi_

ndvi = ee.ImageCollection(years.map(ndviAnnualMean))

def convert_to_float(image):
    return image.toFloat()

# print(ndvi.getInfo())
# prism = prism.map(convert_to_float)

print(ndvi.size().getInfo())

n = ndvi.size().getInfo() # number of images to download
    
colList = ndvi.toList(n)

folder_name = "ndvi_annual_apr_sep"  
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


