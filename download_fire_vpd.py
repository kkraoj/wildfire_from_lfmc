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
fireCol = ee.ImageCollection("MODIS/006/MCD64A1")
prism = ee.ImageCollection("OREGONSTATE/PRISM/AN81m")

prism = prism.filterDate("2001-01-01","2019-12-31").filter(ee.Filter.calendarRange(3, 9, 'month'))

fireCol = fireCol.filterDate("2001-01-01","2019-12-31")


def addMonthYear(image):
  month = ee.Date(image.get("system:time_start")).get("month")
  year = ee.Date(image.get("system:time_start")).get("year")
  image = image.set("month", month)
  image = image.set("year", year)
  
  return image

fireCol = fireCol.map(addMonthYear)


years = ee.List.sequence(2001,2019);


def addAnnualBurnedArea(image):
    
    start = ee.Date(image.get("system:time_start"))
    end = ee.Date(image.get("system:time_end"))
    
    yr = ee.Number(image.get("year"))
    filter_ = ee.Filter.Or(ee.Filter.And(ee.Filter.eq('year',yr.add(1)),ee.Filter.lte('month',2)),
            ee.Filter.And(ee.Filter.eq('year',yr),ee.Filter.gte('month',3))) #// Mar - Dec of current year, and Jan - Feb of next year
    fireColyear = fireCol.filter(filter_).mosaic();
    
    image = image.addBands(fireColyear.select("BurnDate"))    
    
    image = image.set("system:time_start",start)
    image = image.set("system:time_end", end)
    image = image.set('system:index',str(yr))
    
    return image.set('year', yr)

def prismAnnualMean(y):
    prism_ = prism.filter(ee.Filter.calendarRange(y, y, 'year')).mean().set('year', y).select("vpdmax")
    start =ee.Date.fromYMD(year = y, month = 1, day = 1)
    end =ee.Date.fromYMD(year = y, month = 12, day = 31)
    prism_ = prism_.set("system:time_start", start)
    prism_ = prism_.set("system:time_end", end)
    
    return prism_

prism = ee.ImageCollection(years.map(prismAnnualMean))
prism = prism.map(addAnnualBurnedArea)

def convert_to_float(image):
    return image.toFloat()


prism = prism.map(convert_to_float)


print(prism.size().getInfo())

n = prism.size().getInfo() # number of images to download
    
colList = prism.toList(n)

folder_name = "ba_vpd"  
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


