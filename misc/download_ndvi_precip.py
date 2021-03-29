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
prism = ee.ImageCollection("OREGONSTATE/PRISM/AN81m")
# prism.first().getInfo()

def addMonthYear(image):
  month = ee.Date(image.get("system:time_start")).get("month")
  year = ee.Date(image.get("system:time_start")).get("year")
  image = image.set("month", month)
  image = image.set("year", year)
  
  return image

prism = prism.map(addMonthYear)
years = ee.List.sequence(2001,2019);

def prismAnnualMean(y):
    # print(y.getInfo())
    yr = ee.Number(y)
    filter_ = ee.Filter.Or(ee.Filter.And(ee.Filter.eq('year',y),ee.Filter.gte('month',10)),
            ee.Filter.And(ee.Filter.eq('year',yr.add(1)),ee.Filter.lte('month',3))) #// Mar - Dec of current year, and Jan - Feb of next year
    prism_ = prism.filter(filter_).sum().select('ppt');

    # prism_ =prism.filterDate(start,end)
    start =ee.Date.fromYMD(year = y, month = 1, day = 1)
    end =ee.Date.fromYMD(year = y, month = 12, day = 31)
    prism_ = prism_.set("system:time_start", start)
    prism_ = prism_.set("system:time_end", end)
    
    return prism_

prism = ee.ImageCollection(years.map(prismAnnualMean))

# print(prismAnnualMean(ee.Number(2002)).getInfo())


def convert_to_float(image):
    return image.toFloat()

# prism = prism.map(convert_to_float)

print(prism.size().getInfo())
print(prism.getInfo())

n = prism.size().getInfo() # number of images to download
    
colList = prism.toList(n)

folder_name = "precip_annual_sep_mar"  
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


