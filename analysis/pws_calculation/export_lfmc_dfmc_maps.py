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


## Initialize (a ee python thing)

ee.Initialize()

####################################################

roi = ee.FeatureCollection("users/kkraoj/west_usa")
lfmc =  ee.ImageCollection("users/kkraoj/lfm-mapper/lfmc_col_24_jul_2020")
gridmet =  ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')

#rename b1 band to lfmc
lfmc = lfmc.select(['b1'],['lfmc'])

# Make an image who pixel value is number of times that pixel has a valid, 
# non-masked LFMC data.
nImages = lfmc.count().select("lfmc")

# if 50% of LFMC data is null, mask that pixel
threshCount = int(0.5*(lfmc.size().getInfo()))

def countMask(image):
  """Mask pixel if a pixel has more than threshCount number of null values"""
  image = image.updateMask(nImages.gte(threshCount))
  return image

months = ee.List.sequence(1,12);
def lfmcSeasonalCycle(m):
    return lfmc.filter(ee.Filter.calendarRange(m, m, 'month')).mean().set('month', m)

def gridmetSeasonalCycle(m):
    return gridmet.filter(ee.Filter.calendarRange(m, m, 'month')).mean().set('month', m)


def calcLfmcAnom(image):
  start = ee.Date(image.get("system:time_start"))
  end = ee.Date(image.get("system:time_end"))
  month = start.get('month')
  image = image.subtract(lfmcMean.filter(ee.Filter.eq('month',month)).mean()) #.mean(0 is required to convert imagecollection of 1 image to image)
  image = image.set("system:time_start",start)
  image = image.set("system:time_end", end)
  
  return image

def addGridmetAnom(image):
  
  start = ee.Date(image.get("system:time_start"))
  month = start.get('month')
  # startAdvanced = start.advance(-6,"month")
  end = ee.Date(image.get("system:time_end"))
  gridmet_ = gridmet.filterDate(start, end).mean()
  
  gridmetAnom = gridmet_.subtract(gridmetMean.filter(ee.Filter.eq('month',month)).mean())
  
  # image = image.multiply(ee.Image.constant(-1))
  
  image = image.addBands(gridmetAnom.select("fm100"))
  image = image.addBands(gridmetAnom.select("fm1000"))
  image = image.set("system:time_start",start)
  image = image.set("system:time_end", end)
  return image

def convert_to_float(image):
    """Cast each image to float."""
    return image.toFloat()

lfmcMean = ee.ImageCollection.fromImages(\
                months.map(lfmcSeasonalCycle))
    
gridmetMean = ee.ImageCollection.fromImages(\
                months.map(gridmetSeasonalCycle))
    
lfmcAnom = lfmc.map(calcLfmcAnom) \
                .map(addGridmetAnom) \
                .map(countMask) \
                .map(convert_to_float)

n = lfmcAnom.size().getInfo() # number of images to download


colList = lfmcAnom.toList(n)

folder_name = "lfmc_dfmc_anomalies"
print(f"Exporting {n} images to your Google driver folder called {folder_name}.")

scale = 4000 #resolution in meters

#export image one by one.
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

print("Jobs sent GEE server. Open https://code.earthengine.google.com/tasks for status.")


