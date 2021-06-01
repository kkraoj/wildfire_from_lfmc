# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:41:53 2020

@author: kkrao
"""


import os
from init import dir_root, dir_data, dir_fig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import gdal
import matplotlib as mpl
import seaborn as sns
import scipy
from plotmap import plotmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.filters import gaussian_filter
import matplotlib.patches as patches

import fiona
import rasterio
import rasterio.mask


# with fiona.open("D:/Krishna/projects/vod_from_mortality/codes/data/Mort_Data/CA/CA_shapefile/CA_shapefile.shp", "r") as shapefile:
#     shapes = [feature["geometry"] for feature in shapefile]
    
# with rasterio.open(plantClimatePath) as src:
#     out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
#     out_meta = src.meta
    
# out_meta.update({"driver": "GTiff",
#                  "height": out_image.shape[1],
#                  "width": out_image.shape[2],
#                  "transform": out_transform})

# with rasterio.open(os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","PAS_6_jan_2021_CA.tif"), "w", **out_meta) as dest:
#     dest.write(out_image)
    
    

SAVEPLOT = False
sns.set(font_scale = 1., style = "ticks")
plt.style.use("pnas")


plantClimatePath = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","PAS_6_jan_2021_CA.tif")
ds = gdal.Open(plantClimatePath)
plantClimate = np.array(ds.GetRasterBand(1).ReadAsArray())
plantClimate[plantClimate==0] = np.nan

#%% plant climate map
fig, ax = plt.subplots(figsize = (3,3),frameon = False)
scatter_kwargs = dict(cmap = "PiYG_r",vmin = 0, vmax = 2,alpha = 1)
ax.axis("off")
gt = ds.GetGeoTransform()
# plantClimate[plantClimate<1.9] = np.nan
map_kwargs = dict(llcrnrlon=-125,llcrnrlat=32,urcrnrlon=-114,urcrnrlat=42.5,
        projection='cyl')
fig, _, m, plot = plotmap(gt = gt, var = plantClimate,map_kwargs=map_kwargs ,scatter_kwargs=scatter_kwargs, marker_factor = 1, 
                      fill = "white",background="white",fig = fig,ax=ax,
                      shapefilepath = 'D:/Krishna/projects/wildfire_from_lfmc/data/CA_Counties_TIGER2016', 
                  shapefilename ='CA_Counties_TIGER2016')
cax = fig.add_axes([0.7, 0.52, 0.03, 0.3])
    
# cax.annotate('Plant  (%) \n', xy = (0.,0.94), ha = 'left', va = 'bottom', color = "w")
cb0 = fig.colorbar(plot,ax=ax,cax=cax,ticks = np.linspace(0,2,5))
# plt.tight_layout()
ax.axis("off")
plt.savefig(os.path.join(dir_fig, "PWS_CA.jpg"), dpi=300, facecolor='w', edgecolor='w',
        frameon = False)
plt.show()
