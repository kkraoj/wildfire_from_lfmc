# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:21:37 2021

@author: kkrao
"""

import os

from init import dir_root, dir_data

import numpy as np
from osgeo import gdal

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 1., style = "ticks")
plt.style.use("pnas")





path = os.path.join(dir_root, \
                    "data/arr_pixels_lfmc_vpd_anomalies",\
                        "lfmc_vpd_lag_6_both_norm_positive_coefSum.tif")
ds = gdal.Open(path)
pwsVPD = np.array(ds.GetRasterBand(1).ReadAsArray())

path = os.path.join(dir_root, "data","arr_pixels_lfmc_dfmc_anomalies","PAS_6_jan_2021.tif")
ds = gdal.Open(path)
pwsDFMC = np.array(ds.GetRasterBand(1).ReadAsArray())

r2 = pd.DataFrame({"vpd": pwsVPD.flatten(), "dfmc":pwsDFMC.flatten()}).corr().iloc[0,1]**2                  


fig, ax = plt.subplots(figsize = (3,3))

ax.scatter(pwsDFMC, pwsVPD, s = 2, color = "k",alpha = 0.01)
ax.set_xlabel("PWS$_{DMFC}$")
ax.set_ylabel("PWS$_{VPD}$")
ax.set_xlim(0,2)
ax.set_ylim(0,2)

ax.annotate(f'R$^2$={r2:0.2f}', 
            xy=(0.9, 0.9), ha = "right",textcoords='axes fraction')
            
plt.show()