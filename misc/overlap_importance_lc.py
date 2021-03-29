# -*- coding: utf-8 -*-
"""
scatter plots of LFMC vs vpd, erc, ppt to check if they are weakly correlated or not (they should be)
@author: kkrao
"""
import os
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from osgeo import gdal, osr, gdal_array

dir_data = "D:\Krishna\projects\wildfire_from_lfmc\data"

#%% Plot control settings
ZOOM=1
FS=12*ZOOM
PPT = 0
DPI = 300
sns.set_style('ticks')
#%% fix plot dims
mpl.rcParams['font.size'] = FS
mpl.rcParams['axes.titlesize'] = 'medium'
SC = 3.54331*ZOOM
DC = 7.48031*ZOOM

#%% initialize plot
lc_dict = {14: 'crop',
            20: 'crop',
            30: 'crop',
                50: 'Closed broadleaf\ndeciduous',
            70: 'Closed needleleaf\nevergreen',
            90: 'Mixed forest',
            100:'Mixed forest',
            110:'Shrub/grassland',
            120:'grassland/shrubland',
            130:'Shrubland',
            140:'Grassland',
            150:'sparse vegetation',
            160:'regularly flooded forest'}
color_dict = {'Closed broadleaf\ndeciduous':'darkorange',
              'Closed needleleaf\nevergreen': 'forestgreen',
              'Mixed forest':'darkslategrey',
              'Shrub/grassland' :'y' ,
              'Shrubland':'tan',
              'Grassland':'lime',
              }  
##### 50,70,[90,100], [110, 120],130,140
SEED = 1
np.random.seed(SEED)
#%% functions
ind = {'lfmc':0,'vpd':1,'ppt':2,'erc':3,'fwi':4,'lc':5}
units= {'lfmc':'%','vpd':'hPa','ppt':'mm','erc':'-','fwi':'-','lc':'-'}
  
def get_average_trait(trait = "p50"):
    arr = gdal_array.LoadFile(os.path.join(dir_data,'mean/lfmc_vpd_ppt_erc_fwi_lc.tif'))    
    ## subsetting for law VPD locations only
#    arr[0,arr[0,:,:]>25] = np.nan
    x = arr[ind['lc'],:,:]
#     y = arr[ind[var],:,:].flatten()
#     xarr = arr[ind['lfmc'],:,:].flatten()
#     yarr = arr[ind[var],:,:].flatten()
#     lcarr = arr[ind['lc'],:,:].flatten()
#     sampling_ratio = 100

#     fig, axs = plt.subplots(2,3,figsize = (DC,0.67*DC),sharex = True, sharey= True)
    
#     for lc, ax in zip([50,70,90,110,130,140],axs.reshape(-1)):
#         x = xarr[lcarr==lc]
#         y = yarr[lcarr==lc]
#         # cmap = mpl.colors.LinearSegmentedColormap.\
#         #           from_list("", ["w",color_dict[lc_dict[lc]]])
#         cmap = sns.cubehelix_palette(rot = -0.4,as_cmap = True)
#         x,y = sample(x,y,sampling_ratio = sampling_ratio)
#         sns.kdeplot(x,y,cmap = cmap, shade = True, legend = False, ax = ax, shade_lowest = False)
#         # plot_pred_actual(x,y,
#         #               xlabel = "Mean LFMC (%)", ylabel = "Mean VPD (hPa)",\
#         #               ax = ax,annotation = False,\
#         #               oneone=False,\
#         #               cmap = ListedColormap(sns.cubehelix_palette(rot = -0.4).as_hex()))
#         non_nan_ind=np.where(~np.isnan(x))[0]
#         x=x.take(non_nan_ind);y=y.take(non_nan_ind)
#         slope, intercept, r_value, p_value, std_err =\
#             stats.linregress(x,y)
#         xs = np.linspace(50,150)
#         ys = slope*xs+intercept
#         ax.plot(xs,ys,color = 'k', lw = 1)
        
#         ax.set_xlim(50,150)
# #        ax.set_ylim(0,5)
#         # ax.set_aspect('auto')
#         ax.set_xticks([50,100,150])
# #        ax.set_yticks([0,2.5,5])
#         # ax.set_xlabel('Mean LFMC (%)')
#         # ax.set_ylabel('Mean VPD (hPa)')
#         # ax.invert_xaxis()
#         # print('p value = %0.3f'%p_value)
        
#         ax.annotate('$R$ = %0.2f'%r_value, \
#                     xy=(0.95, 0.95), xycoords='axes fraction',\
#                     ha='right',va='top')
#         ax.set_title(lc_dict[lc])
    return x

    
    
def main():
    x = get_average_trait()
if __name__ == '__main__':
    main()