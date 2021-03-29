# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:43:52 2020

@author: kkrao
"""

from init import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



sns.set(style='ticks',font_scale = 1.1)
df = pd.read_csv(os.path.join(dir_data, "fire_collection_median_with_climate_500m_variogram.csv"))
df = df.loc[df.landcover.isin(lc_dict.keys())]
df['landcover'] = df.landcover.map(lc_dict)
filters = (df.BurnDate>=160)&(df.area<10)&(df.area>1)

df = df.loc[filters]
#%%
# fig, ax = plt.subplots(figsize = (3,3))
# ax.scatter(df['lfmc_t_1_inside'],df['area'])
# plt.yscale('log')

#%%
for lc in df.landcover.unique():
    fig, ax = plt.subplots(figsize = (3,3))
    ax.set_title(lc)
    df.loc[df.landcover==lc,'lfmc_t_1_inside'].hist(ax = ax, color = color_dict[lc])
    ax.set_xlim(0,300)
    ax.set_ylabel('Number of fires')
    ax.set_xlabel('Pre-fire LFMC (%)') 
    
    ax.axvline(lfmc_thresholds[lc][0],color = 'darkred',linewidth = 2, label = 'extreme danger')
    ax.axvline(lfmc_thresholds[lc][1],color = 'darkorange',linewidth = 2, label = 'high danger')
    plt.legend()