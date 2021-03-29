# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:33:04 2020

@author: kkrao
"""


import pandas as pd
from init import *
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import plot_roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm




sns.set(style='ticks',font_scale = 0.9)
df = pd.read_csv(os.path.join(dir_data, "fire_collection_median_with_climate_500m_variogram.csv"))

dfr = pd.read_csv(os.path.join(dir_data, "fire_collection_median_extra_lfmc_vars_500m_variogram.csv"))
dfr = dfr[['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside', 'lfmc_t_2_inside', 'lfmc_t_2_outside']]
df = df.join(dfr)

dfr = pd.read_csv(os.path.join(dir_data, "fire_collection_median_fwi_500m_variogram.csv"))
dfr = dfr[['fwi_t_4_inside','fwi_t_4_outside']]
df = df.join(dfr)


df = df.loc[df.landcover.isin(lc_dict.keys())]
df['landcover'] = df.landcover.map(lc_dict)

print(df.shape)


#%% just lfmc first 

size_dict = {'small':(df.area<=1),
             'large':(df.area>1)}

fig, ax = plt.subplots(figsize = (3,3))
df.area.hist(bins = 400, ax = ax)
ax.set_xlabel('Fire size (km$^2$)')
ax.set_ylabel('Frequency')
ax.set_xlim(0,10)

bar = pd.Series([df.loc[size_dict['small']].shape[0],
df.loc[size_dict['large']].shape[0]],index = ["small","large"])

fig, ax = plt.subplots(figsize = (3,3))
bar.plot.bar(ax = ax)
ax.set_ylabel('Frequency')

