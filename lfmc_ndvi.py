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
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import plot_roc_curve, roc_auc_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import scipy.stats
import statsmodels.stats.api as sms

def unstack(series):
    return pd.DataFrame(data = series.values.reshape(4,12), index = [2016,2017,2019,2019],columns = range(1,13)).mean()

sns.set(style='ticks',font_scale = 0.9)
df = pd.read_csv(os.path.join(dir_root, "data","longterm","lfmc_ndvi_time_series_at_fire_points.csv"))
df = df.loc[df.landcover.isin(lc_dict.keys())]
df['landcover'] = df.landcover.map(lc_dict)

df.shape
df.head()
df.columns
df.rename(columns = {'LFMC':'LFMC_0','NDVI':'NDVI_0'}, inplace = True)

for lc in df.landcover.unique():
    dfcat = df.loc[df.landcover==lc]
    R = []
    for point in dfcat['system:index']:
        sub = dfcat.loc[df['system:index']==point]
        lfmc = sub[[col for col in sub.columns if 'LFMC' in col]]
        ndvi = sub[[col for col in sub.columns if 'NDVI' in col]]
        
        lfmc = unstack(lfmc)
        ndvi = unstack(ndvi)
        R.append(lfmc.corr(ndvi))
        # np.corrcoef(lfmc,ndvi)[0][1]
    fig, ax = plt.subplots(figsize = (3,3))
    ax.hist(R)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('$R(\overline{LFMC}, \overline{NDVI})$')
    ax.set_title(lc)
    R= np.array(R)
    print(lc)
    print('Mean correlation = %0.2f'%np.nanmean(R))
    print('Correlation > 0.5 in %0.2f cases'%(R>0.5).mean())
    
