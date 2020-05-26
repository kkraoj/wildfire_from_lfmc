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




sns.set(style='ticks',font_scale = 0.9)
df = pd.read_csv(os.path.join(dir_data, "fire_collection_median_with_climate_500m_variogram.csv"))
df = df.loc[df.landcover.isin(lc_dict.keys())]
df['landcover'] = df.landcover.map(lc_dict)

dfr = pd.read_csv(os.path.join(dir_data, "fire_collection_median_extra_lfmc_vars_500m_variogram.csv"))
dfr = dfr.loc[dfr.landcover.isin(lc_dict.keys())]
dfr['landcover'] = dfr.landcover.map(lc_dict) 

print(df.shape)
dfr.shape
dfr = dfr[['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside', 'lfmc_t_2_inside', 'lfmc_t_2_outside']]

df = df.join(dfr)
print(df.shape)


#%% just lfmc first 

size_dict = {'small':(df.area<=1),
             'medium':(df.area>1)&(df.area<=10),
             'large':(df.area>10)}
master = pd.DataFrame(index = sorted(df.landcover.unique()),columns = size_dict.keys())

cols = [col for col in df.columns if 'lfmc' in col]
df = df[cols]
df['lfmc_t_1_inside_anomaly'] = df['lfmc_t_1_inside']

for fire_size in size_dict.keys():
    dfcat = df.loc[size_dict[fire_size]].copy()
    fig, ax = plt.subplots(figsize = (3,3))
    for lc in sorted(dfcat.landcover.unique()):
        sub = dfcat.loc[dfcat.landcover==lc]
        ndf = pd.DataFrame()
        
        for var in ['outside','inside']:    
            cols = [col for col in sub.columns if var in col]
            # cols.remove('lfmc_t_1_%s'%var)
            data = sub[cols].copy()
            new_cols = [col.split('_')[0] for col in data.columns]
            data.columns = (new_cols)
            data['fire'] = int(var=='inside')
            ndf = pd.concat([ndf, data], axis = 0).reset_index(drop=True)
            
        
        # lfmc_df = pd.get_dummies(ndf['lfmc'], prefix='lfmc')
        # ndf = ndf.drop('lfmc',axis = 1)
        # ndf = ndf.join(lfmc_df)
        ndf = ndf.sample(frac=1).reset_index(drop=True)
        ndf.dropna(inplace = True)
        X = ndf.drop('fire', axis = 1)
        y = ndf['fire']
        
        clf = RandomForestClassifier(max_depth=5, random_state=0)
        clf.fit(X, y)
        
        rfc_disp = plot_roc_curve(clf, X, y, ax=ax,label = lc,color = color_dict[lc])
        # print('%s:\t %0.2f'%(lc,roc_auc_score(y, clf.predict(X))))
        # print('%0.2f'%(roc_auc_score(y, clf.predict(X))))
        master.loc[lc,fire_size] = roc_auc_score(y, clf.predict(X))

print(master)
      # clf.fe