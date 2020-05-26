# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:43:52 2020

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

filters = (df.BurnDate<=160)
# (df['lfmc_t_1_inside']<120)

df = df.loc[filters]

def classify_lfmc(row):
    # print(row)
    dangers = ['extreme','high','moderate','low']
    for col in ['lfmc_t_1_inside','lfmc_t_1_outside']:
        # print(row[col])
        bucket = (row[col] >= np.array(lfmc_thresholds[row['landcover']])).sum()
        row['%s_class'%col] = dangers[bucket]
    return row
#%%

# df['lfmc_t_1_inside_class'] = 'extreme'
# df['lfmc_t_1_outside_class'] = 'extreme'

# df = df.apply(classify_lfmc,axis = 1)

dangers = ['extreme','high','moderate','low']
# for lc in sorted(df.landcover.unique()):
#     sub = df.loc[df.landcover==lc]
#     fig, ax = plt.subplots(figsize = (3,3))
#     sns.boxplot('lfmc_t_1_inside_class', 'area',data = sub,ax = ax,order = dangers)
#     plt.yscale("log")
#     # df['lfmc_t_1_inside_class'].hist(ax = ax,color = 'darkred',alpha)
#     # sns.barplot('lfmc_t_1_inside_class','area',ax = ax)
#     # sub['lfmc_t_1_inside_class'].plot.hist(color = 'darkred',alpha = 0.5,ax=ax,label = 'Burned area',linewidth = 0)
#     # sub['lfmc_t_1_outside'].plot.hist(color = 'lime',alpha = 0.5,ax=ax,label = 'Unaffected area',linewidth = 0)
#     # ax.axvline(sub['lfmc_t_1_inside'].mean(),color = 'darkred',linewidth = 2, label = '_nolegend_')
#     # ax.axvline(sub['lfmc_t_1_outside'].mean(),color = 'darkgreen',linewidth = 2, label = '_nolegend_')
#     ax.set_xlabel('LFMC class')
#     ax.set_title('%s'%lc)
#     inside = np.array([dangers.index(val) for val in sub['lfmc_t_1_inside_class']])
#     outside = np.array([dangers.index(val) for val in sub['lfmc_t_1_outside_class']])
#     # print(lc)
#     # print((inside==0).mean())
#     change =  inside- outside
#     total = ((change<0)|(outside==0)).mean()
#     # print('%s:\t %0d%%'%(lc, total*100))
#     print('%0.2f'%total)
    # print(lc)
    # plt.legend()
    
#%% logit

size_dict = {'small':(df.area<=1),
             'medium':(df.area>1)&(df.area<=10),
             'large':(df.area>10)}
master = pd.DataFrame(index = sorted(df.landcover.unique()),columns = size_dict.keys())


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
      # clf.feature_importances_.sum()
# ax.get_legend().remove()
# >>> print(clf.predict([[0, 0, 0, 0]]))


