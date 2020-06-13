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




sns.set(style='ticks',font_scale = 1.5)
df = pd.read_csv(os.path.join(dir_data, "fire_collection_median_with_climate_500m_variogram.csv"))
df = df.loc[df.landcover.isin(lc_dict.keys())]
df['landcover'] = df.landcover.map(lc_dict)

size_dict = {'small':(df.area<=1),
             'medium':(df.area>1)&(df.area<=10),
             'large':(df.area>10)}

dfr = pd.read_csv(os.path.join(dir_data, "fire_collection_median_extra_lfmc_vars_500m_variogram.csv"))
dfr = dfr.loc[dfr.landcover.isin(lc_dict.keys())]
dfr['landcover'] = dfr.landcover.map(lc_dict) 

# print(df.shape)
dfr.shape
dfr = dfr[['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside', 'lfmc_t_2_inside', 'lfmc_t_2_outside']]

df = df.join(dfr)

filters = (df.BurnDate>=180)
# (df['lfmc_t_1_inside']<120)

df = df.loc[filters]
# print(df.shape)


#%% LFMC + climate

cols = [col for col in df.columns if 'lfmc' in col]+['landcover','area']
cols+=[col for col in df.columns if 'erc' in col]
cols+=[col for col in df.columns if 'ppt' in col]
cols+=[col for col in df.columns if 'vpd' in col]


df = df[cols]
df['lfmc_t_1_inside_anomaly'] = df['lfmc_t_1_inside'] - df['lfmc_t_1_seasonal_mean_inside']
df['lfmc_t_1_outside_anomaly'] = df['lfmc_t_1_outside'] - df['lfmc_t_1_seasonal_mean_outside']

df.drop(['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside'],axis = 1, inplace = True)

df.columns
df = df.rename(columns = {'lfmc_t_1_inside_anomaly':'lfmct1anomaly_inside',
                          'lfmc_t_1_outside_anomaly':'lfmct1anomaly_outside',
                          'lfmc_t_2_inside':'lfmct2_inside',
                          'lfmc_t_2_outside':'lfmct2_outside',
                          'lfmc_t_1_inside':'lfmct1_inside',
                          'lfmc_t_1_outside':'lfmct1_outside',
                          })



# for fire_size in size_dict.keys():
#     dfcat = df.loc[size_dict[fire_size]].copy()
#%% plot for fire pixels only    
sns.set(font_scale = 1.0, style = 'ticks')
for lc in sorted(df.landcover.unique()):
    
    
    fig = plt.figure(constrained_layout=True, figsize = (4,3.7))
    gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48, bottom = 0.1,wspace=0.5,hspace = 0.5)
    ax = fig.add_subplot(gs1[1:, :2])
    ax1 = fig.add_subplot(gs1[0, :2])
    ax2 = fig.add_subplot(gs1[1:, 2:])


    # fig, ax = plt.subplots(figsize = (3,3))
    sub = df.loc[df.landcover==lc]
    ndf = pd.DataFrame()
    
    for var in ['inside']:    
        cols = [col for col in sub.columns if var in col]
        # cols.remove('lfmc_t_1_%s'%var)
        data = sub[cols].copy()
        new_cols = [col.split('_')[0] for col in data.columns]
        data.columns = (new_cols)
        data['fire'] = int(var=='inside')
        if var=='inside':
            data = data.join(sub['area'])
        else:
            data['area'] = 0
        ndf = pd.concat([ndf, data], axis = 0).reset_index(drop=True)
    ndf['color'] = 'grey'
    ndf.loc[ndf.fire==1,'color'] = color_dict[lc]
    ndf.sort_values('area',inplace = True,ascending = False)
    ax.scatter(ndf['lfmct1'],ndf['vpd'],color = ndf['color'],s = np.log(ndf['area'])*30,edgecolor = 'lightgrey',linewidth = 0.5)
    ax.set_xlabel('LFMC (%)')
    ax.set_ylabel('VPD (hPa)')
    # ax.set_xlim(0,250)
    # ax.set_ylim(0,80)
    # ax.set_title(lc)
    
    
    
    # sns.set(font_scale = 1.0, style = 'ticks')
    ndf['bin'] = ndf.lfmct1 - ndf.lfmct1 % 40 + 20
    s = ndf[['bin', 'area']].groupby('bin').sum()
    s.index = s.index.astype(int)
    # fig, ax = plt.subplots(figsize = (3,1))
    s.plot(kind = 'bar', ax = ax1,legend = False, color = color_dict[lc])
    # ax.hist(ndf['lfmct1'], weights = ndf['area'], color = color_dict[lc])
    ax1.set_xlabel('')
    ax1.set_ylabel('BA (km$^2$)')
    ax1.set_title(lc)
    # ax1.set_xlim(0,250)
    ax.set_xlim(s.index.min()-20,s.index.max()+20)
    ax.set_xticks(s.index)
    
    ndf['bin'] = ndf.vpd - ndf.vpd % 10 + 5
    s = ndf[['bin', 'area']].groupby('bin').sum()
    s.index = s.index.astype(int)
    # fig, ax = plt.subplots(figsize = (1,3))
    s.plot(kind = 'barh', ax = ax2,legend = False, color = color_dict[lc], )
    # ax.hist(ndf['lfmct1'], weights = ndf['area'], color = color_dict[lc])
    # ax.set_xlabel(' (%)')
    ax2.set_xlabel('BA (km$^2$)')
    ax2.set_ylabel("")
    
    ax.set_ylim(s.index.min()-5,s.index.max()+5)
    ax.set_yticks(s.index)
    # ax.set_ylim(*ax2.get_ylim())
    # ax2.set_ylim(0,80)

    # break

#%% plot for fire pixels and nonf ire  
# for fire_size in size_dict.keys():
#     dfcat = df.loc[size_dict[fire_size]].copy()
#     for lc in sorted(dfcat.landcover.unique()):
#         # fig, ax = plt.subplots(figsize = (3,3))
#         sub = dfcat.loc[dfcat.landcover==lc]
#         ndf = pd.DataFrame()
        
#         for var in ['inside']:    
#             cols = [col for col in sub.columns if var in col]
#             # cols.remove('lfmc_t_1_%s'%var)
#             data = sub[cols].copy()
#             new_cols = [col.split('_')[0] for col in data.columns]
#             data.columns = (new_cols)
#             data['fire'] = int(var=='inside')
#             if var=='inside':
#                 data = data.join(sub['area'])
#             else:
#                 data['area'] = 0
#             ndf = pd.concat([ndf, data], axis = 0).reset_index(drop=True)
#         ndf['color'] = 'grey'
#         ndf.loc[ndf.fire==1,'color'] = 'orange'
#         ndf.sort_values('area',inplace = True,ascending = True)
#         ax = sns.jointplot(ndf['lfmct1'],ndf['vpd'],kind= 'kde')
    
#         # sns.kdeplot(ndf['lfmct1'],ndf['vpd'], ax=ax)
#         # sns.rugplot(ndf['lfmct1'], color="g", ax=ax)
#         # sns.rugplot(ndf['vpd'], vertical=True, ax=ax);
#         # ax.scatter(ndf['lfmct1'],ndf['vpd'],color = ndf['color'],s = 0.5)
#         # print(lc)
#         ax.set_axis_labels(xlabel = 'LFMC (%)' , ylabel = 'VPD (hPa)')
        
#         # ax.set_xlabel('LFMC (%)')
#         # ax.set_ylabel('VPD (hPa)')
#         text = '%s fires in %s'%(fire_size,lc)
#         ax.ax_joint.annotate(text, \
#                     xy=(0.5, 0.95), xycoords='axes fraction',\
#                     ha='center',va='top')
#         plt.show()
