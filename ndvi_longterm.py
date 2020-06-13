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


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


sns.set(style='ticks',font_scale = 0.9)
df = pd.read_csv(os.path.join(dir_root, "data","longterm","NDVI_seasonality_longterm_and_2016_2019.csv"))
df = df.loc[df.landcover.isin(lc_dict.keys())]
df['landcover'] = df.landcover.map(lc_dict)

df.shape
df.head()
df[['NDVIMeanH','NDVIMeanR','NDVISdH','NDVISdR' ]]/=10000
df.drop('.geo',axis = 1, inplace = True)
df['moy'] = [int(x.split('_')[0])+1 for x in df['system:index']]
df['feature'] = [int(x.split('_')[1]) for x in df['system:index']]

fig, ax = plt.subplots(figsize = (3,3))
df.groupby('landcover').moy.count().plot.bar(ax = ax)
ax.set_ylabel('Frequency')

def NRMSE(sub):
    rmse = np.sqrt(mean_squared_error(sub['NDVIMeanR'],sub['NDVIMeanH']))
    nrmse = rmse/sub['NDVIMeanH'].mean()
    return rmse

#%% plot seasonalities

for feature in [10,32,802,1000,2000]:
    # lc = s
    sub = df.loc[df.feature==feature]
    
    fig, ax = plt.subplots(figsize = (3,3))
    ax.plot(sub['moy'],sub['NDVIMeanH'],color = 'k',label = 'Historic')
    ax.fill_between(sub['moy'], sub['NDVIMeanH'] - sub['NDVISdH'], sub['NDVIMeanH'] + sub['NDVISdH'],alpha = 0.1, color = 'k')
    
    color = color_dict[sub['landcover'].values[0]]
    ax.plot(sub['moy'],sub['NDVIMeanR'],color = color,label = '2016-2019')
    ax.fill_between(sub['moy'], sub['NDVIMeanR'] - sub['NDVISdR'], sub['NDVIMeanR'] + sub['NDVISdR'],alpha = 0.2, color = color)
        
    ax.set_xticks([1,4,7,10])
    ax.set_xticklabels(['Jan','Apr','Jul','Oct'])
    ax.set_ylabel('NDVI')
    error = NRMSE(sub)
    ax.annotate('NRMSE=%0.3f'%error, \
                    xy=(0.1, 0.95), xycoords='axes fraction',\
                    ha='left',va='top')

    ax.legend(loc = 'lower right')
    
#%% is distribution of RMSEs for(2016 - 2019, history) similar to other periods?

for lc in df.landcover.unique():
    fig, ax = plt.subplots(figsize = (3,3))
    ax.set_title(lc)
    nrmse = pd.DataFrame()
    ctr = 0
    colors = ['C0','C1','C2','C3']
    for startyear in [2004,2008,2012,2016]:
        endyear = startyear+3
        
        df = pd.read_csv(os.path.join(dir_root, "data","longterm","NDVI_seasonality_longterm_and_%04d_%04d.csv"%(startyear, endyear)))
        df = df.loc[df.landcover.isin(lc_dict.keys())]
        df['landcover'] = df.landcover.map(lc_dict)
        df = df.loc[df.landcover==lc]
        
        df[['NDVIMeanH','NDVIMeanR','NDVISdH','NDVISdR' ]]/=10000
        df.drop('.geo',axis = 1, inplace = True)
        df['moy'] = [int(x.split('_')[0])+1 for x in df['system:index']]
        df['feature'] = [int(x.split('_')[1]) for x in df['system:index']]
        
        nrmse_ = df.groupby('feature').apply(NRMSE)
        label = '%s - %s'%(startyear,endyear)
        nrmse[label] = nrmse_
        nrmse_.plot.kde(ax = ax,label = label,color = colors[ctr])
        
        x = ax.lines[ctr*2].get_xdata() # Get the x data of the distribution
        y = ax.lines[ctr*2].get_ydata() # Get the y data of the distribution
        
        mean = nrmse_.mean()
        choose = np.argmax(x>mean)
    
        
        maxid = np.argmax(y) # The id of the peak (maximum of y data)
        ax.axvline(mean, ymax = y[choose]/50,linewidth = 2, alpha = 0.5,color = colors[ctr])
        ctr+=1
        
        mean_confidence_interval(nrmse_)
        
    ax.set_xlabel('NRMSE(Historic, Period)')
    ax.set_ylabel('Density')
    ax.set_xlim(0,0.2)
    ax.set_ylim(0,50)
    ax.legend()
    plt.show()

nrmse.head()
nrmse.mean()


mannwhitneyu(nrmse['2016 - 2019'], nrmse['2008 - 2011'], alternative = 'greater')

mannwhitneyu(np.random.normal(0,size = 100), np.random.normal(10,size = 100), alternative = 'less')

ttest_ind(np.random.normal(0,size = 100), np.random.normal(10,size = 100))
ttest_ind(nrmse['2016 - 2019'], nrmse['2008 - 2011'])
