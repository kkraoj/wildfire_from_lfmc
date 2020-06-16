# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:33:04 2020

@author: kkrao
"""


import pandas as pd
from init import dir_data, lc_dict, color_dict
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

def assemble_df():
    df = pd.read_csv(os.path.join(dir_data, "fire_collection_median_with_climate_500m_variogram.csv"))
    
    dfr = pd.read_csv(os.path.join(dir_data, "fire_collection_median_extra_lfmc_vars_500m_variogram.csv"))
    dfr = dfr[['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside', 'lfmc_t_2_inside', 'lfmc_t_2_outside']]
    df = df.join(dfr)
    
    dfr = pd.read_csv(os.path.join(dir_data, "fire_collection_median_fwi_500m_variogram.csv"))
    dfr = dfr[['fwi_t_4_inside','fwi_t_4_outside']]
    df = df.join(dfr)
    
    df = df.loc[df.landcover.isin(lc_dict.keys())]
    df['landcover'] = df.landcover.map(lc_dict)
    return df

df = assemble_df()


#%% just lfmc first 

SIZE_DICT = {'small':(df.area<=4),
             'large':(df.area>4)}


def calc_auc_occurence(dfsub, size_dict, clf):
    df = dfsub.copy()
    auc = pd.DataFrame(index = sorted(df.landcover.unique()),columns = size_dict.keys())
    for fire_size in size_dict.keys():
        dfcat = df.loc[size_dict[fire_size]].copy()
        # fig, ax = plt.subplots(figsize = (3,3))
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
                
    
            ndf = ndf.sample(frac=1).reset_index(drop=True)
            ndf.dropna(inplace = True)
    
            X = ndf.drop('fire', axis = 1)
            y = ndf['fire']
            
            try:
                clf.fit(X, y)
                # rfc_disp = plot_roc_curve(clf, X, y, ax=ax,label = lc,color = color_dict[lc])
                auc.loc[lc,fire_size] = roc_auc_score(y, clf.predict(X))
            except: 
                print("Could not fit RF for combo of fire size: %s,\tland cover: %s"%(fire_size, lc))
                
    return auc

def ensemble_auc(dfsub, size_dict, clf, iters = 100, label = 'All variables'):
    clf.random_state = 0
    dummy = calc_auc_occurence(dfsub, size_dict, clf)
    aucs = np.expand_dims(dummy.values, axis = 2)
    for itr in range(1, iters):
        clf.random_state = itr
        auc = np.expand_dims(calc_auc_occurence(dfsub, size_dict, clf).values, axis = 2)
        
        aucs = np.append(aucs,auc, axis = 2)
    # print("aucs ready")
    dummy.loc[:,:] = np.nanmean(aucs.astype(float), axis = 2)
    mean = dummy.copy()
    dummy.loc[:,:] = np.nanstd(aucs.astype(float), axis = 2)
    sd = dummy.copy()
    
    return mean, sd
    

def calc_auc_diff(dfs, size_dict):
    df = dfs.copy()
    allVars = pd.DataFrame(index = sorted(df.landcover.unique()),columns = size_dict.keys())
    onlyClimate = allVars.copy()
    cols = [col for col in df.columns if 'lfmc' in col]+['landcover']
    # cols = ['landcover']
    cols+=[col for col in df.columns if 'erc' in col]
    cols+=[col for col in df.columns if 'ppt' in col]
    cols+=[col for col in df.columns if 'vpd' in col]
    cols+=[col for col in df.columns if 'fwi' in col]
    
    df = df[cols]
    df['lfmc_t_1_inside_anomaly'] = df['lfmc_t_1_inside'] - df['lfmc_t_1_seasonal_mean_inside']
    df['lfmc_t_1_outside_anomaly'] = df['lfmc_t_1_outside'] - df['lfmc_t_1_seasonal_mean_outside']
    
    df.drop(['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside'],axis = 1, inplace = True)
        
    ###testing with random numbers instead of LFMC
    # df.loc[:,remove_lfmc] = np.zeros(shape = df.loc[:,remove_lfmc].shape)
    # clf = RandomForestClassifier(max_depth=15, min_samples_leaf = 5, random_state=0, oob_score = True,n_estimators = 50)
    clf = RandomForestClassifier(max_depth=6, random_state=0, oob_score = True,n_estimators = 20)

    allVars, s1 = ensemble_auc(df, size_dict, clf)
    
    
    # allVars = calc_auc(df, size_dict, clf)

    remove_lfmc = [col for col in df.columns if 'lfmc' in col]
    onlyClimate, s2 = ensemble_auc(df.drop(remove_lfmc, axis = 1), size_dict, clf)
    
    diff = (allVars - onlyClimate).copy().astype(float).round(3)
    onlyClimate.index.name = "only climate"
    diff.index.name = "difference, mean"
    allVars.index.name = "all variables"
    
    sd = (s1.pow(2)+s2.pow(2)).pow(0.5).astype(float).round(3)
    sd.index.name = "difference, sd"
    # print(onlyClimate.astype(float).round(2))
    # print(allVars.astype(float).round(2))
    # print(diff.astype(float).round(2))
    # print(sd.astype(float).round(2))
    
    return diff, sd, onlyClimate

def plot_importance(mean, std, onlyClimate):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6,3), sharey = True)
    
    ax1.barh(width = onlyClimate['small'],y = onlyClimate.index,edgecolor = list(mean.index.map(color_dict).values), color = "w")
    ax2.barh(width = onlyClimate['large'],y = onlyClimate.index,edgecolor = list(mean.index.map(color_dict).values), color = "w")

    
    ax1.barh(width = mean['small'],y = mean.index, left = onlyClimate['small'],\
             color = list(mean.index.map(color_dict).values), \
             edgecolor = list(mean.index.map(color_dict).values),\
                 xerr = std['small'])
    ax2.barh(width = mean['large'], y = mean.index, left = onlyClimate['large'], \
             color = list(mean.index.map(color_dict).values),\
                edgecolor = list(mean.index.map(color_dict).values),\
                 xerr = std['large'])
    
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    ax1.set_xlabel('AUC')
    ax2.set_xlabel('AUC')
    
    ax1.set_xlim(0.5,1)
    ax2.set_xlim(0.5,1)
    ax1.set_title("Small fires")
    ax2.set_title("Large fires")
    
    
mean, std, onlyClimate = calc_auc_diff(df, SIZE_DICT)

plot_importance(mean, std, onlyClimate)
# print(mean)
# print(std)
