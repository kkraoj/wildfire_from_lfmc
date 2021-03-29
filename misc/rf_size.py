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


def calc_auc_size(dfsub, clf):
    df = dfsub.copy()
    auc = pd.DataFrame(index = sorted(df.landcover.unique()),columns = ['auc'])
    for lc in sorted(df.landcover.unique()):
        ndf = df.loc[df.landcover==lc].copy()
      
        # ndf['size'] = 0
        # ndf.loc[ndf.area>4,'size'] = 1
        ndf = ndf.sample(frac=1).reset_index(drop=True)
        ndf.dropna(inplace = True)
        # print(ndf.columns)
        X = ndf.drop(['size', 'area', "landcover"], axis = 1)
        y = ndf['size']
        # print(y.mean())
        try:
            clf.fit(X, y)
            # rfc_disp = plot_roc_curve(clf, X, y, ax=ax,label = lc,color = color_dict[lc])
            auc.loc[lc,'auc'] = roc_auc_score(y, clf.predict(X))
            # print(roc_auc_score(y, clf.predict(X)))
        except: 
            print("Could not fit RF for land cover: %s"%(lc))
    # print(auc)        
    return auc

def ensemble_auc(dfsub, clf, iters = 100, label = 'All variables'):
    clf.random_state = 0
    dummy = calc_auc_size(dfsub, clf)
    aucs = np.expand_dims(dummy.values, axis = 2)
    for itr in range(1, iters):
        clf.random_state = itr
        auc = np.expand_dims(calc_auc_size(dfsub, clf).values, axis = 2)
        
        aucs = np.append(aucs,auc, axis = 2)
    # print("aucs ready")
    dummy.loc[:,:] = np.nanmean(aucs.astype(float), axis = 2)
    mean = dummy.copy()
    dummy.loc[:,:] = np.nanstd(aucs.astype(float), axis = 2)
    sd = dummy.copy()
    
    return mean, sd
    

def calc_auc_diff(dfs, replace_by_random = False):
    df = dfs.copy()
    allVars = pd.DataFrame(index = sorted(df.landcover.unique()),columns = ['auc'])
    onlyClimate = allVars.copy()
    cols = [col for col in df.columns if 'lfmc' in col]+['landcover', "area"]
    # cols = ['landcover']
    cols+=[col for col in df.columns if 'erc' in col]
    cols+=[col for col in df.columns if 'ppt' in col]
    cols+=[col for col in df.columns if 'vpd' in col]
    cols+=[col for col in df.columns if 'fwi' in col]
    
    df = df[cols]
    df['lfmc_t_1_inside_anomaly'] = df['lfmc_t_1_inside'] - df['lfmc_t_1_seasonal_mean_inside']
    df['lfmc_t_1_outside_anomaly'] = df['lfmc_t_1_outside'] - df['lfmc_t_1_seasonal_mean_outside']
    
    df.drop(['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside'],axis = 1, inplace = True)
    df['size'] = 0
    df.loc[df.area>4,'size'] = 1
    
    
    remove_outside = [col for col in df.columns if "outside" in col]
    df.drop(remove_outside, axis = 1,inplace = True)
    
    ###testing with random numbers instead of LFMC
    # df.loc[:,remove_lfmc] = np.zeros(shape = df.loc[:,remove_lfmc].shape)
    # clf = RandomForestClassifier(max_depth=15, min_samples_leaf = 5, random_state=0, oob_score = True,n_estimators = 50)
    clf = RandomForestClassifier(max_depth=10, min_samples_leaf = 1, random_state=0, oob_score = True,n_estimators = 20)

    allVars, s1 = ensemble_auc(df, clf)
    
    remove_lfmc = [col for col in df.columns if 'lfmc' in col]
    
    if replace_by_random:
        ###testing with random numbers instead of LFMC
        df.loc[:,remove_lfmc] = np.zeros(shape = df.loc[:,remove_lfmc].shape)
        onlyClimate, s2 = ensemble_auc(df, clf)
    else:
        onlyClimate, s2 = ensemble_auc(df.drop(remove_lfmc, axis = 1), clf)
    
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
    fig, ax1= plt.subplots(figsize = (3,3))
    
    ax1.barh(width = onlyClimate['auc'],y = onlyClimate.index,edgecolor = list(onlyClimate.index.map(color_dict).values), color = "w")

    ax1.barh(width = mean['auc'],y = mean.index, left = onlyClimate['auc'],\
             color = list(mean.index.map(color_dict).values), \
             edgecolor = list(mean.index.map(color_dict).values),\
                 xerr = std['auc'])

    ax1.set_ylabel("")
    ax1.set_xlabel('AUC')
    
    ax1.set_xlim(0.5,1)
    # ax1.set_title("Small fires")


  
def overlap_importance_trait(mean_, std_):
    mean = mean_.copy()
    std = std_.copy()
    trait = pd.read_excel(os.path.join(dir_root, "working.xlsx"), sheet_name = "mean_traits", index_col = "landcover", dtype = {'landcover':str})
    new_index =  list(trait.index)
    new_index = [x.replace("\\n"," ") for x in new_index]
    trait.index= new_index
    traitSd = pd.read_excel(os.path.join(dir_root, "working.xlsx"), sheet_name = "std_traits", index_col = "landcover", dtype = {'landcover':str})
    traitSd.index= new_index
    # mean.index = mean.index.astype(str)
    mean.index.name = "landcover"
    colors = [color_dict[lc] for lc in mean.index]
    mean.index = mean.index.str.replace("\n"," ")
    
    std.index.name = "landcover"
    std.index = std.index.str.replace("\n"," ")
    
    mean = trait.join(mean)
    std = trait.join(std)
        
    
    sns.set(style='ticks',font_scale = 1.1, rc = {"xtick.direction": "in","ytick.direction": "in"})
    fig,  axs = plt.subplots(1, 3, figsize = (9,3), sharey = "row",sharex = "col")
    ctr = 0
    
    ecolor = "lightgrey"
    s = 100
    
    fire_size = "auc"
    axs[0].errorbar(x = mean['p50'], y = mean[fire_size], yerr = std[fire_size], xerr = traitSd['p50'], fmt = 'o', color = ecolor, capsize = 2, zorder = -1)
    axs[0].scatter(x = mean['p50'], y = mean[fire_size],marker = 'o', edgecolor = ecolor,color = colors, s = s)
    # axs[0,ctr].plot(mean,'o-',color = color_dict[lc], markeredgecolor = "grey")
    
    # axs[ctr,0].set_xlabel('P50 (Mpa)')
    axs[0].set_xlim(-3, -6)
    
    axs[1].errorbar(x = mean['sigma'], y = mean[fire_size], yerr = std[fire_size], xerr = traitSd['sigma'], fmt = 'o', color = ecolor, capsize = 2, zorder = -1)
    axs[1].scatter(x = mean['sigma'], y = mean[fire_size],marker = 'o', edgecolor = ecolor,color = colors,s = s)
    # axs[0,ctr].plot(mean,'o-',color = color_dict[lc], markeredgecolor = "grey")
    
    # axs[ctr,1].set_xlabel('$\sigma$')
    axs[1].set_xlim(0.7,0.5)
    
    axs[2].errorbar(x = mean['rootDepth'], y = mean[fire_size], yerr = std[fire_size], xerr = traitSd['rootDepth'], fmt = 'o', color = ecolor, capsize = 2, zorder = -1)
    axs[2].scatter(x = mean['rootDepth'], y = mean[fire_size],marker = 'o', edgecolor =ecolor,color = colors,s = s)
    # axs[0,ctr].plot(mean,'o-',color = color_dict[lc], markeredgecolor = "grey")
    # axs[2,0].set_ylabel("LFMC Importance")
    # axs[2,ctr].set_xlabel('Rooting depth (m)')
    axs[2].set_xlim(2.5,5.5)

    # axs[0,0].set_xticklabels(None)
    axs[0].set_ylabel("LFMC Importance")
    axs[0].set_ylabel("LFMC Importance")
    axs[0].set_xlabel('P50 (Mpa)')
    axs[1].set_xlabel('Anisohydricity')
    axs[2].set_xlabel('Rooting depth (m)')
    # axs[0,0].set_title("Small fires")
    # axs[0,1].set_title("Large fires")
    # ax.set_xticks(xticks)
    # ax1.set_xlim(0.5,1)
    # ax.set_title("%s, N = %d"%(lc, n))
    
    return axs


mean, std, onlyClimate = calc_auc_diff(df)

plot_importance(mean, std, onlyClimate)

axs = overlap_importance_trait(mean, std)
# print(mean)
# print(std)
