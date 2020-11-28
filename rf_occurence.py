# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:33:04 2020

@author: kkrao
"""


import pandas as pd
from init import dir_data, lc_dict, color_dict, dir_root, short_lc
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
    # df = pd.read_csv(os.path.join(dir_data, "fire_collection_median_with_climate_500m_variogram_6_jul_2020.csv"))
    df = pd.read_csv(os.path.join(dir_data, "fire_collection_median_with_climate_500m_variogram.csv"))
    
    dfr = pd.read_csv(os.path.join(dir_data, "fire_collection_median_extra_lfmc_vars_500m_variogram.csv"))
    dfr = dfr[['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside', 'lfmc_t_2_inside', 'lfmc_t_2_outside']]
    df = df.join(dfr)
    
    dfr = pd.read_csv(os.path.join(dir_data, "fire_collection_median_fwi_500m_variogram.csv"))
    dfr = dfr[['fwi_t_4_inside','fwi_t_4_outside']]
    df = df.join(dfr)
    
    df = df.loc[df.landcover.isin(lc_dict.keys())]
    df['landcover'] = df.landcover.map(lc_dict)
    df = df.loc[df.BurnDate>=150]
    drop_erc = [col for col in df.columns if "erc" in col]
    df.drop(drop_erc, axis = 1, inplace = True)
    return df

df = assemble_df()


#%% just lfmc first 

SIZE_DICT = {'small':(df.area<=4),
             'large':(df.area>4)}


def calc_auc_occurence(dfsub, size_dict, clf):
    df = dfsub.copy()
    auc = pd.DataFrame(index = sorted(df.landcover.unique()),columns = size_dict.keys())
    for fire_size in size_dict.keys():
        if fire_size=='small':
            clf = RandomForestClassifier(max_depth=10, random_state=0, oob_score = False,n_estimators = 40)
        else:
            clf = RandomForestClassifier(max_depth=6, random_state=0, oob_score = True,n_estimators = 40)
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
    
            X = ndf.drop(['fire'], axis = 1)
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
    

def calc_auc_diff(dfs, size_dict, replace_by_random = False):
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
    # clf = RandomForestClassifier(max_depth=6, random_state=0, oob_score = True,n_estimators = 40)
    clf = RandomForestClassifier(max_depth=10, random_state=0, oob_score = False,n_estimators = 40)
    # clf = RandomForestClassifier(min_samples_leaf = 10, random_state=0, oob_score = True,n_estimators = 40)
    allVars, sdl = ensemble_auc(df, size_dict, clf)
    
    
    # allVars = calc_auc(df, size_dict, clf)

    remove_lfmc = [col for col in df.columns if 'lfmc' in col]
    if replace_by_random:
        ###testing with random numbers instead of LFMC
        df.loc[:,remove_lfmc] = np.ones(shape = df.loc[:,remove_lfmc].shape)
        onlyClimate, sdc = ensemble_auc(df, size_dict, clf)
    else:
        onlyClimate, sdc = ensemble_auc(df.drop(remove_lfmc, axis = 1), size_dict, clf)
    
    diff = (allVars - onlyClimate).copy().astype(float).round(3)
    onlyClimate.index.name = "only climate"
    diff.index.name = "difference, mean"
    allVars.index.name = "all variables"
    
    # sd = (s1.pow(2)+s2.pow(2)).pow(0.5).astype(float).round(3)
    # sd.index.name = "difference, sd"
    # print(onlyClimate.astype(float).round(2))
    # print(allVars.astype(float).round(2))
    # print(diff.astype(float).round(2))
    # print(sd.astype(float).round(2))
    
    return diff, sdl,sdc, onlyClimate
def plot_importance(mean, stdl,std, onlyClimate):
    
    height = 0.3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6,3), sharey = True, dpi = 300)
    
    ax1.barh(width = onlyClimate['small'],y = onlyClimate.index,edgecolor = list(mean.index.map(color_dict).values), height = height,color = "w")
    ax2.barh(width = onlyClimate['large'],y = onlyClimate.index,edgecolor = list(mean.index.map(color_dict).values), height = height,color = "w")

    onlyClimate = onlyClimate.fillna(0.0)
    ax1.barh(width = mean['small']+onlyClimate['small'],y = mean.index,\
             color = list(mean.index.map(color_dict).values), \
             edgecolor = list(mean.index.map(color_dict).values),\
                 xerr = std['small'])
    ax2.barh(width = mean['large'], y = mean.index, left = onlyClimate['large'], \
             color = list(mean.index.map(color_dict).values),\
                edgecolor = list(mean.index.map(color_dict).values),\
                 xerr = std['large'])
    
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    ax1.set_xlabel('Area under curve')
    ax2.set_xlabel('Area under curve')
    ax1.set_xticks(np.linspace(0.5,1,6))
    ax2.set_xticks(np.linspace(0.5,1,6))
    
    ax1.set_xlim(0.5,1)
    ax2.set_xlim(0.5,1)
    ax1.set_title("Small fires ($\leq$400 Ha)", color = "saddlebrown")
    ax2.set_title("Large fires (>400 Ha)", color = "saddlebrown")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # plt.yticks(ax1.get_yticks(), mean.index,linespacing = 0.0)
    ax1.set_yticklabels(mean.index,linespacing = 0.8)
    
def plot_importance_old(mean, std, onlyClimate):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6,3), sharey = True, dpi = 300)
    
    ax1.barh(width = onlyClimate['small'],y = onlyClimate.index,edgecolor = list(mean.index.map(color_dict).values), color = "w")
    ax2.barh(width = onlyClimate['large'],y = onlyClimate.index,edgecolor = list(mean.index.map(color_dict).values), color = "w")

    onlyClimate = onlyClimate.fillna(0.0)
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
    ax1.set_xlabel('Area under curve')
    ax2.set_xlabel('Area under curve')
    ax1.set_xticks(np.linspace(0.5,1,6))
    ax2.set_xticks(np.linspace(0.5,1,6))
    
    ax1.set_xlim(0.5,1)
    ax2.set_xlim(0.5,1)
    ax1.set_title("Small fires ($\leq$400 Ha)", color = "saddlebrown")
    ax2.set_title("Large fires (>400 Ha)", color = "saddlebrown")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # plt.yticks(ax1.get_yticks(), mean.index,linespacing = 0.0)
    ax1.set_yticklabels(mean.index,linespacing = 0.8)
    
    
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
    fig,  axs = plt.subplots(2, 3, figsize = (9,6), sharey = "row",sharex = "col")
    ctr = 0
    
    ecolor = "lightgrey"
    s = 100
    for fire_size in ['small','large']:

        axs[ctr,0].errorbar(x = mean['p50'], y = mean[fire_size], yerr = std[fire_size], xerr = traitSd['p50'], fmt = 'o', color = ecolor, capsize = 2, zorder = -1)
        axs[ctr,0].scatter(x = mean['p50'], y = mean[fire_size],marker = 'o', edgecolor = ecolor,color = colors, s = s)
    # axs[0,ctr].plot(mean,'o-',color = color_dict[lc], markeredgecolor = "grey")
        
        # axs[ctr,0].set_xlabel('P50 (Mpa)')
        axs[ctr,0].set_xlim(-3, -6)
        
        axs[ctr,1].errorbar(x = mean['sigma'], y = mean[fire_size], yerr = std[fire_size], xerr = traitSd['sigma'], fmt = 'o', color = ecolor, capsize = 2, zorder = -1)
        axs[ctr,1].scatter(x = mean['sigma'], y = mean[fire_size],marker = 'o', edgecolor = ecolor,color = colors,s = s)
    # axs[0,ctr].plot(mean,'o-',color = color_dict[lc], markeredgecolor = "grey")
        
        # axs[ctr,1].set_xlabel('$\sigma$')
        axs[ctr,1].set_xlim(0.7,0.5)
        
        axs[ctr,2].errorbar(x = mean['rootDepth'], y = mean[fire_size], yerr = std[fire_size], xerr = traitSd['rootDepth'], fmt = 'o', color = ecolor, capsize = 2, zorder = -1)
        axs[ctr,2].scatter(x = mean['rootDepth'], y = mean[fire_size],marker = 'o', edgecolor =ecolor,color = colors,s = s)
    # axs[0,ctr].plot(mean,'o-',color = color_dict[lc], markeredgecolor = "grey")
        # axs[2,0].set_ylabel("LFMC Importance")
        # axs[2,ctr].set_xlabel('Rooting depth (m)')
        axs[ctr,2].set_xlim(2.5,5.5)
        
        ctr+=1
    # axs[0,0].set_xticklabels(None)
    axs[0,0].set_ylabel("LFMC Importance")
    axs[1,0].set_ylabel("LFMC Importance")
    axs[1,0].set_xlabel('P50 (Mpa)')
    axs[1,1].set_xlabel('Anisohydricity')
    axs[1, 2].set_xlabel('Rooting depth (m)')
    # axs[0,0].set_title("Small fires")
    # axs[0,1].set_title("Large fires")
    # ax.set_xticks(xticks)
    # ax1.set_xlim(0.5,1)
    # ax.set_title("%s, N = %d"%(lc, n))
    
    return axs

def overlap_importance_trait_TRY(mean_, std_):
    mean = mean_.copy()
    std = std_.copy()
    
    df = pd.read_excel(os.path.join(dir_root, "data","traits","TRY","dictionary_fuels_species.xlsx"))
    
    df = df.loc[df.lc.isin(lc_dict.keys())]
    df.lc = df.lc.map(lc_dict)
    
    traitmean = df.groupby('lc').p50.mean()
    
    traitstd = df.groupby('lc').p50.std()
    
    mean.index.name = "lc"
    std.index.name = "lc"
    
    mean = mean.join(traitmean)
    std = std.join(traitstd)
    colors = [color_dict[lc] for lc in mean.index]
    sns.set(style='ticks',font_scale = 1.1, rc = {"xtick.direction": "in","ytick.direction": "in"})
    fig,  axs = plt.subplots(2, 1, figsize = (3,6), sharey = "row",sharex = "col")
    ctr = 0
    
    ecolor = "lightgrey"
    s = 100
    for fire_size in ['small','large']:

        axs[ctr].errorbar(x = mean['p50'], y = mean[fire_size], yerr = std[fire_size], xerr = std['p50'], fmt = 'o', color = ecolor, capsize = 2, zorder = -1)
        axs[ctr].scatter(x = mean['p50'], y = mean[fire_size],marker = 'o', edgecolor = ecolor,color = colors, s = s)
        axs[ctr].set_xlim(-3, -6)
        ctr+=1
    # axs[0,0].set_xticklabels(None)
    axs[0].set_ylabel("LFMC Importance")
    axs[1].set_ylabel("LFMC Importance")
    axs[1].set_xlabel('P50 (Mpa)')
    
def trait_by_lc():
    trait = pd.read_excel(os.path.join(dir_root, "working.xlsx"), sheet_name = "mean_traits", index_col = "landcover", dtype = {'landcover':str})
    traitSd = pd.read_excel(os.path.join(dir_root, "working.xlsx"), sheet_name = "std_traits", index_col = "landcover", dtype = {'landcover':str})
    
    new_index =  list(trait.index)
    new_index = [x.replace("\\n"," ") for x in new_index]
    trait.index= new_index
    traitSd.index= new_index
    
    fig, axs= plt.subplots(1, 3 , figsize = (9, 3), sharey = True)
    
    axs[0].errorbar(x = trait['p50'], y = range(trait.shape[0]), xerr = traitSd['p50'], fmt = 'o', color = ecolor, capsize = 2, zorder = -1)
    axs[0].scatter(x = trait['p50'], y = range(trait.shape[0]),marker = 'o', edgecolor = ecolor,color = 'k',s = s)
    axs[1].errorbar(x = trait['sigma'], y = range(trait.shape[0]), xerr = traitSd['sigma'], fmt = 'o', color = ecolor, capsize = 2, zorder = -1)
    axs[1].scatter(x = trait['sigma'], y = range(trait.shape[0]),marker = 'o', edgecolor = ecolor,color = 'k',s = s)
    axs[2].errorbar(x = trait['rootDepth'], y = range(trait.shape[0]), xerr = traitSd['rootDepth'], fmt = 'o', color = ecolor, capsize = 2, zorder = -1)
    axs[2].scatter(x = trait['rootDepth'], y = range(trait.shape[0]),marker = 'o', edgecolor = ecolor,color = 'k',s = s)
    
    
    axs[0].set_xlabel('P50 (Mpa)')
    axs[1].set_xlabel('Anisohydricity')
    axs[2].set_xlabel('Rooting depth (m)')
    
    axs[0].set_xlim(-2, -7)
    axs[1].set_xlim(1,0)
    axs[2].set_xlim(1,8)

    axs[0].set_yticks(range(trait.shape[0]))
    axs[0].set_yticklabels(trait.index.values)

   
def overlap_importance_trait_TRY_yanlan_table(mean, std):

    trait = pd.read_excel(os.path.join(dir_root, "data","traits","TRY","TRY_Hydraulic_Traits_Yanlan.xlsx"))
    # new_index =  list(trait.index)
    # new_index = [x.replace("\\n"," ") for x in new_index]
    # trait.index= new_index
    # trait.columns
    trait = trait.loc[trait['PFT'].isin(short_lc.keys())]
    
    trait = trait.rename(columns = {"Water potential at 50% loss of conductivity Psi_50 (MPa)":"p50","PFT":"landcover"})
    trait['landcover'] = trait['landcover'].map(short_lc)
    trait['landcover'] = trait['landcover'].str.replace("\n"," ")

    traitMean = trait.groupby("landcover")["p50"].mean()
    traitSd = trait.groupby("landcover")["p50"].std()
    
    # traitMean.index = traitMean.index.map(short_lc)
    
    # traitSd = pd.read_excel(os.path.join(dir_root, "working.xlsx"), sheet_name = "std_traits", index_col = "landcover", dtype = {'landcover':str})
    # traitSd.index= new_index
    # mean.index = mean.index.astype(str)
    mean_ = mean.copy()
    std_ = std.copy()
    mean_.index.name = "landcover"
    colors = [color_dict[lc] for lc in mean_.index]
    mean_.index = mean_.index.str.replace("\n"," ")
    
    std_.index.name = "landcover"
    std_.index = std_.index.str.replace("\n"," ")
    
    mean_ = mean_.join(traitMean)
    std_ = std_.join(traitSd)
        
    
    sns.set(style='ticks',font_scale = 1.1, rc = {"xtick.direction": "in","ytick.direction": "in"})
    fig,  axs = plt.subplots(1, 2, figsize = (6,3), sharey = True, dpi = 300)
    ctr = 0
    
    ecolor = "grey"
    s = 100
    for fire_size in ['small','large']:
        sns.regplot(x=mean_['p50'], y = mean_[fire_size],ax=axs[ctr], color = "lightgrey", order=1, ci = 95)
        axs[ctr].errorbar(x = mean_['p50'], y = mean_[fire_size], yerr = std_[fire_size], xerr = std_['p50'], fmt = 'o', color = ecolor, capsize = 2, zorder = 20)
        axs[ctr].scatter(x = mean_['p50'], y = mean_[fire_size],marker = 'o', edgecolor = ecolor,color = colors, s = s, zorder = 30)
        axs[ctr].set_xlim(-1, -5)
        axs[ctr].set_ylim(0,0.1)
        axs[ctr].spines['right'].set_visible(False)
        axs[ctr].spines['top'].set_visible(False)
        
        ctr+=1
    # axs[0,0].set_xticklabels(None)
    axs[0].set_ylabel("Gain by including\nlive fuel moisture")
    axs[1].set_ylabel("")
    # axs[1].set_ylabel("LFMC Importance")
    axs[1].set_xlabel('P50 (Mpa)')
    axs[0].set_xlabel('P50 (Mpa)')
    
    return axs

mean, stdl,stdc, onlyClimate = calc_auc_diff(df, SIZE_DICT, replace_by_random = False)
index = ['Grassland', 'Mixed forest', 'Shrub/grassland','Closed broadleaf\ndeciduous','Closed needleleaf\nevergreen', 'Shrubland']
mean = mean.loc[index]
stdl = stdl.loc[index]
stdc = stdc.loc[index]
onlyClimate = onlyClimate.loc[index]
plot_importance(mean, stdl,std, onlyClimate)
axs = overlap_importance_trait_TRY_yanlan_table(mean, std)

# axs = overlap_importance_trait_TRY(mean, std)
# print(mean)
# print(std)
# mean_ = mean.copy()
# std_ = std.copy()
