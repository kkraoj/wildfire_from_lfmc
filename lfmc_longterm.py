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
             'medium':(df.area>1)&(df.area<=10),
             'large':(df.area>10)}

# cols = [col for col in df.columns if 'lfmc' in col]+['landcover']
# df = df[cols]
# df['lfmc_t_1_inside_anomaly'] = df['lfmc_t_1_inside'] - df['lfmc_t_1_seasonal_mean_inside']
# df['lfmc_t_1_outside_anomaly'] = df['lfmc_t_1_outside'] - df['lfmc_t_1_seasonal_mean_outside']

# df.drop(['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside'],axis = 1, inplace = True)

# df.head()
# for fire_size in size_dict.keys():
#     dfcat = df.loc[size_dict[fire_size]].copy()
#     fig, ax = plt.subplots(figsize = (3,3))
#     for lc in sorted(dfcat.landcover.unique()):
#         sub = dfcat.loc[dfcat.landcover==lc]
#         ndf = pd.DataFrame()
        
#         for var in ['outside','inside']:    
#             cols = [col for col in sub.columns if var in col]
#             # cols.remove('lfmc_t_1_%s'%var)
#             data = sub[cols].copy()
#             new_cols = [col.split('_')[0] for col in data.columns]
#             data.columns = (new_cols)
#             data['fire'] = int(var=='inside')
#             ndf = pd.concat([ndf, data], axis = 0).reset_index(drop=True)
            
#         print(ndf.shape)
#         # lfmc_df = pd.get_dummies(ndf['lfmc'], prefix='lfmc')
#         # ndf = ndf.drop('lfmc',axis = 1)
#         # ndf = ndf.join(lfmc_df)
#         ndf = ndf.sample(frac=1).reset_index(drop=True)
#         ndf.dropna(inplace = True)
#         X = ndf.drop('fire', axis = 1)
#         y = ndf['fire']
        
#         clf = RandomForestClassifier(max_depth=5, random_state=0)
#         clf.fit(X, y)
        
#         rfc_disp = plot_roc_curve(clf, X, y, ax=ax,label = lc,color = color_dict[lc])
#         # print('%s:\t %0.2f'%(lc,roc_auc_score(y, clf.predict(X))))
#         # print('%0.2f'%(roc_auc_score(y, clf.predict(X))))
#         master.loc[lc,fire_size] = roc_auc_score(y, clf.predict(X))

# print(master)

#%% LFMC + climate
allVars = pd.DataFrame(index = sorted(df.landcover.unique()),columns = size_dict.keys())
onlyClimate = allVars.copy()
cols = [col for col in df.columns if 'lfmc' in col]+['landcover']
# cols = ['landcover']
cols+=[col for col in df.columns if 'erc' in col]
cols+=[col for col in df.columns if 'ppt' in col]
cols+=[col for col in df.columns if 'vpd' in col]


df = df[cols]
df['lfmc_t_1_inside_anomaly'] = df['lfmc_t_1_inside'] - df['lfmc_t_1_seasonal_mean_inside']
df['lfmc_t_1_outside_anomaly'] = df['lfmc_t_1_outside'] - df['lfmc_t_1_seasonal_mean_outside']

df.drop(['lfmc_t_1_seasonal_mean_inside','lfmc_t_1_seasonal_mean_outside'],axis = 1, inplace = True)
remove_lfmc = [col for col in df.columns if 'lfmc' in col]

###testing with random numbers instead of LFMC
# df.loc[:,remove_lfmc] = np.zeros(shape = df.loc[:,remove_lfmc].shape)
clf = RandomForestClassifier(max_depth=6, random_state=0, oob_score = True,n_estimators = 20)

# df.head()
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
            
        # print(ndf.shape)
        # lfmc_df = pd.get_dummies(ndf['lfmc'], prefix='lfmc')
        # ndf = ndf.drop('lfmc',axis = 1)
        # ndf = ndf.join(lfmc_df)
        ndf = ndf.sample(frac=1).reset_index(drop=True)
        ndf.dropna(inplace = True)
        # try:    
        #     sns.pairplot(ndf, hue = 'fire')
        #     plt.show()
        # except ValueError:
        #     print('error')
            
        # print(ndf.columns)
        # ndf['intercept'] = 1
        X = ndf.drop('fire', axis = 1)
        y = ndf['fire']

        clf.fit(X, y)
        # logit = sm.Logit(y,X) ##statsmodels logit
        # clf = logit.fit() ##statsmodels logit
        rfc_disp = plot_roc_curve(clf, X, y, ax=ax,label = lc,color = color_dict[lc])
        
        # print('%s:\t %0.2f'%(lc,roc_auc_score(y, clf.predict(X))))
        # print('%0.2f'%(roc_auc_score(y, clf.predict(X))))
        # imp = clf.feature_importances_[0] + clf.feature_importances_[1] + clf.feature_importances_[5]
        # print(imp)
        allVars.loc[lc,fire_size] = roc_auc_score(y, clf.predict(X))
        
            
        # break
    # break
        # plt.show()
        # master.loc[lc,fire_size] = imp
# print(df.shape)
remove_lfmc = [col for col in df.columns if 'lfmc' in col]
df.drop(remove_lfmc, inplace = True, axis = 1)

# df.head()
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
            
        # print(ndf.shape)
        # lfmc_df = pd.get_dummies(ndf['lfmc'], prefix='lfmc')
        # ndf = ndf.drop('lfmc',axis = 1)
        # ndf = ndf.join(lfmc_df)
        ndf = ndf.sample(frac=1).reset_index(drop=True)
        ndf.dropna(inplace = True)
        # print(ndf.columns)
        # ndf['intercept'] = 1
        X = ndf.drop('fire', axis = 1)
        y = ndf['fire']
        # clf = RandomForestClassifier(max_depth=5, random_state=0)
        clf.fit(X, y)
        # logit = sm.Logit(y,X) ##statsmodels logit
        # clf = logit.fit() ##statsmodels logit
        rfc_disp = plot_roc_curve(clf, X, y, ax=ax,label = lc,color = color_dict[lc])
        # print('%s:\t %0.2f'%(lc,roc_auc_score(y, clf.predict(X))))
        # print('%0.2f'%(roc_auc_score(y, clf.predict(X))))
        # imp = clf.feature_importances_[0] + clf.feature_importances_[1] + clf.feature_importances_[5]
        # print(imp)
        onlyClimate.loc[lc,fire_size] = roc_auc_score(y, clf.predict(X))

diff = (allVars - onlyClimate).copy()
onlyClimate.index.name = "only climate"
diff.index.name = "difference"
allVars.index.name = "all variables"

print(onlyClimate.astype(float).round(2))
print(allVars.astype(float).round(2))
print(diff.astype(float).round(2))

#%% correct using ndvi


# for lc in ['enf','bdf','mixed','shrub','grass']:
#     lt = pd.read_csv(os.path.join(dir_root,'data','longterm','%s.csv'%lc),dtype = float,thousands=r',')
#     lt['NDVI']/=1e4
    
#     fig, ax = plt.subplots(figsize = (2,2))
#     sns.regplot(y = 'LFMC',x ='NDVI',data = lt,\
#                 order = 1,
#                 ax = ax,color = color_dict[short_lc[lc]],scatter_kws = {'s':5},line_kws={'color' : 'k'})
#     ax.set_title(short_lc[lc])
    
    
#     X= lt['NDVI'].values.reshape(-1,1)
#     y = lt['LFMC'].values
#     reg = LinearRegression().fit(X,y)
#     R2 = reg.score(X, y)    
#     slope = reg.coef_[0]
#     reg.intercept_    
#     # R = np.corrcoef(lt['LFMC'],lt['NDVI'])[0][1]
#     ax.annotate('$R^2$ = %0.2f\nSlope = %4d'%(R2,slope), \
#                     xy=(0.1, 0.95), xycoords='axes fraction',\
#                     ha='left',va='top')
     
#%% longterm NDVI means  
# fig, ax = plt.subplots(figsize = (4,2))

# for lc in ['enf','bdf','mixed','shrub','grass']:
#     lt = pd.read_csv(os.path.join(dir_root,'data','longterm','longterm_%s.csv'%lc),dtype = {0:str, 1:float},thousands=r',',index_col = 0)
#     lt.index = pd.to_datetime(lt.index)
#     lt['NDVI']/=1e4
#     lt['NDVI'].plot(ax = ax,ms = 4,marker = 'o',color = color_dict[short_lc[lc]], label = short_lc[lc])
#     ax.set_xlabel('')
#     ax.set_ylabel('NDVI')
    
#     ### max diff
#     maxdiff = max((lt.loc[lt.index.year>2015] - lt.loc[lt.index.year<=2015].min()).abs().max()[0],(lt.loc[lt.index.year>2015] - lt.loc[lt.index.year<=2015].max()).abs().max()[0])
#     print(lc)
#     print(maxdiff)
