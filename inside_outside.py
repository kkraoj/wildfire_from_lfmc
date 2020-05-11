# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 06:16:01 2020

@author: kkrao
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from scipy.special import expit


dir_data = r"D:\Krishna\projects\wildfire_from_lfmc\data\tables"
os.chdir(dir_data)
sns.set(style='ticks',font_scale = 1.5)

#%% ####################################################################
#distribution of fire per landcover type
df = pd.read_csv(os.path.join(dir_data, "fire_collection_median_no_climate_extra_lfmc_vars.csv"))
df.head()
df.columns
df.landcover.unique()

#%% initialize plot
lc_dict = { 
            50: 'Closed broadleaf\ndeciduous',
            70: 'Closed needleleaf\nevergreen',
            90: 'Mixed forest',
            100:'Mixed forest',
            110:'Shrub/grassland',
            120:'Shrub/grassland',
            130:'Shrubland',
            140:'Grassland',
            }
df = df.loc[df.landcover.isin(lc_dict.keys())]
df['landcover'] = df.landcover.map(lc_dict)


#%% histogram of LFMC inside vs. outside
# sub = df[['lfmc_t_1_inside','lfmc_t_1_outside','landcover']]
# fig, ax = plt.subplots()
# sub['lfmc_t_1_inside'].plot.hist(color = 'darkred',alpha = 0.5,bins = 50,ax=ax,label = 'Burned area')
# sub['lfmc_t_1_outside'].plot.hist(color = 'lime',alpha = 0.5,bins = 50,ax=ax,label = 'Unaffected area')
# ax.set_xlabel('LFMC (%)')
# plt.legend()

fig, ax = plt.subplots()
(df['lfmc_t_1_inside']-df['lfmc_t_1_outside']).plot.hist(color = 'grey',alpha = 0.5,bins = 50,ax=ax,label = 'Difference(BA-UA)')
ax.set_xlabel('LFMC (%)')
plt.legend()


data = df['lfmc_t_2_inside']-df['lfmc_t_2_outside']
fig, ax = plt.subplots()
data.plot.hist(color = 'grey',alpha = 0.5,bins = 50,ax=ax,label = 't-2')
ax.set_xlabel('LFMC (%)')
ax.axvline(data.mean(),label = 'mean',color = 'k',linewidth = 2, linestyle = '--')
plt.legend()

data = df['lfmc_t_2_inside']-df['lfmc_t_1_inside']
fig, ax = plt.subplots()
data.plot.hist(color = 'grey',alpha = 0.5,bins = 50,ax=ax,label = '(t-2)-(t-1) inside')
ax.set_xlabel('LFMC (%)')
ax.axvline(data.mean(),label = 'mean',color = 'k',linewidth = 2, linestyle = '--')
ax.set_xlim(-100,100)
plt.legend()

data = df['lfmc_t_2_outside']-df['lfmc_t_1_outside']
fig, ax = plt.subplots()
data.plot.hist(color = 'grey',alpha = 0.5,bins = 50,ax=ax,label = '(t-2)-(t-1) outside')
ax.set_xlabel('LFMC (%)')
ax.axvline(data.mean(),label = 'mean',color = 'k',linewidth = 2, linestyle = '--')
ax.set_xlim(-100,100)
plt.legend()

data = df['lfmc_t_2_inside']-df['lfmc_t_1_inside'] - df['lfmc_t_2_outside']+df['lfmc_t_1_outside']
fig, ax = plt.subplots()
data.plot.hist(color = 'grey',alpha = 0.5,bins = 50,ax=ax,label = '(t-2)-(t-1) BA-UA')
ax.set_xlabel('LFMC (%)')
ax.axvline(data.mean(),label = 'mean',color = 'k',linewidth = 2, linestyle = '--')
ax.set_xlim(-100,100)
plt.legend()

data = df['lfmc_t_1_inside']-df['lfmc_t_1_seasonal_mean_inside'] - df['lfmc_t_1_outside']+df['lfmc_t_1_seasonal_mean_outside']
fig, ax = plt.subplots()
data.plot.hist(color = 'grey',alpha = 0.5,bins = 50,ax=ax,label = "(t-1)' BA-UA")
ax.set_xlabel('LFMC (%)')
ax.axvline(data.mean(),label = 'mean',color = 'k',linewidth = 2, linestyle = '--')
ax.set_xlim(-100,100)
plt.legend()


#%%
## by landcover
# for lc in df.landcover.unique():
#     sub = df.loc[df.landcover==lc,['lfmc_t_1_inside','lfmc_t_1_outside','landcover']]
#     fig, ax = plt.subplots()
#     sub['lfmc_t_1_inside'].plot.hist(color = 'darkred',alpha = 0.5,bins = 50,ax=ax,label = 'Burned area',linewidth = 0)
#     sub['lfmc_t_1_outside'].plot.hist(color = 'lime',alpha = 0.5,bins = 50,ax=ax,label = 'Unaffected area',linewidth = 0)
#     ax.set_xlabel('LFMC (%)')
#     ax.set_title('%s'%lc)
#     plt.legend()

#%% comparing lfmc to other climate indices
#for var in ['lfmc','vpd','ppt']:    
#    fig, ax = plt.subplots()
#    df['%s_t_1_inside'%var].plot.hist(color = 'darkred',alpha = 0.5,bins = 50,ax=ax,label = 'Burned area')
#    df['%s_t_1_outside'%var].plot.hist(color = 'lime',alpha = 0.5,bins = 50,ax=ax,label = 'Unaffected area')
#    ax.set_xlabel('%s (%%)'%var.upper())
#    plt.legend()

#%% different timescales of VPD
# for var in ['vpd']:    
#     for t in [2,3,4,5,6]:
#         fig, ax = plt.subplots()
#         df['%s_t_%1d_inside'%(var,t)].plot.hist(color = 'darkred',alpha = 0.5,bins = 50,ax=ax,label = 'Burned area')
#         df['%s_t_%1d_outside'%(var,t)].plot.hist(color = 'lime',alpha = 0.5,bins = 50,ax=ax,label = 'Unaffected area')
#         ax.set_xlabel('%s, summed over previous %1d months'%(var.upper(),t))
#         plt.legend()



#%% logit
#X = np.append(no_fire,yes_fire)[np.newaxis].T
#y = np.append(np.repeat(0,len(no_fire)),np.repeat(1,len(yes_fire)))
#clf = LogisticRegression(random_state=0).fit(X, y)
#clf.predict(X[:2, :])
#clf.predict_proba(X[:2, :])
#
#clf.score(X, y)
#X_test = np.linspace(0, 250, 300)
#
#loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
#
#fig, ax = plt.subplots(figsize = (4,4))
#
#ax.scatter(x = no_fire, y = np.repeat(0,len(no_fire)),marker = 'o',color = 'grey',alpha = 0.01)
#ax.scatter(x = yes_fire, y = np.repeat(1,len(yes_fire)),marker = 'o',color = 'crimson',alpha = 0.01)
#
#ax.plot(X_test, loss, color='orange', linewidth=3,label = 'Prediction')
#ax.set_xlabel('LFMC(%)')
#ax.set_yticks([0,1])
#ax.set_yticklabels(['No fire','Fire'])
#plt.legend()

#%% variogram
# varh = [127.4,
# 124.9,
# 126.2,
# 127.3,
# 128.9,
# 131.4,
# 133.3,
# 135.5,
# 137.9,
# 139.5,
# 140.7,
# 141.4,
# 141.9,
# 142.5,
# 143.3,
# 143.9,
# 144.6,
# 144.9,
# 145.3,
# 145.7,
# 146.1,
# 146.4,
# 146.8,
# 147.1,
# 147.5,
# 147.9,
# 148.2,
# 148.6,
# 148.8,
# 149.1
# ]
# len(varh)
# fig, ax = plt.subplots()
# ax.plot(varh,'ok')
# ax.set_xlabel('Radius (km)')
# ax.set_ylabel('Variance')
