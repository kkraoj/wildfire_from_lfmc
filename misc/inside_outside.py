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
from scipy.stats import mannwhitneyu


dir_data = r"D:\Krishna\projects\wildfire_from_lfmc\data\tables"
os.chdir(dir_data)
sns.set(style='ticks',font_scale = 1.)

#%% ####################################################################
#distribution of fire per landcover type
df = pd.read_csv(os.path.join(dir_data, "fire_collection_median_with_climate_500m_variogram.csv"))
df.head()
df.columns
df.landcover.unique()

#%% initialize plot
units = {'lfmc':'(%)','vpd':'(hPa)','erc':'','ppt':r'(mm/month)'}
axis_lims = {'lfmc':[75,125],'vpd':[15,50],'erc':[20,70],'ppt':[0,120]}
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

#%% histogram of buffer zone ring sizes
# fig, ax = plt.subplots(figsize = (3,1))

# df.bufferDist.plot.hist(ax = ax,bins = 9)
# ax.set_xlabel('Buffer zone radius (m)')


#%% variograms

# fig, ax = plt.subplots(figsize = (3,1))
# df.loc[df.bufferDist == 2001].groupby('landcover').landcover.count().plot.bar(ax = ax)
# ax.set_xlabel('Landcover classes')
# ax.set_ylabel('Frequency')
# ax.set_title('Buffer zone = 2km')

# fig, ax = plt.subplots(figsize = (3,1))
# df.loc[df.bufferDist == 10001].groupby('landcover').landcover.count().plot.bar(ax = ax)
# ax.set_xlabel('Landcover classes')
# ax.set_ylabel('Frequency')
# ax.set_title('Buffer zone = 10km')

# fig, ax = plt.subplots(figsize = (3,3))
# df.bufferDist = (df.bufferDist/1000).astype(int)
# sns.boxplot('bufferDist','area',data = df.loc[df.area>=16],ax=ax,fliersize = 0)
# # ax.scatter(df.bufferDist, df.area)
# ax.set_xlabel('Buffer zone radius (km)')
# ax.set_ylabel('Fire size (km$^2$)')
# plt.yscale('log')
# print(df.shape)
# ax.set_title('Buffer zone = 2km')

#%% histogram of LFMC inside vs. outside
filters = (df.BurnDate>=160)&(df.area<=1)&(df['lfmc_t_1_inside']<120)

df = df.loc[filters]
fig, ax = plt.subplots()
df['lfmc_t_1_inside'].plot.hist(color = 'darkred',alpha = 0.5,bins = 50,ax=ax,label = 'Burned area')
df['lfmc_t_1_outside'].plot.hist(color = 'lime',alpha = 0.5,bins = 50,ax=ax,label = 'Unaffected area')
ax.set_xlabel('LFMC (%)')
plt.legend()

print('Number of fires: %d'%df.shape[0])

# data = df['lfmc_t_1_inside']-df['lfmc_t_1_outside']
# fig, ax = plt.subplots(figsize = (3,3))
# data.plot.hist(color = 'grey',alpha = 0.5,ax=ax,label = 'Difference(BA-UA)')
# ax.set_xlabel('LFMC (%)')
# ax.axvline(data.mean(),color = 'k',linewidth = 2, label = 'mean')
# plt.legend()


#%% histograms by landcover
# 
for lc in df.landcover.unique():
    sub = df.loc[df.landcover==lc,['lfmc_t_1_inside','lfmc_t_1_outside','landcover']]
    fig, ax = plt.subplots(figsize = (3,3))
    sub['lfmc_t_1_inside'].plot.hist(color = 'darkred',alpha = 0.5,ax=ax,label = 'Burned area',linewidth = 0)
    sub['lfmc_t_1_outside'].plot.hist(color = 'lime',alpha = 0.5,ax=ax,label = 'Unaffected area',linewidth = 0)
    ax.axvline(sub['lfmc_t_1_inside'].mean(),color = 'darkred',linewidth = 2, label = '_nolegend_')
    ax.axvline(sub['lfmc_t_1_outside'].mean(),color = 'darkgreen',linewidth = 2, label = '_nolegend_')
    ax.set_xlabel('LFMC (%)')
    ax.set_title('%s'%lc)
    plt.legend()
    
#%% is the difference in histograms significant?
for lc in df.landcover.unique():
    sub = df.loc[df.landcover==lc,['lfmc_t_1_inside','lfmc_t_1_outside','landcover']]
    U, p = mannwhitneyu(sub['lfmc_t_1_inside'] , sub['lfmc_t_1_outside'], alternative = 'less')
    print("Landcover: %s,\tU = %0.2f,\tp = %0.3f"%(lc,U,p))

#%% comparing lfmc to other climate indices
# for var in ['lfmc','vpd','ppt','erc']:    
#     cols = [col for col in df.columns if var in col]
#     fig, ax = plt.subplots(figsize = (3,3))
#     df[cols[0]].plot.hist(color = 'darkred',alpha = 0.5,ax=ax,label = 'Burned area')
#     df[cols[1]].plot.hist(color = 'lime',alpha = 0.5,ax=ax,label = 'Unaffected area')
#     # ax.axvline(data.mean(),color = 'k',linewidth = 2, label = 'mean')
    
#     ax.set_xlabel('%s %s'%(var.upper(),units[var]))
#     plt.legend()


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
