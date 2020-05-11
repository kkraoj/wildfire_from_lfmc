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
import statsmodels.api as sm
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


dir_data = r"D:\Krishna\projects\wildfire_from_lfmc\data\tables"
os.chdir(dir_data)
sns.set(style='ticks',font_scale = 1.5)

#%% ####################################################################
#distribution of fire per landcover type
df = pd.read_csv(os.path.join(dir_data, "fire_collection_present_past.csv"))
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
sub = df[['lfmc_t_0_n_15','area','ppt_t_0_n_12','vpd_t_0_n_4','erc_t_0_n_15']].copy()
sub = sub.loc[sub.area>=17]
sub.dropna(inplace = True)
fig, ax = plt.subplots()
ax.plot(sub.lfmc_t_0_n_15,np.log(sub.area),'ko',alpha = 0.5)
ax.set_ylabel('Area (Km$^2$)')
ax.set_xlabel('LFMC$_{t-1}$(%)')


regr = linear_model.LinearRegression()

# Train the model using the training sets
X = sub.drop('area',axis = 1)
y = np.log(sub.area)
# y = sub.area
regr.fit(X,y)
print(regr.score(X,y))
print(regr.coef_)
print(regr.intercept_)
 

xs = np.linspace(0,300)
ys = regr.predict(X.sort_values(by = 'lfmc_t_0_n_15'))
ax.plot(X[['lfmc_t_0_n_15']].sort_values(by = 'lfmc_t_0_n_15'),ys,'-r')

#sm

# X = sm.add_constant(X)
# model = sm.OLS(y,X)
# results = model.fit()
# print(results.summary())


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

#%% scatter plots against all predictors
axis_lims = {'lfmc':[25,200],'vpd':[0,40],'erc':[20,100],'ppt':[0,5]}
units = {'lfmc':'(%)','vpd':'(hPa)','erc':'','ppt':r'(mm/day)'}
for col in X.columns:
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(X[col],np.exp(y),'ko',alpha = 0.5)
    sns.regplot(X[col],np.exp(y),ax = ax,scatter = False)
    ax.set_ylabel('Burned area (Km$^2$)')
    ax.set_xlabel('%s %s'%(col.split('_')[0].upper(),units[col.split('_')[0]]))
    ax.set_xlim(axis_lims[col.split('_')[0]][0],axis_lims[col.split('_')[0]][1])
    ax.set_ylim(bottom = 30)
    plt.yscale('log')
