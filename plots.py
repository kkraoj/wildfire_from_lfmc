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


dir_data = "D:\Krishna\projects\wildfire_from_lfmc\data"
os.chdir(dir_data)
sns.set(style='ticks',font_scale = 1.5)

#df = pd.DataFrame({'area':[11000,23661,23942,6850],\
#                   'number':[324,466,429,273]},\
#                index = np.arange(2016,2020).astype(int) )
#
#fig, ax = plt.subplots()
#df.plot.bar(y = 'number',ax = ax, rot= 0,legend = False)
#ax.set_ylabel('No. of Fires')
#
#fig, ax = plt.subplots()
#df.plot.bar(y = 'area',ax = ax, rot= 0,legend = False)
#ax.set_ylabel('Area. of Fires (km$^2$)')

#%% ####################################################################
##distribution of fire per landcover type
#df = pd.read_csv(os.path.join(dir_data, "fire_collection.csv"))
#df.head()
#df.landcover.unique()

#%% initialize plot
#lc_dict = { 11: 'crop',
#            14: 'crop',
#            20: 'crop',
#            30: 'crop',
#            40: 'broadleaved evergreen',
#                50: 'Closed broadleaf\ndeciduous',
#            60: 'Broadleaved deciduous',    
#            70: 'Closed needleleaf\nevergreen',
#            90: 'Mixed forest',
#            100:'Mixed forest',
#            110:'Shrub/grassland',
#            120:'Shrub/grassland',
#            130:'Shrubland',
#            140:'Grassland',
#            150:'sparse vegetation',
#            160:'regularly flooded forest',
#            170: 'Broadleaved deciduous',  
#            180: 'Shrub/grassland',
#            190: 'Develped land',
#            200: 'Barren',
#            210: 'Water'}

#lc_dict = { 
#            50: 'Closed broadleaf\ndeciduous',
#            70: 'Closed needleleaf\nevergreen',
#            90: 'Mixed forest',
#            100:'Mixed forest',
#            110:'Shrub/grassland',
#            120:'Shrub/grassland',
#            130:'Shrubland',
#            140:'Grassland',
#            }
#df = df.loc[df.landcover.isin(lc_dict.keys())]
#df = df.groupby('landcover').count()
#df['landcover'] = df.index.map(lc_dict)
#df = df.groupby('landcover').year.sum()
#
#
#
#
#
#fig, ax = plt.subplots()
#df.plot.bar(x = 'landcover',y='area', ax = ax,legend = False,color = 'mediumslateblue')
#ax.set_ylabel('Fires')
#ax.set_xlabel('')
#%% joining fire and lfmc
df = pd.read_csv('fire_collection_with_lfmc.csv')
df.columns
len(df.index.unique())
df.head()

df['fireid'] = str(df['system:index'].str[-21:])
df = df.infer_objects()
len(df['fireid'].unique())
df['lfmcdate'] = df['system:index'].str[9:19]
table = df.pivot_table(index = 'fireid', columns = 'lfmcdate', values = 'mean')
table.join(df[['area','landcover']],on = 'fireid')

sns.set(style='ticks',font_scale = 1)

for index in range(1410,1420):
    fig, ax  = plt.subplots(figsize = (4,1.5))
    series = table.iloc[index]
    series.index = pd.to_datetime(series.index)
    series.plot(ax = ax,color='grey',marker = 'o',ms = 3)
#    ax.plot(series,color='grey',marker = 'o')
    ax.set_ylabel('LFMC(%)')
    ax.set_xlabel('')
#    
    ax.axvline(x= pd.to_datetime(series.name[:10], format = '%Y_%m_%d'),color = 'crimson')
    
yes_fire = np.empty([1,0])
no_fire = np.empty([1,0])
for index, series in table.iterrows():
    series.index = series.infer_objects().index
    fire_date = pd.to_datetime(series.name[:10], format = '%Y_%m_%d')
    
    prev_lfmc_date = '%04d-%02d-01'%(fire_date.year,fire_date.month-1)
    if prev_lfmc_date in series.index:
        yes_fire = np.append(yes_fire, series.loc[prev_lfmc_date])
        no_fire = np.append(no_fire, series.drop(prev_lfmc_date).values)
no_fire = no_fire[~np.isnan(no_fire)]
yes_fire = yes_fire[~np.isnan(yes_fire)]

fig, ax = plt.subplots(figsize = (4,4))

ax.scatter(x = no_fire, y = np.repeat(0,len(no_fire)),marker = 'o',color = 'grey',alpha = 0.1)
ax.scatter(x = yes_fire, y = np.repeat(1,len(yes_fire)),marker = 'o',color = 'crimson',alpha = 0.1)

ax.set_xlabel('LFMC(%)')
ax.set_yticks([0,1])
ax.set_yticklabels(['No fire','Fire'])

#%% logit
X = np.append(no_fire,yes_fire)[np.newaxis].T
y = np.append(np.repeat(0,len(no_fire)),np.repeat(1,len(yes_fire)))
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])

clf.score(X, y)
X_test = np.linspace(0, 250, 300)

loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()

fig, ax = plt.subplots(figsize = (4,4))

ax.scatter(x = no_fire, y = np.repeat(0,len(no_fire)),marker = 'o',color = 'grey',alpha = 0.01)
ax.scatter(x = yes_fire, y = np.repeat(1,len(yes_fire)),marker = 'o',color = 'crimson',alpha = 0.01)

ax.plot(X_test, loss, color='orange', linewidth=3,label = 'Prediction')
ax.set_xlabel('LFMC(%)')
ax.set_yticks([0,1])
ax.set_yticklabels(['No fire','Fire'])
plt.legend()

