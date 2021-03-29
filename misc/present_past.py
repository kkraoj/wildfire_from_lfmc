# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:30:15 2020

@author: kkrao

analyse is there is difference between climate and fuel aridity in the current
year (fire year) to previous year (no fire)s

"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.stats import skew


dir_data = "D:\Krishna\projects\wildfire_from_lfmc\data"
os.chdir(dir_data)
sns.set(style='ticks',font_scale = 1.5)

df = pd.read_csv(os.path.join(dir_data, "tables/fire_collection_present_past.csv"))
df.head()
df.shape
sorted(df.columns)
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

#for var in ['lfmc','erc','vpd','ppt']:    
#    cols = [col for col in df.columns if var in col]
#    fig, ax = plt.subplots()
#    data = df[cols[0]]-df[cols[1]]
#    data.plot.hist(color = 'burlywood',alpha = 0.5,bins = 50,ax=ax,label='_nolegend_')
#    sk = skew(data,nan_policy = 'omit')
##    df[cols[1]].plot.hist(color = 'lime',alpha = 0.5,bins = 50,ax=ax,label = 'Previous year')
#    ax.set_xlabel('{var}$_t$ - {var}$_{{t-2}}$'.format(var=var.upper()))
#    ax.axvline(data.mean(),label = 'mean',color = 'k',linewidth = 2, linestyle = '--')
#    ax.annotate('Skew = {skew:.2f}'.format(skew=sk), xy=(0.05, 0.9), xycoords='axes fraction')
#    plt.legend()


#%%

variables = ['lfmc','erc','vpd','ppt']
cols = [col for col in df.columns if col.split('_')[0] in variables]
sub = df[cols]
master = pd.DataFrame()
for t in [0,1]:
    data = sub[[col for col in sub.columns if col.split('_t_')[1][0]=='%1d'%t]].copy()
    new_cols = [col[0] for col in data.columns.str.split('_')]
    data.columns = new_cols
    data.loc[:,'fire'] = int(not(t))
    master = master.append(data)
master.dropna(inplace = True)
X = master[['lfmc','vpd','ppt','erc']]
y = master.loc[:,'fire'].ravel()
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X)
y_pred = clf.predict_proba(X)[:,0]

clf.score(X, y)
X_test = X.copy()

loss = expit(np.array(X_test) * clf.coef_ + clf.intercept_).ravel()

fig, ax = plt.subplots(figsize = (4,4))

ax.scatter(x = master.loc[master.fire==0,'lfmc'], y = np.repeat(0,len(master.loc[master.fire==0,'lfmc'])),marker = 'o',color = 'green',alpha = 0.01)
ax.scatter(x = master.loc[master.fire==1,'lfmc'], y = np.repeat(1,len(master.loc[master.fire==1,'lfmc'])),marker = 'o',color = 'darkorange',alpha = 0.01)

ax.scatter(X_test.loc[:,'lfmc'], y_pred, color='k', label = 'Prediction',alpha = 0.1,linewidth = 0)
ax.set_xlabel('LFMC(%)')
ax.set_yticks([0,1])
ax.set_yticklabels(['No fire','Fire'])
