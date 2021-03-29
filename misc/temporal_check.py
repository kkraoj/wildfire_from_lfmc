# -*- coding: utf-8 -*-
"""
Created on Tue May 12 00:12:23 2020

@author: kkrao
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

dir_data = r"D:\Krishna\projects\wildfire_from_lfmc\data\tables"
os.chdir(dir_data)
sns.set(style='ticks',font_scale = 1.1)

#%% ####################################################################
#distribution of fire per landcover type
df = pd.read_csv(os.path.join(dir_data, "ba_vpd_2001_2019.csv"),dtype = float)
df.year = df.year.astype(int)

df["color"] = "slateblue"
df.loc[df.year>=2016,'color'] = 'gold'

df.head()

fig, ax = plt.subplots(figsize = (3,3))
# ax.errorbar(df['mean'],df.ba, fmt = 'o',color = 'grey',xerr = (df.iq1,df.iq3))
ax.scatter(df['mean'], df.ba,c =  df.color, s = 40,edgecolor = 'lightgrey')
plt.yscale("log")
ax.set_xlabel('VPD (hPa)')
ax.set_ylabel('Burned area (km$^2$)')

np.corrcoef(df['mean'],np.log10(df.ba))[0,1]
