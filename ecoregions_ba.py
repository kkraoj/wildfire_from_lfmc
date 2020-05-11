# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:08:15 2020

@author: kkrao
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl



dir_data = r"D:\Krishna\projects\wildfire_from_lfmc\data\tables"
os.chdir(dir_data)
sns.set(style='ticks',font_scale = 1.5)
units = {'lfmc':'(%)','vpd':'(hPa)','erc':'','ppt':r'(mm/month)'}
axis_lims = {'lfmc':[75,125],'vpd':[15,50],'erc':[20,70],'ppt':[0,120]}
#%% cmap utility

cmap = plt.cm.viridis  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

def discrete_matshow(plot, data,cax):
    #get discrete colormap
    
    #tell the colorbar to tick at integers
    fig.colorbar(plot, cax= cax,ticks=np.arange(np.min(data),np.max(data)+1))


#%% ####################################################################
# load and cleanuip 
df = pd.read_csv(os.path.join(dir_data, "ecoregions_annual_aridity.csv"))
variables = ['lfmc','vpd','ppt','erc']
years = range(2016,2020)
cmap = plt.get_cmap('viridis', np.max(years)-np.min(years)+1)


l1names = ['MEDITERRANEAN CALIFORNIA','SOUTHERN SEMI-ARID HIGHLANDS','TROPICAL WET FORESTS','NORTH AMERICAN DESERTS','GREAT PLAINS','EASTERN TEMPERATE FORESTS','MARINE WEST COAST FOREST','NORTHWESTERN FORESTED MOUNTAINS','NORTHERN FORESTS','TEMPERATE SIERRAS']
len(l1names)    
    
df = df.loc[df.na_l1name.isin(l1names)]

cols = []
for var in variables + ['ba']:
    for year in years:
        cols.append('%s_%s'%(var,year))

cols.append('us_l3codenum')
df = df[cols].astype(float)

for year in years:
    df = df.loc[~df['lfmc_%s'%year].isnull()]
    

for var in variables:
    fig, ax = plt.subplots(figsize = (4,4))
    for year in years:
        x = df['{var}_{year:4d}'.format(var = var, year = year)]
        y =  df['ba_{year:4d}'.format(year = year)]
        plot = ax.scatter(x,y,40,c = np.repeat(year,df.shape[0]),cmap = cmap,vmin = 2016, vmax = 2019,alpha = 1)
        # ax.errorbar(x.mean(),y.mean(),xerr = x.std())
    
    
    x = [col for col in df.columns if var in col]
    x = df[x].melt()['value']
    
    y = [col for col in df.columns if 'ba' in col]
    y = df[y].melt()['value']
    
    sub = pd.DataFrame({'aridity':x,'ba':y})    
    sub = sub.loc[sub.ba>0]
    
    sns.regplot('aridity','ba',data = sub,ax = ax,scatter = False,color = 'k')
    
    ax.set_xlabel('%s %s'%(var.upper(),units[var]))
    ax.set_ylabel('Burned area (km$^{2}$)')
    plt.yscale('log')
    ax.set_ylim(bottom = 100)
    ax.set_xlim(axis_lims[var][0],axis_lims[var][1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    discrete_matshow(plot, years,cax)
    # fig.colorbar(plot, cax=cax, orientation='vertical',ticks = years,format='%1i')


    
    plt.show()

#%% entire western USA

for var in variables:
    fig, ax = plt.subplots(figsize = (4,4))
    for year in years:
        x = df['{var}_{year:4d}'.format(var = var, year = year)]
        y =  df['ba_{year:4d}'.format(year = year)]
        plot = ax.scatter(x.mean(),y.sum(),80,c = [year],cmap = cmap,vmin = 2016, vmax = 2019,alpha = 1)
        ax.errorbar(x.mean(),y.sum(),xerr = x.std(),color = 'grey')
        
    x = [col for col in df.columns if var in col]
    x = df[x].mean()
    
    y = [col for col in df.columns if 'ba' in col]
    y = df[y].sum()
    
    sub = pd.DataFrame({'aridity':x,'ba':y})    
    # sub = sub.loc[sub.ba>0]
    
    sns.regplot('aridity','ba',data = sub,ax = ax,scatter = False,color = 'k')
    
    ax.set_xlabel('%s %s'%(var.upper(),units[var]))
    ax.set_ylabel('Burned area (km$^{2}$)')
    plt.yscale('log')
    # ax.set_ylim(bottom = 100)
    ax.set_xlim(axis_lims[var][0],axis_lims[var][1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    discrete_matshow(plot, years,cax)

#%% ecoregions anomalies

for var in variables:
    cols = [col for col in df.columns if var in col]
    df[cols] = df[cols] - df[cols].mean()
    

for var in variables:
    fig, ax = plt.subplots(figsize = (4,4))
    for year in years:
        x = df['{var}_{year:4d}'.format(var = var, year = year)]
        y =  df['ba_{year:4d}'.format(year = year)]
        plot = ax.scatter(x,y,40,c = np.repeat(year,df.shape[0]),cmap = cmap,vmin = 2016, vmax = 2019,alpha = 1)
        # ax.errorbar(x.mean(),y.mean(),xerr = x.std())
    
    
    x = [col for col in df.columns if var in col]
    x = df[x].melt()['value']
    
    y = [col for col in df.columns if 'ba' in col]
    y = df[y].melt()['value']
    
    sub = pd.DataFrame({'aridity':x,'ba':y})    
    sub = sub.loc[sub.ba>0]
    
    sns.regplot('aridity','ba',data = sub,ax = ax,scatter = False,color = 'k')
    
    ax.set_xlabel('%s %s'%(var.upper(),units[var]))
    ax.set_ylabel('Burned area (km$^{2}$)')
    plt.yscale('log')
    ax.set_ylim(bottom = 100)
    # ax.set_xlim(axis_lims[var][0],axis_lims[var][1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    discrete_matshow(plot, years,cax)