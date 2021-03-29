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
from sklearn.linear_model import LogisticRegression, LinearRegression

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter




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

def addMeanVpd(df):
    cols = [col for col in df.columns if 'vpd' in col]# select vpd
    df['aridityMean'] = df[cols].mean(axis =1)
    
    return df

#%% load and combine
# df = pd.read_csv(os.path.join(dir_data, "ecoregions_fire_2001_2019_no_geo.csv"))
# dfr = pd.read_csv(os.path.join(dir_data, "ecoregions_plantClimate.csv"))
# df = df.join(dfr['plantClimateSensitivity'])
# dfr = pd.read_csv(os.path.join(dir_data, "ecoregions_fire_vpd_ndvi_2001_2019_no_geo.csv"))
# cols = [col for col in dfr.columns if 'vpd' in col]# select vpd
# cols = cols + ['ndviMean']
# dfr = dfr[cols]
# df = df.join(dfr[cols])
df= pd.read_csv(os.path.join(dir_data, "arr_ecoregions_fire_climate_plant.csv"))
df = addMeanVpd(df)
df.shape
df.head()

def fit_reg(df):
    
    regr = LinearRegression()
    
    # Train the model using the training sets
    X = df['aridity'].values.reshape(-1, 1)
    y = df['logba'].values
    regr.fit(X,y)    
    xline = np.linspace(df['aridity'].min(), df['aridity'].max()).reshape(-1, 1)    
    # Make predictions using the testing set
    yline = regr.predict(xline)
    
    slope = regr.coef_[0]
    
    score = regr.score(X, y)
    # print(slope)
    
    return xline, yline, slope, score
    

years = range(2001,2020)
var = "vpd"

make_ba_vpd_plot = False
minslope = np.inf
maxslope = -np.inf
maxscore = -np.inf
minscore = np.inf

ctr = 0

master = pd.DataFrame()
for index, row in df.iterrows():
    x = row[[col for col in row.index if var in col]]
    y = row[[col for col in row.index if "ba" in col]]*0.25 ## because actual number are just 500m wide pixels
    
    sub = pd.DataFrame({'aridity':x.values,'ba':y.values}, dtype = float)    
    sub = sub.loc[sub.ba>0]
    
    if sub.shape[0]<10:
        continue
    ctr+=1
    sub['logba'] = np.log10(sub.ba)
    xline, yline, slope, score = fit_reg(sub)
    

    if make_ba_vpd_plot:
        
        fig, ax = plt.subplots(figsize = (3,3))
        
        ax.scatter(x,y,40,alpha = 0.5)
                
        ax.plot(xline, 10**yline, 'k')
    
        ax.set_xlabel('%s %s'%(var.upper(),units[var]))
        ax.set_ylabel('Burned area (km$^{2}$)')
        plt.yscale('log')
        
        ax.annotate(r"$\beta$ = %0.2f"%slope,
                        xy=(0.05,0.95), 
                        xycoords="axes fraction",
                        ha='left', va='top', size = 18)
        ax.get_yaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x))))
        
        plt.show()
    
    appendRow = pd.Series(index = ['slope', 'score','plantClimateR2','plantClimateCoefSum','plantClimateCoefDiff','ndvi','vpd'], 
                          data = [slope, score, row['plantClimateR2'], row['plantClimateCoefSum'], row['plantClimateCoefDiff'], row['ndviMean'], row['aridityMean']])
    master = master.append(appendRow, ignore_index = True)


master.head()
master.shape
# master = master.loc[master.ndvi>=0.4]

# fig, ax = plt.subplots(figsize = (3,3))
# plot = ax.scatter(master.sigma, master.score, c = master.ndvi,vmin = 0.2, vmax = 0.8)
# sns.regplot(x = 'sigma', y = 'score', data = master, ax = ax, scatter = False)
# plt.colorbar(plot)
# ax.set_xlabel(r"$\frac{dLFMC^\prime}{dVPD^\prime_6}$")
# # ax.set_ylabel(r"$\frac{dlog(BA)}{dVPD}$")
# ax.set_ylabel(r"$R^2(log(BA),VPD$")

# ax.set_xlim(0,1)
# ax.set_ylim(bottom = -0.05)
# plt.show()

# fig, ax = plt.subplots(figsize = (3,3))
# plot = ax.scatter(master.sigma, master.slope, c = master.ndvi,vmin = 0.2, vmax = 0.8)
# sns.regplot(x = 'sigma', y = 'slope', data = master, ax = ax, scatter = False)
# ax.set_xlabel(r"$\frac{dLFMC^\prime}{dVPD^\prime_6}$")
# ax.set_ylabel(r"$\frac{dlog(BA)}{dVPD}$")
# # ax.set_ylabel(r"$R^2(log(BA),VPD$")
# plt.colorbar(plot)

# ax.set_xlim(0,1)
# # ax.set_ylim(bottom = -0.05)
# plt.show()


# fig, ax = plt.subplots(figsize = (3,3))

# sns.regplot(x = 'ndvi', y = 'slope', data = master, ax = ax)
# ax.set_xlabel(r"Mean NDVI")
# ax.set_ylabel(r"$\frac{dlog(BA)}{dVPD}$")
# # ax.set_ylabel(r"$R^2(log(BA),VPD$")
# # ax.set_xlim(0,1)
# # ax.set_ylim(bottom = -0.05)
# plt.show()

fig, ax = plt.subplots(figsize = (3,3))
master['plantClimateR2'].hist(ax = ax, bins = 40)
ax.set_ylabel(r"Ecoregions")
ax.set_xlabel(r"$R^2(LFMC',ARR(VPD'))$")
plt.show()

fig, ax = plt.subplots(figsize = (3,3))
sns.regplot(x = 'vpd', y = 'ndvi', data = master, ax = ax)
ax.set_xlabel(r"VPD")
ax.set_ylabel(r"NDVI")
plt.show()
master.corr()

fig, ax = plt.subplots(figsize = (3,3))
sns.regplot(x = 'vpd', y = 'slope', data = master, ax = ax)
ax.set_xlabel(r"Mean VPD")
ax.set_ylabel(r"$\frac{dlog(BA)}{dVPD}$")
plt.show()

fig, ax = plt.subplots(figsize = (3,3))
sns.regplot(x = 'vpd', y = 'plantClimateCoefDiff', data = master, ax = ax)
ax.set_xlabel(r"Mean VPD")
ax.set_ylabel(r"$\sum\beta_{[0,-2]} - \sum\beta_{[-4,-6]}$")

fig, ax = plt.subplots(figsize = (3,3))
sns.regplot(x = 'ndvi', y = 'slope', data = master, ax = ax,color = "green")
ax.set_xlabel(r"NDVI")
ax.set_ylabel(r"$\frac{dlog(BA)}{dVPD}$")
plt.show()

fig, ax = plt.subplots(figsize = (3,3))
sns.regplot(x = 'ndvi', y = 'plantClimateCoefDiff', data = master, ax = ax,color = "green")
ax.set_xlabel(r"NDVI")
ax.set_ylabel(r"$\sum\beta_{[0,-2]} - \sum\beta_{[-4,-6]}$")


fig, ax = plt.subplots(figsize = (3,3))
plot = ax.scatter(master.plantClimateR2, master.score, c = master.ndvi,vmin = 0.2, vmax = 0.8)
sns.regplot(x = 'plantClimateR2', y = 'score', data = master, ax = ax, scatter = False)
plt.colorbar(plot)
ax.set_xlabel(r"$R^2(LFMC',ARR(VPD'))$")
ax.set_ylabel(r"$R^2(log(BA),VPD$")

# ax.set_xlim(-0.1,1)
ax.set_ylim(bottom = -0.05)
plt.show()

fig, ax = plt.subplots(figsize = (3,3))
plot = ax.scatter(master.plantClimateR2, master.slope, c = master.ndvi,vmin = 0.2, vmax = 0.8)
sns.regplot(x = 'plantClimateR2', y = 'slope', data = master, ax = ax, scatter = False)
ax.set_xlabel(r"$R^2(LFMC',ARR(VPD'))$")
ax.set_ylabel(r"$\frac{dlog(BA)}{dVPD}$")
plt.colorbar(plot)

# ax.set_xlim(-0.1,1)
# ax.set_ylim(bottom = -0.05)
plt.show()

fig, ax = plt.subplots(figsize = (3,3))
plot = ax.scatter(master.plantClimateCoefSum, master.score, c = master.ndvi,vmin = 0.2, vmax = 0.8)
sns.regplot(x = 'plantClimateCoefSum', y = 'score', data = master, ax = ax, scatter = False)
plt.colorbar(plot)
ax.set_xlabel(r"$\sum(\beta)$")
ax.set_ylabel(r"$R^2(log(BA),VPD$")

fig, ax = plt.subplots(figsize = (3,3))
plot = ax.scatter(master.plantClimateCoefSum, master.slope, c = master.ndvi,vmin = 0.2, vmax = 0.8)
sns.regplot(x = 'plantClimateCoefSum', y = 'slope', data = master, ax = ax, scatter = False)
plt.colorbar(plot)
ax.set_xlabel(r"$\sum(\beta)$")
ax.set_ylabel(r"$\frac{dlog(BA)}{dVPD}$")




fig, ax = plt.subplots(figsize = (3,3))
plot = ax.scatter(master.plantClimateCoefDiff, master.score, c = master.ndvi,vmin = 0.2, vmax = 0.8)
sns.regplot(x = 'plantClimateCoefDiff', y = 'score', data = master, ax = ax, scatter = False)
plt.colorbar(plot)
ax.set_xlabel(r"$\sum\beta_{[0,-2]} - \sum\beta_{[-4,-6]}$")
ax.set_ylabel(r"$R^2(log(BA),VPD$")

fig, ax = plt.subplots(figsize = (3,3))
plot = ax.scatter(master.plantClimateCoefDiff, master.slope, c = master.ndvi,vmin = 0.2, vmax = 0.8)
sns.regplot(x = 'plantClimateCoefDiff', y = 'slope', data = master, ax = ax, scatter = False)
plt.colorbar(plot)
ax.set_xlabel(r"$\sum\beta_{[0,-2]} - \sum\beta_{[-4,-6]}$")
ax.set_ylabel(r"$\frac{dlog(BA)}{dVPD}$")

r2 = master.corr().loc['plantClimateCoefDiff','slope']**2
print("[INFO] R2 of relationship = %0.2f"%r2)




# ax.set_xlim(-0.1,1)
# ax.set_ylim(bottom = -0.05)
plt.show()




print("[INFO] Max slope = %0.2f"%maxslope)
print("[INFO] min slope = %0.2f"%minslope)
print("[INFO] Max R2 = %0.2f"%maxscore)
print("[INFO] min R2 = %0.2f"%minscore)



#%% ####################################################################
### load and cleanuip 
# df = pd.read_csv(os.path.join(dir_data, "ecoregions_annual_aridity.csv"))
# variables = ['lfmc','vpd','ppt','erc']
# years = range(2016,2020)
# cmap = plt.get_cmap('viridis', np.max(years)-np.min(years)+1)


# l1names = ['MEDITERRANEAN CALIFORNIA','SOUTHERN SEMI-ARID HIGHLANDS','TROPICAL WET FORESTS','NORTH AMERICAN DESERTS','GREAT PLAINS','EASTERN TEMPERATE FORESTS','MARINE WEST COAST FOREST','NORTHWESTERN FORESTED MOUNTAINS','NORTHERN FORESTS','TEMPERATE SIERRAS']
# len(l1names)    
    
# df = df.loc[df.na_l1name.isin(l1names)]

# cols = []
# for var in variables + ['ba']:
#     for year in years:
#         cols.append('%s_%s'%(var,year))

# cols.append('us_l3codenum')
# df = df[cols].astype(float)

# for year in years:
#     df = df.loc[~df['lfmc_%s'%year].isnull()]
    

# for var in variables:
#     fig, ax = plt.subplots(figsize = (4,4))
#     for year in years:
#         x = df['{var}_{year:4d}'.format(var = var, year = year)]
#         y =  df['ba_{year:4d}'.format(year = year)]
#         plot = ax.scatter(x,y,40,c = np.repeat(year,df.shape[0]),cmap = cmap,vmin = 2016, vmax = 2019,alpha = 1)
#         # ax.errorbar(x.mean(),y.mean(),xerr = x.std())
    
    
#     x = [col for col in df.columns if var in col]
#     x = df[x].melt()['value']
    
#     y = [col for col in df.columns if 'ba' in col]
#     y = df[y].melt()['value']
    
#     sub = pd.DataFrame({'aridity':x,'ba':y})    
#     sub = sub.loc[sub.ba>0]
    
#     sns.regplot('aridity','ba',data = sub,ax = ax,scatter = False,color = 'k')
    
#     ax.set_xlabel('%s %s'%(var.upper(),units[var]))
#     ax.set_ylabel('Burned area (km$^{2}$)')
#     plt.yscale('log')
#     ax.set_ylim(bottom = 100)
#     ax.set_xlim(axis_lims[var][0],axis_lims[var][1])

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
    
#     discrete_matshow(plot, years,cax)
#     # fig.colorbar(plot, cax=cax, orientation='vertical',ticks = years,format='%1i')


    
#     plt.show()

#%% entire western USA

# for var in variables:
#     fig, ax = plt.subplots(figsize = (4,4))
#     for year in years:
#         x = df['{var}_{year:4d}'.format(var = var, year = year)]
#         y =  df['ba_{year:4d}'.format(year = year)]
#         plot = ax.scatter(x.mean(),y.sum(),80,c = [year],cmap = cmap,vmin = 2016, vmax = 2019,alpha = 1)
#         ax.errorbar(x.mean(),y.sum(),xerr = x.std(),color = 'grey')
        
#     x = [col for col in df.columns if var in col]
#     x = df[x].mean()
    
#     y = [col for col in df.columns if 'ba' in col]
#     y = df[y].sum()
    
#     sub = pd.DataFrame({'aridity':x,'ba':y})    
#     # sub = sub.loc[sub.ba>0]
    
#     sns.regplot('aridity','ba',data = sub,ax = ax,scatter = False,color = 'k')
    
#     ax.set_xlabel('%s %s'%(var.upper(),units[var]))
#     ax.set_ylabel('Burned area (km$^{2}$)')
#     plt.yscale('log')
#     # ax.set_ylim(bottom = 100)
#     ax.set_xlim(axis_lims[var][0],axis_lims[var][1])

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
    
#     discrete_matshow(plot, years,cax)

#%% ecoregions anomalies

# for var in variables:
#     cols = [col for col in df.columns if var in col]
#     df[cols] = df[cols] - df[cols].mean()
    

# for var in variables:
#     fig, ax = plt.subplots(figsize = (4,4))
#     for year in years:
#         x = df['{var}_{year:4d}'.format(var = var, year = year)]
#         y =  df['ba_{year:4d}'.format(year = year)]
#         plot = ax.scatter(x,y,40,c = np.repeat(year,df.shape[0]),cmap = cmap,vmin = 2016, vmax = 2019,alpha = 1)
#         # ax.errorbar(x.mean(),y.mean(),xerr = x.std())
    
    
#     x = [col for col in df.columns if var in col]
#     x = df[x].melt()['value']
    
#     y = [col for col in df.columns if 'ba' in col]
#     y = df[y].melt()['value']
    
#     sub = pd.DataFrame({'aridity':x,'ba':y})    
#     sub = sub.loc[sub.ba>0]
    
#     sns.regplot('aridity','ba',data = sub,ax = ax,scatter = False,color = 'k')
    
#     ax.set_xlabel('%s %s'%(var.upper(),units[var]))
#     ax.set_ylabel('Burned area (km$^{2}$)')
#     plt.yscale('log')
#     ax.set_ylim(bottom = 100)
#     # ax.set_xlim(axis_lims[var][0],axis_lims[var][1])

#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
    
#     discrete_matshow(plot, years,cax)