# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:30:10 2020

@author: kkrao
"""

import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from init import dir_root, dir_data
import fiona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import mapping
from pandas.tseries.offsets import DateOffset
from sklearn.linear_model import LinearRegression
import sys
import gdal
import matplotlib as mpl
from osgeo import gdal, osr
import seaborn as sns
from sklearn.linear_model import Lasso
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from pingouin import partial_corr
from scipy import stats
from matplotlib.dates import DateFormatter, MonthLocator
import scipy as sc


sns.set(font_scale = 1., style = "ticks")
plt.style.use("pnas")


dfmcDict = {"100hr":2, "1000hr":3}

def get_dates(date, maxLag = 6):
    subsetDates = []
    for delta in range(1, maxLag):
        shiftedDate = pd.to_datetime(date,format = "%Y-%m-%d") - DateOffset(months = delta)
        shiftedDate = shiftedDate.date().strftime(format = "%Y-%m-%d")                    
        for day in [1,15]:
            subsetDates+= [shiftedDate[:-2]+"%02d"%day]
    
    subsetDates = pd.to_datetime(subsetDates,format = "%Y-%m-%d")
    subsetDates = list(subsetDates.sort_values(ascending = False).strftime(date_format = "%Y-%m-%d") )
    
    return subsetDates        
    
def create_time_df(maxLag = 6, folder = "lfmc_vpd_anomalies"):
    date = "2016-01-01"
    filename = os.path.join(dir_root, "data",folder,"lfmc_map_%s.tif"%date)
    ds = gdal.Open(filename)
    array = np.array(ds.GetRasterBand(1).ReadAsArray())
    
    
    x_loc, y_loc = np.meshgrid(range(array.shape[1]),range(array.shape[0]) )
    
    master = pd.DataFrame()
    for date in dates:
        df = pd.DataFrame()
        filename = os.path.join(dir_root, "data",folder,"lfmc_map_%s.tif"%date)
        ds = gdal.Open(filename)
        lfmc = np.array(ds.GetRasterBand(1).ReadAsArray())
        dfmc = np.array(ds.GetRasterBand(2).ReadAsArray())
        # out_image[:,cluster_image[0,:,:]!=zone] = np.nan
        
        df['lfmc(t)'] = lfmc.flatten()
        df['vpd(t)'] = dfmc.flatten()
        df['x_loc'] = x_loc.flatten()
        df['y_loc'] = y_loc.flatten()
        df['pixel_index'] = df.index
        
        df['date'] = date
        ctr = 1
        sys.stdout.write('\r'+'[INFO] Time step %s'%date)
        sys.stdout.flush()
        # print(date)
        subsetDates = get_dates(date, maxLag = maxLag)
        for t in subsetDates:
            shiftedFile = os.path.join(dir_root, "data",folder,"lfmc_map_%s.tif"%t)
            ds = gdal.Open(shiftedFile)
            df['vpd(t-%d)'%ctr] = np.array(ds.GetRasterBand(2).ReadAsArray()).flatten()
            ctr+=1
        df.dropna(inplace = True)
        master = master.append(df,ignore_index = True) 
    master = master.dropna()
    master.to_pickle(os.path.join(dir_root, "data","arr_pixels_%s"%folder,f"arr_pixels_time_wise_lag_{maxLag}_{norm}"))
    return master
    

def pValue(reg, x, y):
    
    n, k = x.shape
    yHat = np.matrix(reg.predict(x)).T

    # Change X and Y into numpy matricies. x also has a column of ones added to it.
    x = np.hstack((np.ones((n,1)),np.matrix(x)))
    y = np.matrix(y).T

    # Degrees of freedom.
    df = float(n-k-1)

    # Sample variance.     
    sse = np.sum(np.square(yHat - y),axis=0)
    sampleVariance = sse/df

    # Sample variance for x.
    sampleVarianceX = x.T*x

    # Covariance Matrix = [(s^2)(X'X)^-1]^0.5. (sqrtm = matrix square root.  ugly)
    covarianceMatrix = sc.linalg.sqrtm(sampleVariance[0,0]*sampleVarianceX.I)

    # Standard erros for the difference coefficients: the diagonal elements of the covariance matrix.
    se = covarianceMatrix.diagonal()[1:]

    # T statistic for each beta.
    betasTStat = np.zeros(len(se))
    for i in range(len(se)):
        betasTStat[i] = reg.coef_[i]/se[i]

    # P-value for each beta. This is a two sided t-test, since the betas can be 
    # positive or negative.
    betasPValue = 1 - sc.stats.t.cdf(abs(betasTStat),df)
    return np.min(betasPValue)
        
def regress(df,norm = "no_norm", coefs_type = "unrestricted"):            
    cols = [col for col in df.columns if "vpd" in col]        
    X = df.loc[:,cols]
    y = df.iloc[:,0] ### 
    if norm=="lfmc_norm":
        y = (y-y.mean())/y.std()
    elif norm=="feature_norm":
        X = (X - X.mean())/X.std()
    elif norm == "both_norm":
        y = (y-y.mean())/(y.std()+1e-5)
        X = (X - X.mean())/(X.std()+1e-5)

    if coefs_type=="positive":
        reg = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random').fit(X,y)
    else:
        reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    coefs = [reg.intercept_]+list(reg.coef_)
    p = pValue(reg, X, y)

    return r2, coefs, df['x_loc'].iloc[0],df['y_loc'].iloc[0], p    


def save_tif(data, geotransform, savepath = None):
    
    nrows, ncols = data.shape
    
    output_raster = gdal.GetDriverByName('GTiff').Create(savepath,ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
    srs = osr.SpatialReference()                 # Establish its coordinate encoding
    srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                                # Anyone know how to specify the 
                                                 # IAU2000:49900 Mars encoding?
    output_raster.SetProjection(srs.ExportToWkt() )   # Exports the coordinate system 
                                                       # to the file
    output_raster.GetRasterBand(1).WriteArray(data)   # Writes my array to the raster
    
    output_raster.FlushCache()
    output_raster = None 
    print("output_saved")
    
    
def anomalize(df): ##useless if regression is done on pixel-scale
    df['lfmc(t)'] = df['lfmc(t)'] -  df['lfmc(t)'].mean()
    
    cols = [col for col in df.columns if "dfmc" in col]
    df[cols] = df[cols] - df[cols].mean()
    
    return df
    

def get_ts(x_locs, y_locs):
    lfmc = pd.DataFrame()
    dfmc = pd.DataFrame()
    for date in dates:
        filename = os.path.join(dir_root, "data",folder,"lfmc_map_%s.tif"%date)
        ds = gdal.Open(filename)
        lfmc = lfmc.append(pd.Series(np.array(ds.GetRasterBand(1).ReadAsArray())[y_locs, x_locs], name = date))
        dfmc = dfmc.append(pd.Series(np.array(ds.GetRasterBand(dfmcDict[hr]).ReadAsArray())[y_locs, x_locs],name = date))
    return lfmc, dfmc
        
        
#%%
years = range(2016, 2021)
months = range(1,13)
days = [1,15]

dates = []
for year in years:
    for month in months:
        for day in days:
            dates+=["%s-%02d-%02d"%(year, month, day)]

dates = dates[12:-11]

folder = "lfmc_vpd_anomalies"
lag = 6
norm = "both_norm"
coefs_type = "positive"

# create_time_df(folder = folder, maxLag = lag)

master = pd.read_pickle(\
        os.path.join(dir_root, "data","arr_pixels_%s"%folder,\
                 f"arr_pixels_time_wise_lag_{lag}_{norm}"))

master.date = pd.to_datetime(master.date)
master = master.loc[(master.date.dt.month>=6)&(master.date.dt.month<=11)]
master = master.groupby("pixel_index").filter(lambda x: len(x) > 25)


# # # # #########
# print('\r')
# print('[INFO] Regressing')
# out = master.groupby('pixel_index').apply(regress,norm = norm, coefs_type = coefs_type)
# out.to_pickle(os.path.join(dir_root, "data","arr_pixels_%s"%folder,"lfmc_vpd_regressed_lag_%d_%s_%s"%(lag,norm,coefs_type)))

out = pd.read_pickle(os.path.join(dir_root, "data","arr_pixels_%s"%folder,"lfmc_vpd_regressed_lag_%d_%s_%s"%(lag,norm, coefs_type)))
  
#%% p value
if len(out.iloc[0])==5:
     p = [x[-1] for x in out]
     print((np.array(p)<0.05).mean())
     fig, ax = plt.subplots(figsize = (3,3))
     ax.hist(p, bins = 50)
     ax.set_xlabel("p value of regression")
     ax.set_ylabel("Pixels")

#%% r2
r2 = [x[0] for x in out]
x_loc = [x[2] for x in out]
y_loc = [x[3] for x in out]


filename = os.path.join(dir_root, "data","mean","vpd_mean.tif")
ds = gdal.Open(filename)
vpd = np.array(ds.GetRasterBand(1).ReadAsArray())


filename = os.path.join(dir_root, "data","mean","ndvi_mean.tif")
ds = gdal.Open(filename)
ndvi = np.array(ds.GetRasterBand(1).ReadAsArray())
geotransform = ds.GetGeoTransform()


#%% variants with coefs
coefs = [x[1] for x in out]
betas = [x[1:] for x in coefs]
coefSum = [np.sum(x[1:]) for x in coefs]
coefMax = [np.max(x[1:]) for x in coefs]
coefRMS = [np.sqrt(np.sum([y**2 for y in x[1:]])) for x in coefs]
coefRMS = np.clip(np.array(coefRMS), a_min = 0, a_max = 0.6)
# coefPositiveSum = [np.sum([y for y in x[1:] if y>=0]) for x in coefs]
coefAbsSum = [np.sum(np.abs(x[1:])) for x in coefs]
coefMean = [np.mean(x[x!=0]) for x in betas]

# coefDiff = [np.sum(x[1:5]) - np.sum(x[8:12]) for x in coefs]
# coefAbsDiff = [np.sum(np.abs(x[1:5])) - np.sum(np.abs(x[8:12])) for x in coefs]

# plantClimate = ndvi.copy()
# plantClimate[:,:] = np.nan
# plantClimate[y_loc, x_loc] = coefAbsSum
# plantClimate = np.clip(plantClimate,np.nanquantile(plantClimate, q = 0.01),np.nanquantile(plantClimate, q = 0.99)) 
# savepath = os.path.join(dir_root, "data","arr_pixels_%s"%folder,"lfmc_dfmc_%s_lag_%d_%s_coefAbsSum.tif"%(hr,lag,norm))
# save_tif(plantClimate, geotransform, savepath)

# plantClimate = ndvi.copy()
# plantClimate[:,:] = np.nan
# plantClimate[y_loc, x_loc] = coefRMS
# plantClimate = np.clip(plantClimate,np.nanquantile(plantClimate, q = 0.01),np.nanquantile(plantClimate, q = 0.99)) 
# savepath = os.path.join(dir_root, "data","arr_pixels_%s"%folder,"lfmc_dfmc_%s_lag_%d_%s_coefRMS.tif"%(hr,lag,norm))
# save_tif(plantClimate, geotransform, savepath)

plantClimate = ndvi.copy()
plantClimate[:,:] = np.nan
plantClimate[y_loc, x_loc] = coefSum
# plantClimate = np.clip(plantClimate,np.nanquantile(plantClimate, q = 0.01),np.nanquantile(plantClimate, q = 0.99)) 

savepath = os.path.join(dir_root, "data","arr_pixels_%s"%folder,"lfmc_vpd_lag_%d_%s_%s_coefSum.tif"%(lag,norm, coefs_type))
# save_tif(plantClimate, geotransform, savepath)


#%%plots

fig, ax = plt.subplots(figsize = (3,3))
ax.hist([x for x in r2 if x>=0],bins = 100, histtype=u'step', color = "darkgoldenrod", linewidth = 2)
ax.set_xlim(0,1.0)
# ax.set_ylim(0,17500)
ax.set_xlabel("$R^2$")
ax.set_ylabel("Frequency")
plt.show()



fig, ax = plt.subplots(figsize = (3,3))
ax.hist(plantClimate.flatten(),bins = 100, histtype=u'step', color = "purple", linewidth = 2)
ax.set_xlim(0.35,2.5)
# ax.set_ylim(0,17500)
ax.set_xlabel(r'$\sum \beta$')
ax.set_ylabel("Frequency")
plt.show()


##############################################################################

# fig, ax = plt.subplots(figsize = (3,3))
# cmap = plt.cm.viridis  # define the colormap
# # extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(cmap.N)]

# # for i in range(50):
# #     cmaplist[155+i] = mpl.colors.to_rgba("orangered")
# # create the new map
# cmap = mpl.colors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)



# # define the bins and normalize

# vmin = np.nanquantile(plantClimate, q = 0.05)
# vmax = np.nanquantile(plantClimate, q = 0.95)

# # vmin = 0
# # vmax = 1
# bounds = np.linspace(vmin, vmax, 20)
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# sc = ax.imshow(plantClimate,vmin = vmin, vmax = vmax, cmap = cmap)

# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# # create a second axes for the colorbar
# ax2 = fig.add_axes([0.95, 0.21, 0.05, 0.58])
# cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
#     spacing='proportional', boundaries=bounds)

# ax2.set_ylabel(r'$\sum \beta$', size=12)

# plt.show()

#################################################################################
## more beautiful map

fname = 'D:\Krishna\projects\wildfire_from_lfmc\data\mean\lfmc_mean.tif'

ds = gdal.Open(fname)
gt = ds.GetGeoTransform()
data = ds.GetRasterBand(1).ReadAsArray().astype(float)

x = np.linspace(start = gt[0],  stop= gt[0]+data.shape[1]*gt[1], num = data.shape[1])    
y = np.linspace(start = gt[3],  stop= gt[3]+data.shape[0]*gt[5], num = data.shape[0])    

x, y = np.meshgrid(x, y)

enlarge = 2
fig, ax = plt.subplots(figsize=(3*enlarge,3*enlarge))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-92,urcrnrlat=53,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                    name='states', drawbounds=True)
patches   = []

for info, shape in zip(m.states_info, m.states):
    patches.append(Polygon(np.array(shape), True) )
ax.add_collection(PatchCollection(patches, facecolor= 'lightgrey', edgecolor='k', linewidths=0.8))

cmap = ListedColormap(sns.color_palette("Set1").as_hex()[:-1])

plot=m.scatter(x,y, zorder = 2, 
                s=2,c=plantClimate,cmap= "viridis" ,linewidth = 0,\
                    marker='s',latlon = True, vmin =0, vmax = 2)

m.readshapefile('D:/Krishna/projects/vwc_from_radar/data/usa_shapefile/west_usa/cb_2017_us_state_500k', 
                    name='states', drawbounds=True, linewidth = 0.5)

ax.axis('off')
cax = fig.add_axes([0.7, 0.45, 0.03, 0.3])
    
# cax.annotate('Plant  (%) \n', xy = (0.,0.94), ha = 'left', va = 'bottom', color = "w")
cb0 = fig.colorbar(plot,ax=ax,cax=cax,ticks = np.linspace(0,2,5))
# cax.set_yticklabels(['<50','100','150','>200']) 


# plt.savefig(os.path.join(dir_figures,'pred_%s.tiff'%date), \
#                                   dpi =300, bbox_inches="tight",transparent = True)
plt.show()



# ###############################################################################
###PWS vs rest
def clean (x, y):
    inds = np.isnan(x)
    x = x[~inds]
    y = y[~inds]
    
    inds = np.isnan(y)
    x = x[~inds]
    y = y[~inds]

    return x,y



fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(vpd, plantClimate, alpha = 0.3, s = 0.001, color = "k")
# ax.set_xlim(0,1)
ax.set_ylim(0,2)
x,y = clean(vpd,plantClimate)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print("R2 for PWS and VPD mean = %0.3f"%r_value**2)
ax.set_xlabel("Mean VPD (hPa)")
ax.set_ylabel(r'Plant-water sensitivity (PWS)')

################################
# filename = os.path.join(dir_root, "data","mean","vpdMax.tif")
# ds = gdal.Open(filename)
# vpd = np.array(ds.GetRasterBand(1).ReadAsArray())

# fig, ax = plt.subplots(figsize = (3,3))
# ax.scatter(vpd, plantClimate, alpha = 0.3, s = 0.001, color = "k")
# # ax.set_xlim(0,1)
# ax.set_ylim(0,2)
# x,y = clean(vpd,plantClimate)
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# print("R2 for PWS and VPD max = %0.3f"%r_value**2)
# ax.set_xlabel("Max VPD (hPa)")
# ax.set_ylabel(r'Plant-water sensitivity (PWS)')



################################
filename = os.path.join(dir_root, "data","mean","vpdStd.tif")
ds = gdal.Open(filename)
vpd = np.array(ds.GetRasterBand(1).ReadAsArray())

fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(vpd, plantClimate, alpha = 0.3, s = 0.001, color = "k")
# ax.set_xlim(0,1)
ax.set_ylim(0,2)
x,y = clean(vpd,plantClimate)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print("R2 for PWS and VPD std = %0.3f"%r_value**2)
ax.set_xlabel("VPD standard deviation (hPa)")
ax.set_ylabel(r'Plant-water sensitivity (PWS)')
    

################################
filename = os.path.join(dir_root, "data","mean","fireSeasonLength.tif")
ds = gdal.Open(filename)
data = np.array(ds.GetRasterBand(1).ReadAsArray())

fig, ax = plt.subplots(figsize = (3,3))
ax.scatter(data, plantClimate, alpha = 0.3, s = 0.001, color = "k")
# ax.set_xlim(0,1)
ax.set_ylim(0,2)
x,y = clean(data,plantClimate)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print("R2 for PWS and fire season length = %0.3f"%r_value**2)
ax.set_xlabel("Dry season length (days)")
ax.set_ylabel(r'Plant-water sensitivity (PWS)')
##fire season length calculated as number of days in a year when VPD > long term mean for that pixel")

###############################################################################

##############################################################################
# fig, ax = plt.subplots(figsize = (3,3))
# lag =1
# lagLabel = 6
# out = pd.read_pickle(os.path.join(dir_root, "data","arr_pixels_%s"%folder,"plant_climate_regressed_normalized_%s_lag_%d"%(hr,lag)))
# r2 = [x[0] for x in out]
# ax.hist([x for x in r2 if x>=0],bins = 100, histtype=u'step', linewidth = 2, label = "%d months"%lagLabel, density = True)
# for lag in range(6, 1, -1):
#     lagLabel = lag-1
#     out = pd.read_pickle(os.path.join(dir_root, "data","arr_pixels_%s"%folder,"plant_climate_regressed_normalized_%s_lag_%d"%(hr,lag)))
#     r2 = [x[0] for x in out]
#     print(np.mean(r2))
#     ax.hist([x for x in r2 if x>=0],bins = 100, histtype=u'step', linewidth = 2, label = "%d months"%lagLabel, density = True)
# ax.set_xlim(0,1.0)
# # ax.set_ylim(0,17500)

# ax.set_xlabel("$R^2$")
# ax.set_ylabel("Density")
# plt.legend()
# plt.show()


# std = master.groupby("pixel_index").apply(lambda df:df['lfmc(t)'].quantile(0.1))
# std = master.groupby("pixel_index").apply(lambda df:df['lfmc(t)'].std())
# np.corrcoef(coefRMS,r2)**2
# fig, ax = plt.subplots(figsize = (3,3))
# ax.scatter(r2, coefRMS, alpha = 0.3, s = 0.001, color = "k")
# ax.scatter(np.mean(r2), np.quantile(coefRMS,0.5),s = 15, color = "magenta")
# # ax.set_xlim(0,1)
# ax.set_ylim(vmin, vmax)
# ax.set_xlabel("$R^2$")
# ax.set_ylabel("CoefRMS")


#%% regions with low and high R2 look like what?


# x_locs = [x[2] for x in out if x[0]<0.1]
# y_locs = [x[3] for x in out if x[0]<0.1]
# lfmc, dfmc = get_ts(x_locs, y_locs)
# lfmc.index = pd.to_datetime(lfmc.index)
# dfmc.index = pd.to_datetime(dfmc.index)


# for index in [34,53,80,140]:
#     fig, ax = plt.subplots(figsize = (3,1))
#     ax2 = ax.twinx()
#     lfmc.loc[:,index].plot(legend = False, label = "lfmc", ax =ax, color = "goldenrod")
#     dfmc.loc[:,index].plot(legend = False,label = "dfmc", ax=ax2,color = "purple")
    
#     ax.set_ylabel("LFMC", color ="goldenrod" )
#     ax2.set_ylabel("DFMC",color = "purple")


# x_locs = [x[2] for x in out if x[0]>0.8]
# y_locs = [x[3] for x in out if x[0]>0.8]
# lfmc, dfmc = get_ts(x_locs, y_locs)
# lfmc.index = pd.to_datetime(lfmc.index)
# dfmc.index = pd.to_datetime(dfmc.index)


# for index in [34,53,80,140]:
#     fig, ax = plt.subplots(figsize = (3,1))
#     ax2 = ax.twinx()
#     lfmc.loc[:,index].plot(legend = False, label = "lfmc", ax =ax, color = "goldenrod")
#     dfmc.loc[:,index].plot(legend = False,label = "dfmc", ax=ax2,color = "purple")
    
#     ax.set_ylabel("LFMC", color ="goldenrod" )
#     ax2.set_ylabel("DFMC",color = "purple")
#     plt.show()

#%% distribution of coefficients
# df = pd.DataFrame(coefs)    
# df.drop(0,axis = 1,inplace = True)
# fig, ax = plt.subplots(figsize = (3,3))

# for col in df.columns:
#     df[col].hist(bins = 1000,histtype = "step",label = col, ax = ax)
# ax.set_xlim(-1,1)
# ax.set_xlabel("Slope stength")
# ax.set_ylabel("Frequency")
# ax.set_title(norm)
# plt.legend(title = "lag",fontsize = 7)
# plt.show()

#%% sample fire-cilamte areas

# gg = plantClimate.copy()
# low = 1.4
# high = 1.8
# gg[np.where(gg>high)] = np.nan
# gg[np.where(gg<low)] = np.nan
# fig, ax = plt.subplots(figsize = (3,3))
# cmap = plt.cm.Greys  # define the colormap

# sc = ax.imshow(gg,vmin = 0, vmax = 2, cmap = cmap)

# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)

# plt.show()

#%% per nonzero soefs

# df = pd.DataFrame(coefs)
# df.drop(0,axis = 1,inplace = True)
# df.columns = df.columns-1

# fig, ax = plt.subplots(figsize = (3,3))
# (df>0).mean().plot(kind = "bar",ax = ax)
# ax.set_xlabel("lag")
# ax.set_ylabel(r"fraction non-zero $\beta$")

#%% PWS computation figures
    

df = pd.DataFrame({"r2": [x[0] for x in out], "x_loc": [x[2] for x in out], "y_loc": [x[3] for x in out]  })
toJoin = pd.DataFrame([x[1] for x in out])
toJoin.drop(0,inplace = True, axis = 1)
toJoin.columns = range(11)
df = pd.concat([df,toJoin], axis = 1)
df = pd.merge(df,master.loc[:,["x_loc","y_loc","pixel_index"]].drop_duplicates(),on = ["x_loc","y_loc"])
df['coefSum'] = df.loc[:,range(11)].sum(axis = 1)
master = master.rename(columns = {"dfmc(t)":"dfmc(t-0)"})


nbins = 2

fig, (ax2, ax1) = plt.subplots(1, 2, figsize = (3.6,1.6), sharey = True)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
inds = df.loc[(df.r2>=0.8),"pixel_index"]
for ind_ in [60]:
    ind = inds.iloc[ind_]
    subMaster = master.loc[master.pixel_index==ind]
    subDf = df.loc[df.pixel_index==ind]
    print(subDf)
    ax1.set_xlim(-5,5)
    for i in subDf.drop(["r2","x_loc","y_loc","pixel_index","coefSum"],axis = 1).iloc[0].nlargest(1).index:
        if subDf[i].values[0]!=0:
            sns.regplot(subMaster["dfmc(t-%d)"%i], subMaster["lfmc(t)"],color = "k",marker ="o" ,\
                scatter_kws ={"s":30,"edgecolor":"grey","color" :"#f2d600","edgecolor":"grey"},\
                    ax = ax1, truncate = False)

            pws =subDf.drop(["r2","x_loc","y_loc","pixel_index","coefSum"],axis = 1).sum(axis =1).iloc[0]
            # ax1.set_title('Sample pixel PWS=%0.2f'%pws, fontsize = 8)
            ax1.annotate('Sample pixel PWS=%0.2f'%pws, fontsize = 9, 
            xy=(0.5, 1.1), ha = "center",textcoords='axes fraction', weight = "bold")
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            # ax.set_title(ind_)  
            ax1.set_ylim(-20,20)
            
            ax1.set_xticks([-5,0,5])

            ax1.set_title("")
    
inds = df.loc[df.r2<=0.05,"pixel_index"]
for ind_ in [6]:
    ind = inds.iloc[ind_]
    subMaster = master.loc[master.pixel_index==ind]
    subDf = df.loc[df.pixel_index==ind]
    print(subDf)
    mctr=0   
    ax2.set_xlim(-5,5)
    for i in subDf.drop(["r2","x_loc","y_loc","pixel_index","coefSum"],axis = 1).iloc[0].nlargest(1).index:
        if subDf[i].values[0]!=0:
            sns.regplot(subMaster["dfmc(t-%d)"%i], subMaster["lfmc(t)"],color = "k",marker="o", \
                scatter_kws ={'s':30,"edgecolor":"grey","color" : "#1565c0","edgecolor":"grey"}, \
                    ax = ax2, truncate = False)
            mctr+=1
    pws =subDf.drop(["r2","x_loc","y_loc","pixel_index","coefSum"],axis = 1).sum(axis =1).iloc[0]
    # ax2.set_title('Sample pixel PWS=%0.2f'%pws)
    ax2.annotate('Sample pixel PWS=%0.2f'%pws, fontsize = 9, 
            xy=(0.5, 1.1), ha = "center",textcoords='axes fraction', weight = "bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("LFMC anomaly")

    ax2.set_title("")
   
    ax2.set_xticks([-5,0,5])
    ax2.set_yticks([-20,-10,0,10,20])
fig.text(x = 0.5 ,y = -0.07, s = r'Climate-derived fuel moisture anomaly', ha ="center",va="top")
# plt.tight_layout()
plt.show()

# lowBin = np.load(os.path.join(dir_root, "data","pws_bins","PWS_bin_1.npy"))
# ys,xs = np.where(lowBin==True)
# ctr=0
# xPoints = []
# yPoints = []
# for (x,y) in zip(xs,ys):
#     ind = inds.iloc[ind_]
#     subMaster = master.loc[(master.x_loc==x)&(master.y_loc==y)]
#     subDf = df.loc[(df.x_loc==x)&(df.y_loc==y)]
    
#     # mctr=0   
#     for i in subDf.drop(["r2","x_loc","y_loc","pixel_index","coefSum"],axis = 1).iloc[0].nlargest(1).index:
#         if subDf[i].values[0]!=0:
#             xPoints = xPoints + list(subMaster["dfmc(t-%d)"%i])
#             yPoints = yPoints + list(subMaster["lfmc(t)"])
#     print(ctr)     
#     ctr+=1
# fig, ax = plt.subplots(figsize = (1.8,1.8))
# sns.regplot(x =np.array(xPoints), y=np.array(yPoints),color = "k",marker="o", ax = ax, \
#             scatter_kws ={'s':0.1,"color" : colors[0], 'alpha':0.003})
#         # mctr+=1
# # pws =subDf.drop(["r2","x_loc","y_loc","pixel_index","coefSum"],axis = 1).sum(axis =1).iloc[0]
# # ax.annotate('PWS=%0.2f'%pws,
#         # xy=(0.95, 0.05), ha = "right",textcoords='axes fraction')
# ax.set_xlabel("Climatic FMC\nAnomaly")
# ax.set_ylabel("LFMC Anomaly")
# ax.set_title(ind_)
# ax.set_ylim(-50,50)
# ax.set_title("")
#     # ax.set_xlim(-5,5)
#     # ax.set_xticks([-5,0,5])
# plt.show()


# highBin = np.load(os.path.join(dir_root, "data","pws_bins","PWS_bin_15.npy"))
# ys,xs = np.where(highBin==True)
# ctr=0
# xPoints = []
# yPoints = []
# for (x,y) in zip(xs,ys):
#     ind = inds.iloc[ind_]
#     subMaster = master.loc[(master.x_loc==x)&(master.y_loc==y)]
#     subDf = df.loc[(df.x_loc==x)&(df.y_loc==y)]
    
#     # mctr=0   
#     for i in subDf.drop(["r2","x_loc","y_loc","pixel_index","coefSum"],axis = 1).iloc[0].nlargest(1).index:
#         if subDf[i].values[0]!=0:
#             xPoints = xPoints + list(subMaster["dfmc(t-%d)"%i])
#             yPoints = yPoints + list(subMaster["lfmc(t)"])
#     print(ctr)     
#     ctr+=1
# fig, ax = plt.subplots(figsize = (1.8,1.8))
# sns.regplot(x =np.array(xPoints), y=np.array(yPoints),color = "k",marker="o", ax = ax, \
#             scatter_kws ={'s':0.01,"color" : colors[1], 'alpha':0.001})
#         # mctr+=1
# # pws =subDf.drop(["r2","x_loc","y_loc","pixel_index","coefSum"],axis = 1).sum(axis =1).iloc[0]
# # ax.annotate('PWS=%0.2f'%pws,
#         # xy=(0.95, 0.05), ha = "right",textcoords='axes fraction')
# ax.set_xlabel("Climatic FMC\nAnomaly")
# ax.set_ylabel("LFMC Anomaly")
# ax.set_title(ind_)
# # ax.set_ylim(-20,20)
# ax.set_title("")
#     # ax.set_xlim(-5,5)
#     # ax.set_xticks([-5,0,5])
# plt.show()

#%%
# first PWS bin = 0.10661708 0.35133948
# Last bin of PWS =1.58814322 2.06

#%% time series of LFMC and DFMC for low and high PWS pixels
sns.set(style = "ticks",font_scale = 1)

##lfmc Raw calculated by running specific lines of create_time df function above

date = "2016-01-01"
filename = os.path.join(dir_root, "data","lfmc_dfmc_raw","lfmc_map_%s.tif"%date)
ds = gdal.Open(filename)
array = np.array(ds.GetRasterBand(1).ReadAsArray())

x_loc, y_loc = np.meshgrid(range(array.shape[1]),range(array.shape[0]) )

lfmcRaw = pd.DataFrame()
for date in dates:
    dff = pd.DataFrame()
    filename = os.path.join(dir_root, "data","lfmc_dfmc_raw","lfmc_map_%s.tif"%date)
    ds = gdal.Open(filename)
    lfmc = np.array(ds.GetRasterBand(1).ReadAsArray())
    
    dff['lfmc(t)'] = lfmc.flatten()
    dff['x_loc'] = x_loc.flatten()
    dff['y_loc'] = y_loc.flatten()
    dff['pixel_index'] = dff.index
    
    dff['date'] = date
    sys.stdout.write('\r'+'[INFO] Time step %s'%date)
    sys.stdout.flush()
    dff.dropna(inplace = True)
    lfmcRaw = lfmcRaw.append(dff,ignore_index = True) 
lfmcRaw = lfmcRaw.dropna()
lfmcRaw = lfmcRaw.rename(columns = {"lfmc(t)":'lfmc_raw'})

highPWS = df.loc[df.coefSum>=1.9,'pixel_index'].values
lowPWS = df.loc[df.coefSum<=0.2,'pixel_index'].values

# for pixel in [328,432]: ##high, low
for pixel in highPWS: ##high, low
    data = master.loc[master.pixel_index==pixel,['date','lfmc(t)','dfmc(t-0)']].copy()
    data.index = data.date

    lfmcRawSubset = lfmcRaw.loc[lfmcRaw.pixel_index==pixel,['date','lfmc_raw']].copy()
    lfmcRawSubset.index = lfmcRawSubset.date
    lfmcRawSubset.drop("date",axis = 1,inplace = True)

    data = data.join(lfmcRawSubset)
    
    fig, ax2 = plt.subplots(1,1,figsize = (5,2.5))
    fig, ax1 = plt.subplots(1,1,figsize = (5,2.5))
    fig, ax3 = plt.subplots(1,1,figsize = (5,2.5))
    
    data['lfmc(t)'].plot(ax=ax1,marker = "o",linewidth = 1,markersize = 5,rot=0)
    data['dfmc(t-0)'].plot(ax=ax2,color = "k",marker = "o",linewidth = 1,markersize = 5,rot=0)
    data['lfmc_raw'].plot(ax=ax3,color = "C1",marker = "o",linewidth = 1,markersize = 5,rot=0)

    # ax2.plot(dfmc)
    ax1.set_ylabel("LFMC\nanomaly (%)")
    ax2.set_ylabel("Climate\nanomaly (%)")
    ax3.set_ylabel("LFMC (%)")
    
    ax1.set_ylim(-50,60)
    ax3.set_ylim(0,210)
    ax2.set_ylim(-10,10)

    ax1.axhline(color = "grey",linewidth = 0.5)
    ax2.axhline(color = "grey",linewidth = 0.5)
    
    # Hide the right and top spines
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    # Hide the right and top spines
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    
    ax1.set_xlabel("")
    ax2.set_xlabel("")
    ax3.set_xlabel("")
    
    date_form = DateFormatter("%b\n%Y")
    ax1.xaxis.set_major_formatter(date_form)
    ax1.xaxis.set_major_locator(MonthLocator(interval=6))
    
    ax2.xaxis.set_major_formatter(date_form)
    ax2.xaxis.set_major_locator(MonthLocator(interval=6))
    
    ax3.xaxis.set_major_formatter(date_form)
    ax3.xaxis.set_major_locator(MonthLocator(interval=6))
    ax2.set_title(pixel)

    plt.minorticks_off()

    plt.show()
    
