# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:43:35 2020

@author: kkrao
"""

import os
import pandas as pd
import numpy as np
from init import dir_data, dir_root, lc_dict
from googlesearch import search
import gdal
import affine


data = pd.read_table(os.path.join(dir_root, "data","traits","TRY","P50.txt"), dtype = {"OrigUnitStr":str, "StdValue":float, "Reference":str},encoding = "latin1")
data.drop("Unnamed: 27", axis = 1, inplace = True)
data.OriglName.unique()
p50s = ["psi (-MPa)",
        "P50 (MPa)", 
        "P50","Water potential at 50% loss of conductivity Psi_50 (MPa)",
        "Mean ?50 (with all) (MPa)","Choat et al. 2012 reported ?50 (MPa)",
        "Xylem tension at 50% loss of hydraulic conductivity (MPa)"]
# data = data.loc[data.OriglName.isin(p50s),'OrigValueStr'].astype(float).abs().multiply(-1).hist()
data = data.loc[data.OriglName.isin(p50s)]
data.OriglName = data.OriglName.str.lower()

try_species = pd.Series(data.SpeciesName.unique())

def map_fuel_to_species():
    nfmd = pd.read_pickle("D:/Krishna/projects/vwc_from_radar/data/fmc_24_may_2019")
    nfmd.head()
    len(nfmd.fuel.unique())
    fuels = nfmd.fuel.unique()
    
    nfmd.drop_duplicates(subset = ['fuel'], inplace = True)
    nfmd.shape 
    latlon = pd.read_csv("D:/Krishna/projects/vwc_from_radar/data/fuel_moisture/nfmd_queried_latlon.csv")
    latlon.columns = latlon.columns.str.lower()
    nfmd = nfmd.merge(latlon, on = "site", how = "left")
    nfmd.dropna(inplace = True)
    nfmd = nfmd.loc[~nfmd.fuel.isin(['1-Hour','10-Hour','100-Hour', '1000-Hour',\
                        'Duff (DC)', '1-hour','10-hour','100-hour',\
                        '1000-hour', 'Moss, Dead (DMC)' ])]
    nfmd.to_excel(os.path.join(dir_root, "data","traits","TRY","dictionary_fuels_species.xlsx"))
    
    query = "species name Douglas-Fir"
    my_results_list = []
    for i in search(query,        # The query you want to run
                    tld = 'com',  # The top level domain
                    lang = 'en',  # The language
                    num = 2,     # Number of results per page
                    start = 0,    # First result to retrieve
                    stop = 2,  # Last result to retrieve
                    pause = 2.0,  # Lapse between HTTP requests
                   ):
        my_results_list.append(i)
        print(i)
    
def map_genus_to_p50():
    df = pd.read_excel(os.path.join(dir_root, "data","traits","TRY","dictionary_fuels_species.xlsx"))
    df['p50'] = np.nan
    ctr = 0 
    
    for index, row in df.iterrows():
        nfmd_species = row.species.split(' ')[0].lower()
        
        choose = [nfmd_species == tspecies.split(' ')[0].lower() for tspecies in data.SpeciesName]
        p50 = -data.loc[choose, 'OrigValueStr'].astype(float).abs().mean()
        df.loc[index,'p50'] = p50
        
    df.to_excel(os.path.join(dir_root, "data","traits","TRY","dictionary_fuels_species.xlsx"))
def retrieve_pixel_value(geo_coord, data_source):
    """Return floating-point value that corresponds to given point."""
    x, y = geo_coord[0], geo_coord[1]
    forward_transform =  \
        affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    pixel_coord = px, py

    data_array = np.array(data_source.GetRasterBand(5).ReadAsArray())
    return data_array[pixel_coord[0]][pixel_coord[1]]

def map_site_to_lc():
    df = pd.read_excel(os.path.join(dir_root, "data","traits","TRY","dictionary_fuels_species.xlsx"))
    
    # data_source = gdal.Open("D:/Krishna/projects/vod_from_mortality/codes/data/RS_data/Forest/GLOBCOVER/GLOBCOVER_L4_200901_200912_V2.3.tif")    
    data_source = gdal.Open(os.path.join(dir_root, "data","mean","lfmc_vpd_ppt_erc_lc.tif"))
    
    
    df['lc'] = np.nan
    
    # geo_coord = df.loc[:,['longitude', 'latitude']].values
    
    # data_source = gdal.Open(data_source)
    # df.loc[:, "lc"] = retrieve_pixel_value(geo_coord, data_source)
    
    
    for index, row in df.iterrows():
        try:
            df.loc[index, "lc"] = retrieve_pixel_value((row.longitude,row.latitude ), data_source)
        except:
            print("[INFO] Site not found")
    df.to_excel(os.path.join(dir_root, "data","traits","TRY","dictionary_fuels_species.xlsx"))
    return df


def map_lc_to_p50():
    df = pd.read_excel(os.path.join(dir_root, "data","traits","TRY","dictionary_fuels_species.xlsx"))
    
    df = df.loc[df.lc.isin(lc_dict.keys())]
    
    p50 = df.groupby('lc').p50.mean()
    
    return p50
    # return df
    
p50 = map_lc_to_p50()
# df = map_site_to_lc()            
# print(df.head())
# df.lc.isnull().mean()
# df.lc.unique()
    # lc = arr[-1,:,:]
    
        