# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:43:35 2020

@author: kkrao
"""

import os
import pandas as pd
from init import dir_data, dir_root
from googlesearch import search


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

species = pd.Series(data.SpeciesName.unique())

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
    
ctr = 0    
for fuel in fuels:
    if fuel.lower() in species.str.lower():
        print('[INFO] Match found.')
        ctr+=1

print(ctr)