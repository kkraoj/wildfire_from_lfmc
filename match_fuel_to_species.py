# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:12:17 2020

@author: kkrao
"""



#! /usr/bin/env python3.5
# defineterm.py
import os
import requests
import sys
import html
import codecs
import pandas as pd
from bs4 import BeautifulSoup
from init import dir_data, dir_root



# searchterm = ' '.join(sys.argv[1:])

df = pd.read_excel(os.path.join(dir_root, "data","traits","TRY","dictionary_fuels_species.xlsx"))

ctr = 0
for index, row in df.iterrows():
    fuel = row.loc["fuel"]
    searchterm = "species name "+fuel
    url = 'https://www.google.com/search?q=' + searchterm
    res = requests.get(url)
    try:
        res.raise_for_status()
    except Exception as exc:
        print('error while loading page occured: ' + str(exc))
    
    text = html.unescape(res.text)
    soup = BeautifulSoup(text, 'lxml')
    # prettytext = soup.prettify()
    
    ## next lines are for analysis (saving raw page), you can comment them
    # frawpage = codecs.open('D:/Krishna/projects/wildfire_from_lfmc/data/traits/TRY/rawpage3.txt', 'w', 'utf-8')
    # frawpage.write(prettytext)
    # frawpage.close()
    
    # firsttag = soup.find('h3', class_="r")
    tags = soup.findAll("div", class_="BNeawe iBp4i AP7Wnd")
    # <div class="BNeawe iBp4i AP7Wnd">
    # BNeawe tAd8D AP7Wnd
    try:
        df.loc[index, 'species'] = tags[1].getText()
        ctr+=1
        print('[INFO] Found.\t Fuel name:%s.'%fuel)
    except:
        print('[INFO] Not found.\t Fuel name:%s.'%fuel)

print('[INFO] Search Complete. Found species name for %d fuels'%ctr)
df.to_excel(os.path.join(dir_root, "data","traits","TRY","dictionary_fuels_species.xlsx"))