# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:14:03 2020

@author: kkrao
"""

dir_data = r"D:\Krishna\projects\wildfire_from_lfmc\data\tables"
dir_root = r"D:\Krishna\projects\wildfire_from_lfmc"
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

short_lc = {'enf':'Closed needleleaf\nevergreen',
            'bdf':'Closed broadleaf\ndeciduous',
            'mixed':'Mixed forest',
            'shrub':'Shrubland',
            'grass': 'Grassland'}

color_dict = {'Closed broadleaf\ndeciduous':'darkorange',
              'Closed needleleaf\nevergreen': 'forestgreen',
              'Closed broadleaf deciduous':'darkorange',
              'Closed needleleaf evergreen': 'forestgreen',
              'Mixed forest':'darkslategrey',
              'Shrub/grassland' :'y' ,
              'Shrubland':'tan',
              'Grassland':'lime',
              }  

units = {'lfmc':'(%)','vpd':'(hPa)','erc':'','ppt':r'(mm/month)'}
axis_lims = {'lfmc':[75,125],'vpd':[15,50],'erc':[20,70],'ppt':[0,120]}

lfmc_thresholds = {'Closed broadleaf\ndeciduous':[72,105,125],
              'Closed needleleaf\nevergreen': [72,105,125],
              'Mixed forest':[72,105,125],
              'Shrub/grassland' :[55,67,110] ,
              'Shrubland':[106,121,133],
              'Grassland':[55,67,110],
              } 


# {
#     'grass':[55,67,110],
#     'shrub':[106,121,133],
#     'forest':[72,105,125]
#     }
