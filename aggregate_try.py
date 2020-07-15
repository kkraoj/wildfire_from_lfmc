# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:43:35 2020

@author: kkrao
"""

import os
import pandas as pd
from init import dir_data, dir_root

data = pd.read_table(os.path.join(dir_root, "data","traits","TRY","P50.txt"), dtype = {"OrigUnitStr":str, "StdValue":float, "Reference":str},encoding = "latin1")
data.drop("Unnamed: 27", axis = 1, inplace = True)
data.OriglName.unique()
p50s = ["psi (-MPa)","P50 (MPa)", "P50","Water potential at 50% loss of conductivity Psi_50 (MPa)","Mean ?50 (with all) (MPa)","Choat et al. 2012 reported ?50 (MPa)"]
data.loc[data.OriglName.isin(p50s),'OrigValueStr'].astype(float).abs().multiply(-1).hist()
