# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:33:04 2020

@author: kkrao
"""


import pandas as pd
from init import *
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import plot_roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression




sns.set(style='ticks',font_scale = 1.5)
df = pd.read_csv(os.path.join(dir_root,"data","longterm", "lfmc_sample_pixel_seaonality.csv"))
df.rename(columns = {'system:time_start':'time','LFMC':'lfmc'},inplace = True)
df.head()
df.time = pd.to_datetime(df.time)

sns.lineplot('time', 'lfmc',data = df)
