# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:26:27 2020

@author: kkrao
"""

from scipy import stats
import numpy
import numpy as np
import matplotlib.pyplot as plt


for i in range(100):
    np.random.seed(i)
    
    X1 = np.random.normal(size = 100)
    y1 = X1+0.4*np.random.normal(size = 100)
    print(i)
    # plt.scatter(X1,y1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X1,y1)
    print(slope)
    
    X2 = (X1[range(0,len(X1),2)] + X1[range(1,len(X1),2)])/2
    y2 = (y1[range(0,len(y1),2)] + y1[range(1,len(y1),2)])/2
    
    # plt.scatter(X2,y2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X2,y2)
    print(slope)
    

