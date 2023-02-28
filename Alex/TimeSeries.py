# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 05:48:55 2023

@author: foamy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 01:05:29 2023

@author: foamy
"""
import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import seaborn
import cartopy.feature
import matplotlib.pyplot as plt
import pymannkendall as mk
import statsmodels.graphics.tsaplots as sm
import statsmodels.tsa.seasonal as sms

count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
data_dir = '/Users/foamy/Downloads/CHC/'
infile = 'chirps-v2.0.monthly.nc'

CHIRPS = xr.open_dataset(data_dir+infile)
"""
#set spatial subset dimensions
minlat = 16.; maxlat = 21.
minlon = 24; maxlon = 36.
CHsubS =  CHIRPS.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
CHsubS.load()
CHsubS2 = CHsubS.mean(dim=["longitude", "latitude"]).to_pandas()
CHsubS2.plot()
t1= mk.seasonal_test(CHsubS2, period = 12)
"""
minlat2 = 7.; maxlat2 = 10.
minlon2 = -13; maxlon2 = -10.
CHsubSl =  CHIRPS.sel(latitude=slice(minlat2,maxlat2),longitude=slice(minlon2,maxlon2))
CHsubSl.load()
CHsubSl2 = CHsubSl.mean(dim=["longitude", "latitude"]).to_pandas()
CHsubSl2.plot()
t2= mk.seasonal_test(CHsubSl2, period = 12)


#Transformations for Sierra Leone '

#boxplot of data 

seaborn.boxplot(x = CHsubSl2.index.month,
                y = CHsubSl2['precip'])
"""
sm.plot_acf(CHsubSl2, lags=20, alpha=.05) #hist
SL_log = np.log(CHsubSl2) #log transformation

SL_log.plot() #ts of log transformed data 
result=sms.seasonal_decompose(SL_log, model='additive', period=12) 
result.plot()   
"""
