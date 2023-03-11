# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 21:20:54 2023

@author: foamy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:22:40 2023

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
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import ARIMA
from pmdarima.arima import auto_arima
#opening dataset
file_paths = ['/Users/foamy/Downloads/CHC/PETmonthly_01.nc','/Users/foamy/Downloads/CHC/PETmonthly_02.nc',
              '/Users/foamy/Downloads/CHC/PETmonthly_03.nc','/Users/foamy/Downloads/CHC/PETmonthly_04.nc',
              '/Users/foamy/Downloads/CHC/PETmonthly_05.nc','/Users/foamy/Downloads/CHC/PETmonthly_06.nc',
              '/Users/foamy/Downloads/CHC/PETmonthly_07.nc','/Users/foamy/Downloads/CHC/PETmonthly_08.nc',
              '/Users/foamy/Downloads/CHC/PETmonthly_09.nc','/Users/foamy/Downloads/CHC/PETmonthly_10.nc',
              '/Users/foamy/Downloads/CHC/PETmonthly_11.nc','/Users/foamy/Downloads/CHC/PETmonthly_12.nc']
PET_monthly = xr.open_mfdataset(file_paths, combine='nested', concat_dim='date')
PET_monthly = PET_monthly.sortby('date')

"""
#Map of January Averagem PET
PET_Jan = PET_daily.sel(date=PET_daily['date.month']==1).PET.mean('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

janmap = plt.pcolormesh(PET_Jan.lons,PET_Jan.lats,PET_Jan,\
                        vmin=0, vmax=400, cmap='YlOrRd')
        
ax.set_title('January Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(janmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
"""
minlat = 5.025; maxlat = -17.025
minlon = 12; maxlon = 32.
congo =  PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
congo.load()

congo_ps = congo.mean(dim=["lats", "lons"]).to_pandas() #convert to pandas
"""
#Simple EDA
#Time series
plt.figure(figsize=(10,5))
plt.plot(congo_ps, color='blue',label='Original')
plt.xlabel('Year')
plt.ylabel('PET(mm)')
plt.title('PET in mm')

#histogram
plt.hist(congo_ps)

#Summary statistics
seaborn.boxplot(x = congo_ps.index.month,  #boxplot
                y = congo_ps['PET'])

#Time series with rolling mean and variance
plt.figure(figsize=(20,5))
plt.plot(congo_ps, color='blue',label='Original')
plt.plot(congo_ps.rolling(window=12).mean(), label='12-Months Rolling Mean')
plt.plot(congo_ps.rolling(window=12).std(), label='12-Months Rolling STD')
plt.xlabel('Year')
plt.ylabel('PET(mm)')
plt.title('PET in mm')
"""