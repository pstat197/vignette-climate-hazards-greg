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
data_dir = '/Users/foamy/Downloads/CHC/'
infile = 'chirps-v2.0.monthly.nc'
CHIRPS= xr.open_dataset(data_dir+infile,chunks = "auto")
CHIRPS_monthly = CHIRPS.rename({'time': 'date', 'latitude': 'lats', 'longitude': 'lons'})

"""
#Map of January Average PET
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




minlat = -36; maxlat = 0
minlon = 7.5; maxlon = 50
SA_PET =  PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT =  CHIRPS_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT.load()

#congo_ps = congo.mean(dim=["lats", "lons"]).to_pandas() #convert to pandas



#DJF PET map 
SA_PET_DJF = SA_PET.sel(date=SA_PET['date.month'].isin([12, 1, 2, 3 ])).resample(date='1Y').mean(dim='date').PET.std(dim=('date'))
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

SA_PET_DJF_MAP = plt.pcolormesh(SA_PET_DJF.lons,SA_PET_DJF.lats,SA_PET_DJF,\
                        vmin=0, vmax=200, cmap='turbo')
        
ax.set_title('DJF Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(SA_PET_DJF_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar




    #DJF PPT map 
SA_PPT_DJF = SA_PPT.sel(date=SA_PPT['date.month'].isin([12, 1, 2,3 ])).resample(date='1Y').mean(dim='date').precip.std(dim=('date'))
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

SA_PPT_DJF_MAP = plt.pcolormesh(SA_PPT_DJF.lons,SA_PPT_DJF.lats,SA_PPT_DJF,\
                        vmin=0, vmax=200, cmap='turbo')
        
ax.set_title('DJF Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(SA_PPT_DJF_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
"""
#MAM PET map 
SA_PET_MAM = SA_PET.sel(date=SA_PET['date.month'].isin([3, 4, 5])).PET.std('date')
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

SA_PET_MAM_MAP = plt.pcolormesh(SA_PET_MAM.lons,SA_PET_MAM.lats,SA_PET_MAM,\
                        vmin=0, vmax=200, cmap='turbo')
        
ax.set_title('MAM Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(SA_PET_MAM_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar




#MAM PPT map 
SA_PPT_MAM = SA_PPT.sel(date=SA_PPT['date.month'].isin([3, 4, 5])).precip.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

SA_PPT_MAM_MAP = plt.pcolormesh(SA_PPT_MAM.lons,SA_PPT_MAM.lats,SA_PPT_MAM,\
                        vmin=0, vmax=200, cmap='turbo')
        
ax.set_title('MAM Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(SA_PPT_MAM_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar

#JJA PET map 
SA_PET_JJA = SA_PET.sel(date=SA_PET['date.month'].isin([6, 7, 8])).PET.std('date')
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

SA_PET_JJA_MAP = plt.pcolormesh(SA_PET_JJA.lons,SA_PET_JJA.lats,SA_PET_JJA,\
                        vmin=0, vmax=200, cmap='turbo')
        
ax.set_title('JJA Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(SA_PET_JJA_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar




#JJA PPT map 
SA_PPT_JJA = SA_PPT.sel(date=SA_PPT['date.month'].isin([6, 7, 8])).precip.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

SA_PPT_JJA_MAP = plt.pcolormesh(SA_PPT_JJA.lons,SA_PPT_JJA.lats,SA_PPT_JJA,\
                        vmin=0, vmax=200, cmap='turbo')
        
ax.set_title('JJA Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(SA_PPT_JJA_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar

#SND PET map 
SA_PET_SND = SA_PET.sel(date=SA_PET['date.month'].isin([9, 10, 11])).PET.std('date')
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

SA_PET_SND_MAP = plt.pcolormesh(SA_PET_SND.lons,SA_PET_SND.lats,SA_PET_SND,\
                        vmin=0, vmax=200, cmap='turbo')
        
ax.set_title('SND Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(SA_PET_SND_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar



#SND PPT map 
SA_PPT_SND = SA_PPT.sel(date=SA_PPT['date.month'].isin([9, 10, 11])).precip.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

SA_PPT_SND_MAP = plt.pcolormesh(SA_PPT_SND.lons,SA_PPT_SND.lats,SA_PPT_SND,\
                        vmin=0, vmax=200, cmap='turbo')
        
ax.set_title('SND Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(SA_PPT_SND_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
"""
""" 
# MAKE DATASETS MATCH (not sure how important this is)

SA_PPT_DJF = SA_PPT_DJF.assign_coords(lats=SA_PET_DJF.lats)
SA_PPT_DJF = SA_PPT_DJF.assign_coords(lons=SA_PET_DJF.lons)
SA_PPT_DJF=SA_PPT_DJF.expand_dims(dim="time", axis=0)
SA_PET_DJF=SA_PET_DJF.expand_dims(dim="time", axis=0)



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

