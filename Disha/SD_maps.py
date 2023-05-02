
import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import seaborn
import cartopy.feature
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sm
import statsmodels.tsa.seasonal as sms
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import ARIMA
from pmdarima.arima import auto_arima
import time
from datetime import date
from scipy.stats import spearmanr, pearsonr, mode, linregress
import matplotlib as mpl
import matplotlib.pyplot as plt

count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')

#%%

# read in data 
in_dir = '/Users/disha/downloads/'
infile = in_dir + 'sst.mnmean.nc'
SSTS = xr.open_dataset(infile)

PPT_dir = '/Users/disha/downloads/'
infile = 'chirps-v2.0.monthly.nc'
PPT = xr.open_dataset(PPT_dir+infile)


file_paths = ['/Users/disha/Downloads/PETmonthly_01.nc','/Users/disha/Downloads/PETmonthly_02.nc',
              '/Users/disha/Downloads/PETmonthly_03.nc','/Users/disha/Downloads/PETmonthly_04.nc',
              '/Users/disha/Downloads/PETmonthly_05.nc','/Users/disha/Downloads/PETmonthly_06.nc',
              '/Users/disha/Downloads/PETmonthly_07.nc','/Users/disha/Downloads/PETmonthly_08.nc',
              '/Users/disha/Downloads/PETmonthly_09.nc','/Users/disha/Downloads/PETmonthly_10.nc',
              '/Users/disha/Downloads/PETmonthly_11.nc','/Users/disha/Downloads/PETmonthly_12.nc']
PET = xr.open_mfdataset(file_paths, combine='nested', concat_dim='date').sortby('date')

CHIRPS_monthly = PPT.rename({'time': 'date', 'latitude': 'lats', 'longitude': 'lons'})

#%%
## STANDARD DEVIATION GRAPHS OF PPT AND PET OF CENTRAL AND SOUTHERN AFRICA
## DJF

minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.

CS_PET =  PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
CS_PPT =  CHIRPS_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
CS_PPT_DJF = CS_PPT.sel(date=(CS_PPT['date.month']==12) | (CS_PPT['date.month']<=2)).precip.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

CS_PPT_DJF_MAP = plt.pcolormesh(CS_PPT_DJF.lons,CS_PPT_DJF.lats,CS_PPT_DJF,\
                        vmin=0, vmax=200, cmap='tab20')
        
ax.set_title('DJF Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='black') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='grey',zorder=100) #color the oceans
cb = plt.colorbar(CS_PPT_DJF_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar


# %%
CS_PET_DJF = CS_PET.sel(date=(CS_PET['date.month']==12) | (CS_PET['date.month']<=2)).PET.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

CS_PET_DJF_MAP = plt.pcolormesh(CS_PET_DJF.lons,CS_PET_DJF.lats,CS_PET_DJF,\
                        vmin=0, vmax=200, cmap='tab20')
        
ax.set_title('DJF Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='black') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='blue',zorder=100) #color the oceans
cb = plt.colorbar(CS_PET_DJF_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
# %%

CS_PPT_MAM = CS_PPT.sel(date=CS_PPT.date.dt.month.isin([3,4,5])).precip.std('date')
CS_PPT =  CHIRPS_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

CS_PPT_MAM_MAP = plt.pcolormesh(CS_PPT_MAM.lons,CS_PPT_MAM.lats,CS_PPT_MAM,\
                        vmin=0, vmax=200, cmap='tab20')
        
ax.set_title('MAM Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='black') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='grey',zorder=100) #color the oceans
cb = plt.colorbar(CS_PPT_MAM_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar


# %%
CS_PET_MAM = CS_PET.sel(date=CS_PET.date.dt.month.isin([3,4,5])).PET.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

CS_PET_MAM_MAP = plt.pcolormesh(CS_PET_MAM.lons,CS_PET_MAM.lats,CS_PET_MAM,\
                        vmin=0, vmax=200, cmap='tab20')
        
ax.set_title('MAM Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='black') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='grey',zorder=100) #color the oceans
cb = plt.colorbar(CS_PET_MAM_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar

#%%
CS_PPT =  CHIRPS_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
CS_PPT_JJA = CS_PPT.sel(date=CS_PPT.date.dt.month.isin([6,7,8])).precip.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

CS_PPT_JJA_MAP = plt.pcolormesh(CS_PPT_JJA.lons,CS_PPT_JJA.lats,CS_PPT_JJA,\
                        vmin=0, vmax=200, cmap='tab20')
        
ax.set_title('JJA Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='black') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='grey',zorder=100) #color the oceans
cb = plt.colorbar(CS_PPT_JJA_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar


# %%
CS_PET_JJA = CS_PET.sel(date=CS_PET.date.dt.month.isin([6,7,8])).PET.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

CS_PET_JJA_MAP = plt.pcolormesh(CS_PET_JJA.lons,CS_PET_JJA.lats,CS_PET_JJA,\
                        vmin=0, vmax=200, cmap='tab20')
        
ax.set_title('JJA Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='black') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='grey',zorder=100) #color the oceans
cb = plt.colorbar(CS_PET_JJA_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
# %%
CS_PPT_SON = CS_PPT.sel(date=CS_PPT.date.dt.month.isin([9,10,11])).precip.std('date')
CS_PPT =  CHIRPS_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

CS_PPT_SON_MAP = plt.pcolormesh(CS_PPT_SON.lons,CS_PPT_SON.lats,CS_PPT_SON,\
                        vmin=0, vmax=200, cmap='tab20')
        
ax.set_title('SON Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='black') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='grey',zorder=100) #color the oceans
cb = plt.colorbar(CS_PPT_SON_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar


# %%
CS_PET_SON = CS_PET.sel(date=CS_PET.date.dt.month.isin([9,10,11])).PET.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

CS_PET_SON_MAP = plt.pcolormesh(CS_PET_SON.lons,CS_PET_SON.lats,CS_PET_SON,\
                        vmin=0, vmax=200, cmap='tab20')
        
ax.set_title('SON Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='black') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='grey',zorder=100) #color the oceans
cb = plt.colorbar(CS_PET_SON_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
# %%