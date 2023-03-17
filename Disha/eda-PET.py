
#%%
import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import seaborn
import cartopy.feature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import statsmodels.graphics.tsaplots as sm
import statsmodels.tsa.seasonal as sms
from statsmodels.tsa.stattools import adfuller

#opening dataset
file_paths = ['/Users/disha/Downloads/PETmonthly_01.nc','/Users/disha/Downloads/PETmonthly_02.nc',
              '/Users/disha/Downloads/PETmonthly_03.nc','/Users/disha/Downloads/PETmonthly_04.nc',
              '/Users/disha/Downloads/PETmonthly_05.nc','/Users/disha/Downloads/PETmonthly_06.nc',
              '/Users/disha/Downloads/PETmonthly_07.nc','/Users/disha/Downloads/PETmonthly_08.nc',
              '/Users/disha/Downloads/PETmonthly_09.nc','/Users/disha/Downloads/PETmonthly_10.nc',
              '/Users/disha/Downloads/PETmonthly_11.nc','/Users/disha/Downloads/PETmonthly_12.nc']
PET_monthly = xr.open_mfdataset(file_paths, combine='nested', concat_dim='date')
PET_monthly = PET_monthly.sortby('date')

data_dir = '/Users/disha/downloads/'
infile = 'chirps-v2.0.monthly.nc'

CHIRPS = xr.open_dataset(data_dir+infile)
CHIRPS_monthly = CHIRPS.rename({'time': 'date', 'latitude': 'lats', 'longitude': 'lons'})

#%%
# EAST AFRICA SLICING
minlat = -5; maxlat = 20
minlon = 30; maxlon = 55
EA_PET =  PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
EA_PPT =  CHIRPS_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
EA_PPT.load()
 
#%%
## EAST AFRICA PPT DJF STD
EA_PPT_DJF = EA_PPT.sel(date=(EA_PPT['date.month']==12) | (EA_PPT['date.month']<=2)).precip.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

EA_PPT_DJF_MAP = plt.pcolormesh(EA_PPT_DJF.lons,EA_PPT_DJF.lats,EA_PPT_DJF,\
                        vmin=0, vmax=200, cmap='plasma')
        
ax.set_title('DJF Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(EA_PPT_DJF_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar


# %%
# EAST AFRICA PET DJF STD
EA_PET_DJF = EA_PET.sel(date=(EA_PET['date.month']==12) | (EA_PET['date.month']<=2)).PET.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

EA_PET_DJF_MAP = plt.pcolormesh(EA_PET_DJF.lons,EA_PET_DJF.lats,EA_PET_DJF,\
                        vmin=0, vmax=200, cmap='plasma')
        
ax.set_title('DJF Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(EA_PET_DJF_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
# %%

# PPT OF MAM
EA_PPT_MAM = EA_PPT.sel(date=EA_PPT.date.dt.month.isin([3,4,5])).precip.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

EA_PPT_MAM_MAP = plt.pcolormesh(EA_PPT_MAM.lons,EA_PPT_MAM.lats,EA_PPT_MAM,\
                        vmin=0, vmax=200, cmap='plasma')
        
ax.set_title('MAM Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(EA_PPT_MAM_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar

# %%
# EAST AFRICA PET MAM STD
EA_PET_MAM = EA_PET.sel(date=EA_PET.date.dt.month.isin([3,4,5])).PET.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

EA_PET_MAM_MAP = plt.pcolormesh(EA_PET_MAM.lons,EA_PET_MAM.lats,EA_PET_MAM,\
                        vmin=0, vmax=200, cmap='plasma')
        
ax.set_title('MAM Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(EA_PET_MAM_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
# %%
# PPT JJA
EA_PPT_JJA = EA_PPT.sel(date=EA_PPT.date.dt.month.isin([6,7,8])).precip.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

EA_PPT_JJA_MAP = plt.pcolormesh(EA_PPT_JJA.lons,EA_PPT_JJA.lats,EA_PPT_JJA,\
                        vmin=0, vmax=200, cmap='plasma')
        
ax.set_title('JJA Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(EA_PPT_JJA_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
# %%
# PET JJA
EA_PET_JJA = EA_PET.sel(date=EA_PET.date.dt.month.isin([6,7,8])).PET.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

EA_PET_JJA_MAP = plt.pcolormesh(EA_PET_JJA.lons,EA_PET_JJA.lats,EA_PET_JJA,\
                        vmin=0, vmax=200, cmap='plasma')
        
ax.set_title('JJA Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(EA_PET_JJA_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
# %%
# PPT SON
EA_PPT_SON = EA_PPT.sel(date=EA_PPT.date.dt.month.isin([9,10,11])).precip.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

EA_PPT_SON_MAP = plt.pcolormesh(EA_PPT_SON.lons,EA_PPT_SON.lats,EA_PPT_SON,\
                        vmin=0, vmax=200, cmap='plasma')
        
ax.set_title('SON Rainfall Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(EA_PPT_SON_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
# %%
# PET SON
EA_PET_SON = EA_PET.sel(date=EA_PET.date.dt.month.isin([9,10,11])).PET.std('date')

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

EA_PET_SON_MAP = plt.pcolormesh(EA_PET_SON.lons,EA_PET_SON.lats,EA_PET_SON,\
                        vmin=0, vmax=200, cmap='plasma')
        
ax.set_title('SON Evapotranspiration Standard Deviation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(EA_PET_SON_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
# %%

# covariance of PET and PPT in MAM
PET_MAM = EA_PET.sel(date=EA_PET.date.dt.month.isin([3,4,5])).PET
PPT_MAM = EA_PPT.sel(date=EA_PPT.date.dt.month.isin([3,4,5])).precip
EXY = (PET_MAM * PPT_MAM).mean('date')
EXEY = PET_MAM.mean('date')*PPT_MAM.mean('date')
COV = EXY - EXEY
sdxsdy = PET_MAM.std('date')*PPT_MAM.std('date')
rsquared = (COV/sdxsdy)**2

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.5,0.8],projection=projection)  # set the drawing area within the window

rsquared_MAP = plt.pcolormesh(rsquared.lons,rsquared.lats,rsquared,\
                        vmin=-0, vmax=2, cmap='plasma')
        
ax.set_title('rsquared MAM')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(cartopy.feature.BORDERS)
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(rsquared_MAP,cax=fig.add_axes([0.1,0.09,0.4,0.03]), orientation='horizontal') #add colorbar
# %%
rsquared
# %%
# %%
