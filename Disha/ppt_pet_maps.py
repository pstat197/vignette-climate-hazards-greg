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
#%%

#set spatial subset dimensions and specific month
minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.
moi = [12,1,2] # month of interest 1=Jan, 2=Feb, ... 12=Dec

### for JJA
PPTsub = PPT.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),\
                 time=PPT['time.month'].isin([6, 7, 8]))
PETsub = PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
                 date=PET['date.month'].isin([6, 7, 8])).sortby('date')
 
tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should have the same numbers, but out of order')
# %%
subNY,subNX,nyrs = tmpPET.shape
years = 1981 + np.arange(nyrs)

r_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])

for i in range(nvals):
  r_vals[gvals[0][i],gvals[1][i]] = linregress(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                      tmpPET[gvals[0][i],gvals[1][i],:])[2]
# %%
## JJA
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = fig.add_axes([0.05,0.05,0.9,0.9],projection=projection) # set the drawing area
# ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=-1, vmax=1, cmap='tab20')

ax.set_title('Map of Correlation Month = JJA'.format(moi))  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
#fig.tight_layout()
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
plt.savefig('testmap_{:02d}.png',dpi=200)
# %%


## FOR MAM
PPTsub = PPT.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),\
                 time=PPT['time.month'].isin([3, 4, 5]))
PETsub = PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
                 date=PET['date.month'].isin([3, 4, 5])).sortby('date')
 
tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should have the same numbers, but out of order')
# %%
subNY,subNX,nyrs = tmpPET.shape
years = 1981 + np.arange(nyrs)

r_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])

for i in range(nvals):
  r_vals[gvals[0][i],gvals[1][i]] = linregress(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                      tmpPET[gvals[0][i],gvals[1][i],:])[2]
# %%
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = fig.add_axes([0.05,0.05,0.9,0.9],projection=projection) # set the drawing area
# ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=-1, vmax=1, cmap='tab20')

ax.set_title('Map of Correlation Month = MAM'.format(moi))  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
#fig.tight_layout()
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
plt.savefig('testmap_{:02d}.png',dpi=200)

#%%
## FOR DJF
PPTsub = PPT.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),\
                 time=PPT['time.month'].isin([12, 1, 2]))
PETsub = PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
                 date=PET['date.month'].isin([12, 1, 2])).sortby('date')
 
tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET[:,:,:-1].shape,'these should have the same numbers, but out of order')
# %%
subNY,subNX,nyrs = tmpPET[:,:,:-1].shape
years = 1981 + np.arange(nyrs)

r_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])

for i in range(nvals):
  r_vals[gvals[0][i],gvals[1][i]] = linregress(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                      tmpPET[:,:,:-1][gvals[0][i],gvals[1][i],:])[2]
# %%
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = fig.add_axes([0.05,0.05,0.9,0.9],projection=projection) # set the drawing area
# ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=-1, vmax=1, cmap='tab20')

ax.set_title('Map of Correlation Month = DJF'.format(moi))  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
#fig.tight_layout()
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
plt.savefig('testmap_{:02d}.png',dpi=200)

#%%

## for SON
PPTsub = PPT.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),\
                 time=PPT['time.month'].isin([9, 10, 11]))
PETsub = PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
                 date=PET['date.month'].isin([9, 10, 11])).sortby('date')
 
tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should have the same numbers, but out of order')
# %%
subNY,subNX,nyrs = tmpPET.shape
years = 1981 + np.arange(nyrs)

r_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])

for i in range(nvals):
  r_vals[gvals[0][i],gvals[1][i]] = linregress(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                      tmpPET[gvals[0][i],gvals[1][i],:])[2]
# %%
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = fig.add_axes([0.05,0.05,0.9,0.9],projection=projection) # set the drawing area
# ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=-1, vmax=1, cmap='tab20')

ax.set_title('Map of Correlation Month = SON')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
#fig.tight_layout()
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
plt.savefig('testmap_{:02d}.png',dpi=200)
# %%
