import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time
from statistics import mean 
from datetime import date
from scipy.stats import spearmanr, pearsonr, mode, linregress
from functions import Precip_2_SPI, GET_LLD_PARS, GET_SPEI_FROM_D


count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')

#%% read in climatology
# get data from https://data.chc.ucsb.edu/products/CHPclim/netcdf/
data_dir = '/home/chc-data-out/products/CHPclim/netcdf/'
infile = 'chpclim.5050.monthly.nc'

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


minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.
PET =  PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
PPT =  CHIRPS_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
moi = 1 # month of interest 1=Jan, 2=Feb, ... 12=Dec
"""
#I am selecting the January, February, and December of each year. Then, the shift function
#will move all the months over one so that ... Feb 1981 = Jan 1981, Jan 1981 = nan. This way, 
#the mean function will take the mean of december of previous year, January, and Febraury, as opposed 
#to January, Febraury, and December of the same year. 
PPTsub = PPT.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
        date=PPT['date.month'].isin([1, 2, 3, 12])).shift(date=1).resample(date='1Y').mean(dim='date')
#Drop 2023 so the datasets match
PETsub = PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
         date=PET['date.month'].isin([1, 2, 3, 12])).shift(date=1).resample(date='1Y').mean(dim='date').drop('2023-12-31', dim = 'date')

tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should be the same since one of the above functions changes formatting')

subNY,subNX = tmpPET[0,:,:].shape


spei_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])

tic = time.time()
#Use the SPEI function for every coordinate, then take the last value since we're mapping DJF of the 2021-2022 year 
for i in range(nvals):
  SPEI_arr =GET_SPEI_FROM_D((tmpPPT[:,gvals[0][i],gvals[1][i]]-tmpPET[:,gvals[0][i],gvals[1][i]]))
  spei_vals[gvals[0][i],gvals[1][i]] = SPEI_arr[-1]
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))  
  
# quick map of correlations
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,spei_vals,\
                        vmin=-2, vmax=2, cmap='BrBG')

ax.set_title('Map of SPEI')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorb]

#####################################################################################################################
spi_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])

tic = time.time()
for i in range(nvals):
  SPI_arr =Precip_2_SPI((tmpPPT[:,gvals[0][i],gvals[1][i]]))
  spi_vals[gvals[0][i],gvals[1][i]] = SPI_arr[-1]
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))  
  
# quick map of correlations
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,spi_vals,\
                        vmin=-2, vmax=2, cmap='BrBG')

ax.set_title('Map of SPI')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar


"""

#I am selecting the January, February, and December of each year. Then, the shift function
#will move all the months over one so that ... Feb 1981 = Jan 1981, Jan 1981 = nan. This way, 
#the mean function will take the mean of december of previous year, January, and Febraury, as opposed 
#to January, Febraury, and December of the same year. 
PPTsub = PPT.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
        date=PPT['date.month'].isin([4, 5, 6, 7])).shift(date=1).resample(date='1Y').mean(dim='date')
#Drop 2023 so the datasets match
PETsub = PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
         date=PET['date.month'].isin([4, 5, 6, 7])).shift(date=1).resample(date='1Y').mean(dim='date')

tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should be the same since one of the above functions changes formatting')

subNY,subNX = tmpPET[0,:,:].shape


spei_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])

tic = time.time()
#Use the SPEI function for every coordinate, then take the last value since we're mapping DJF of the 2021-2022 year 
for i in range(nvals):
  SPEI_arr =GET_SPEI_FROM_D((tmpPPT[:,gvals[0][i],gvals[1][i]]-tmpPET[:,gvals[0][i],gvals[1][i]]))
  spei_vals[gvals[0][i],gvals[1][i]] = SPEI_arr[-1]
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))  
  
# quick map of correlations
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,spei_vals,\
                        vmin=-2, vmax=2, cmap='BrBG')

ax.set_title('Map of SPEI')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorb]

#####################################################################################################################
spi_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])

tic = time.time()
for i in range(nvals):
  SPI_arr =Precip_2_SPI((tmpPPT[:,gvals[0][i],gvals[1][i]]))
  spi_vals[gvals[0][i],gvals[1][i]] = SPI_arr[-1]
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))  
  
# quick map of correlations
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,spi_vals,\
                        vmin=-2, vmax=2, cmap='BrBG')

ax.set_title('Map of SPI')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar



