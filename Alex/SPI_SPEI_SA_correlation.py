# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:34:29 2023

@author: foamy
"""

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

spei_data = xr.DataArray(dims=["date", "lats","lons"], coords={"date": PPTsub["date"], "lons": PPTsub["lons"],"lats": PPTsub["lats"]})


tic = time.time()
#Use the SPEI function for every coordinate, then take the last value since we're mapping DJF of the 2021-2022 year 
for i in range(len((PPT["lats"].values))):
   for j in range(len((PPT["lons"].values))):
       SPEI_arr =GET_SPEI_FROM_D(tmpPPT[:,i,j]-tmpPET[:,i,j])
       spei_data[:,i,j] = SPEI_arr
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))  

#####################################################################################################################
spi_data = xr.DataArray(dims=["date", "lats","lons"], coords={"date": PPTsub["date"], "lons": PPTsub["lons"],"lats": PPTsub["lats"]})

tic = time.time()
#Use the SPI function for every coordinate, then take the last value since we're mapping DJF of the 2021-2022 year 
for i in range(len((PPT["lats"].values))):
   for j in range(len((PPT["lons"].values))):
       SPI_arr =Precip_2_SPI(tmpPPT[:,i,j])
       spi_data[:,i,j] = SPI_arr
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))  

#####################################################################################################################
tmpSPI = spi_data.values
tmpSPEI = spei_data.values

print(tmpSPI.shape,tmpSPEI.shape,'these should have the same numbers')

subNY,subNX = tmpSPI[0,:,:].shape

r_vals = np.zeros([subNY,subNX]) * np.nan

tic = time.time()

for i in range(len((PPT["lats"].values))):
   for j in range(len((PPT["lons"].values))):
       r_vals[i,j] = linregress(tmpSPI[:,i,j],tmpSPEI[:,i,j])[2]
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))    

# quick map of correlations
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=-1, vmax=1, cmap='tab10')

ax.set_title('Map of SPI-SPEI Correlation (DJFM)')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
"""

#####################################################################################################
######################################################################################################
PPTsub = PPT.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
        date=PPT['date.month'].isin([4, 5, 6, 7])).resample(date='1Y').mean(dim='date')
#Drop 2023 so the datasets match
PETsub = PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
         date=PET['date.month'].isin([4, 5, 6, 7])).resample(date='1Y').mean(dim='date')

tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should be the same since one of the above functions changes formatting')

spei_data = xr.DataArray(dims=["date", "lats","lons"], coords={"date": PPTsub["date"], "lons": PPTsub["lons"],"lats": PPTsub["lats"]})


tic = time.time()
#Use the SPEI function for every coordinate, then take the last value since we're mapping DJF of the 2021-2022 year 
for i in range(len((PPT["lats"].values))):
   for j in range(len((PPT["lons"].values))):
       SPEI_arr =GET_SPEI_FROM_D(tmpPPT[:,i,j]-tmpPET[:,i,j])
       spei_data[:,i,j] = SPEI_arr
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))  

#####################################################################################################################
spi_data = xr.DataArray(dims=["date", "lats","lons"], coords={"date": PPTsub["date"], "lons": PPTsub["lons"],"lats": PPTsub["lats"]})

tic = time.time()
#Use the SPI function for every coordinate, then take the last value since we're mapping DJF of the 2021-2022 year 
for i in range(len((PPT["lats"].values))):
   for j in range(len((PPT["lons"].values))):
       SPI_arr =Precip_2_SPI(tmpPPT[:,i,j])
       spi_data[:,i,j] = SPI_arr
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))  

#####################################################################################################################
tmpSPI = spi_data.values
tmpSPEI = spei_data.values

print(tmpSPI.shape,tmpSPEI.shape,'these should have the same numbers')

subNY,subNX = tmpSPI[0,:,:].shape

r_vals = np.zeros([subNY,subNX]) * np.nan

tic = time.time()

for i in range(len((PPT["lats"].values))):
   for j in range(len((PPT["lons"].values))):
       r_vals[i,j] = linregress(tmpSPI[:,i,j],tmpSPEI[:,i,j])[2]
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))    

# quick map of correlations
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=-1, vmax=1, cmap='tab10')

ax.set_title('Map of SPI-SPEI Correlation(AMJJ)')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
"""
##############################################################################################################
##############################################################################################################
PPTsub = PPT.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
        date=PPT['date.month'].isin([8, 9, 10, 11])).resample(date='1Y').mean(dim='date')
#Drop 2023 so the datasets match
PPTsub = PPT.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
        date=PPT['date.month'].isin([8, 9, 10, 11])).resample(date='1Y').mean(dim='date')

tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should be the same since one of the above functions changes formatting')

spei_data = xr.DataArray(dims=["date", "lats","lons"], coords={"date": PPTsub["date"], "lons": PPTsub["lons"],"lats": PPTsub["lats"]})


tic = time.time()
#Use the SPEI function for every coordinate, then take the last value since we're mapping DJF of the 2021-2022 year 
for i in range(len((PPT["lats"].values))):
   for j in range(len((PPT["lons"].values))):
       SPEI_arr =GET_SPEI_FROM_D(tmpPPT[:,i,j]-tmpPET[:,i,j])
       spei_data[:,i,j] = SPEI_arr
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))  

#####################################################################################################################
spi_data = xr.DataArray(dims=["date", "lats","lons"], coords={"date": PPTsub["date"], "lons": PPTsub["lons"],"lats": PPTsub["lats"]})

tic = time.time()
#Use the SPI function for every coordinate, then take the last value since we're mapping DJF of the 2021-2022 year 
for i in range(len((PPT["lats"].values))):
   for j in range(len((PPT["lons"].values))):
       SPI_arr =Precip_2_SPI(tmpPPT[:,i,j])
       spi_data[:,i,j] = SPI_arr
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))  

#####################################################################################################################
tmpSPI = spi_data.values
tmpSPEI = spei_data.values

print(tmpSPI.shape,tmpSPEI.shape,'these should have the same numbers')

subNY,subNX = tmpSPI[0,:,:].shape

r_vals = np.zeros([subNY,subNX]) * np.nan

tic = time.time()

for i in range(len((PPT["lats"].values))):
   for j in range(len((PPT["lons"].values))):
       r_vals[i,j] = linregress(tmpSPI[:,i,j],tmpSPEI[:,i,j])[2]
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))    

# quick map of correlations
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=0.7, vmax=1, cmap='tab10')

ax.set_title('Map of SPI-SPEI Correlation(ASON)')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
"""
