# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:04:05 2023

@author: foamy
"""
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#%% read in climatology
# get data from https://data.chc.ucsb.edu/products/CHPclim/netcdf/
data_dir = '/Users/foamy/Downloads/CHC/netcdf/'
infile = 'chpclim.5050.monthly.nc'

clim = xr.open_dataset(data_dir+infile)
"""
ThrMA = clim.rolling(time=3).mean().dropna('time')

#%% find the index of the maximum value
maxind = ThrMA.precip.argmax(dim="time")

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

monmap = plt.pcolormesh(clim.longitude,clim.latitude,maxind,\
                        vmin=-0.5, vmax=9.5, cmap='hsv')

ax.set_title('3 Month Period of Maximum Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='white',zorder=100) #color the oceans
cb = plt.colorbar(monmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal',\
                  ticks=[0,1,2,3,4,5,6,7,8,9],) #add colorbar
cb.ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O'])
"""
AvgPrec = clim.precip.mean("time")
"""
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window
"""
AvgPrec.plot.hist()
AvgPrecB = AvgPrec.where(AvgPrec.values >200)

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

janmap = plt.pcolormesh(AvgPrecB.longitude,AvgPrecB.latitude,AvgPrecB,\
                        vmin=0, vmax=5, cmap='PuBuGn')

ax.set_title('January Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k')
cb = plt.colorbar(janmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
