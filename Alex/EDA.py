# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:33:15 2023

@author: foamy
"""
from dask.distributed import Client
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#%% read in climatologyss
# get data from https://data.chc.ucsb.edu/products/CHPclim/netcdf/
data_dir = '/Users/foamy/Downloads/CHC/'
infile = 'chirps-v2.0.monthly.nc'
CHIRPS = xr.open_dataset(data_dir+infile,chunks = "auto")
#set spatial subset dimensions
"""
AvgPrec = CHIRPS.precip.mean("time")
AvgPrec25 = AvgPrec.where(AvgPrec.values <25)
AvgPrec10 = AvgPrec.where(AvgPrec.values <10)
AvgPrec5 = AvgPrec.where(AvgPrec.values < 5)
AvgPrec.plot.hist()
#%%

#INITIAL AVERAGE 
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

avgmap = plt.pcolormesh(AvgPrec.longitude,AvgPrec.latitude,AvgPrec,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k')
cb = plt.colorbar(avgmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
"""
"""
#AVERAGE EXLUDING PREC BELOW 25
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

avgmap25 = plt.pcolormesh(AvgPrec25.longitude,AvgPrec25.latitude,AvgPrec25,\
                        vmin=0, vmax=25, cmap='PuBuGn')
    
ax.set_title('Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k')
cb = plt.colorbar(avgmap25,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar


#AVERAGE EXLUDING PREC BELOW 5
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

avgmap5 = plt.pcolormesh(AvgPrec5.longitude,AvgPrec5.latitude,AvgPrec5,\
                        vmin=0, vmax=5, cmap='PuBuGn')
    
ax.set_title('Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k')
cb = plt.colorbar(avgmap5,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
"""
 



