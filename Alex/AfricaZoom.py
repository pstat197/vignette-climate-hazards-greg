# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 01:05:29 2023

@author: foamy
"""
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
#import pymannkendall

#%%
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
data_dir = '/Users/disha/downloads/'
infile = 'chpclim.5050.monthly.nc'

CHIRPS = xr.open_dataset(data_dir+infile)

#set spatial subset dimensions
minlat = -40.; maxlat = 40.
minlon = -20; maxlon = 50.
CHsub =  CHIRPS.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
CHsub.load()
#%%
"""
AvgPrec = CHsub.precip.mean("time")
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

meanmap = plt.pcolormesh(AvgPrec.longitude,AvgPrec.latitude,AvgPrec,\
                        vmin=0, vmax=100, cmap='PuBuGn')

ax.set_title('Africa - Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k')
cb = plt.colorbar(meanmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
"""

CovPrec =  CHsub.precip.std("time")/ CHsub.precip.mean("time")

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

covmap = plt.pcolormesh(CovPrec.longitude,CovPrec.latitude,CovPrec,\
                        vmin=0, vmax=1, cmap='PuBuGn')

ax.set_title('Africa - Coefficient of Variation Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k')
cb = plt.colorbar(covmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

# %%
