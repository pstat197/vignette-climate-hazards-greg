# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
#make a quick visual of the january average rainfall'

agg_1 = []
for i in range(0,12):
   x= float(clim.precip[i,:,:].sum().to_numpy())
   agg_1.append(x)
clim.precip[agg_1.index(max(agg_1)),:,:].plot.pcolormesh()

"""
agg_3 = []
ctr = 0
for i in range(0,10):
   x= float(clim.precip[ctr,:,:].sum()+clim.precip[ctr+1,:,:].sum()+clim.precip[ctr+2,:,:].sum())
   agg_3.append(x)
   ctr+=1   
x_arr = (clim.precip[agg_3.index(max(agg_3)),:,:]+clim.precip[agg_3.index(max(agg_3))+1,:,:]+clim.precip[agg_3.index(max(agg_3))+2,:,:])/3
x_arr.plot.pcolormesh()
"""
"""
clim.precip[0,:,:].plot.pcolormesh()
#%% make a better map of the rainfall
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window
janmap = plt.pcolormesh(clim.longitude,clim.latitude,clim.precip[0,:,:],\
                        vmin=0, vmax=400, cmap='PuBuGn')
ax.set_title('January Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(janmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
"""
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window
janmap = plt.pcolormesh(clim.longitude,clim.latitude,clim.precip[agg_1.index(max(agg_1)),:,:],\
                        vmin=0, vmax=400, cmap='PuBuGn')
ax.set_title('June Average Rainfall')  #put a title on the map
ax.coastlines(color='black') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='black') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='blue',zorder=100) #color the oceans
cb = plt.colorbar(janmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar