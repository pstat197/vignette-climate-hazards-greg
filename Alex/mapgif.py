# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 20:46:06 2023

@author: foamy
"""

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
infile = 'chirps-v2.0.monthly.nc'

CHIRPS = xr.open_dataset(data_dir+infile,chunks = "auto")
"""
#JAN MAP
CHIRPSJan = CHIRPS.precip[0,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

janmap = plt.pcolormesh(CHIRPSJan.longitude,CHIRPSJan.latitude,CHIRPSJan,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('January Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(janmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

#FEB MAP
CHIRPSFeb = CHIRPS.precip[1,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

febmap = plt.pcolormesh(CHIRPSFeb.longitude,CHIRPSFeb.latitude,CHIRPSFeb,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('February Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(febmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

#MARCH MAP
CHIRPSMar = CHIRPS.precip[2,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

marmap = plt.pcolormesh(CHIRPSMar.longitude,CHIRPSMar.latitude,CHIRPSMar,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('March Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(marmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar


#April MAP
CHIRPSApr = CHIRPS.precip[3,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

aprmap = plt.pcolormesh(CHIRPSApr.longitude,CHIRPSApr.latitude,CHIRPSApr,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('April Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(aprmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

#MAY MAP
CHIRPSMay = CHIRPS.precip[4,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

maymap = plt.pcolormesh(CHIRPSMay.longitude,CHIRPSMay.latitude,CHIRPSMay,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('May Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(maymap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

#JUNE MAP
CHIRPSJune = CHIRPS.precip[5,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

junemap = plt.pcolormesh(CHIRPSJune.longitude,CHIRPSJune.latitude,CHIRPSJune,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('June Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(junemap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar


#JULY MAP
CHIRPSJuly = CHIRPS.precip[6,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

julymap = plt.pcolormesh(CHIRPSJuly.longitude,CHIRPSJuly.latitude,CHIRPSJuly,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('July Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(julymap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar


#AUGUST MAP
CHIRPSAugust = CHIRPS.precip[7,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

augustmap = plt.pcolormesh(CHIRPSAugust.longitude,CHIRPSAugust.latitude,CHIRPSAugust,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('August Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(augustmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar


#September MAP
CHIRPSeptember = CHIRPS.precip[8,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

sepmap = plt.pcolormesh(CHIRPSeptember.longitude,CHIRPSeptember.latitude,CHIRPSeptember,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('September Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(sepmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

#October MAP
CHIRPOctober = CHIRPS.precip[9,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

octobermap = plt.pcolormesh(CHIRPOctober.longitude,CHIRPOctober.latitude,CHIRPOctober,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('October Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(octobermap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
#november MAP
CHIRPSNovember = CHIRPS.precip[10,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

novembermap = plt.pcolormesh(CHIRPSNovember.longitude,CHIRPSNovember.latitude,CHIRPSNovember,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('November Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(novembermap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
"""
#december MAP
CHIRPSdecember = CHIRPS.precip[11,:,:]
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

decembermap = plt.pcolormesh(CHIRPSdecember.longitude,CHIRPSdecember.latitude,CHIRPSdecember,\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('December Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(decembermap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar