
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')

 #read in climatology
# get data from https://data.chc.ucsb.edu/products/CHPclim/netcdf/
#%%
data_dir = '/Users/disha/downloads/'
infile = 'chirps-v2.0.monthly.nc'

CHIRPS = xr.open_dataset(data_dir+infile)
#%%
## FINDING THE 95th PERCENTILE RAINFALL AFRICA
# set spatial subset dimensions
minlat = -40 ; maxlat = 40
minlon = -20 ; maxlon = 60
africa_subset = CHIRPS.sel(latitude= slice(minlat,maxlat),longitude = slice(minlon,maxlon))
africa_subset.load()

top_95 = africa_subset.fillna(0).precip.quantile(0.95, dim='time')
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

map_95 = plt.pcolormesh(top_95.longitude,top_95.latitude,top_95,vmin=0,vmax=25, cmap='PuBuGn')
ax.set_title('Africa - 95 percentile Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray

# %%
## HISTOGRAM OF AFRICA'S AVERAGE PRECIPITATION

AfricaAvgPrec = africa_subset.precip.mean("time") #get the average precipitation in Africa per month
AfricaAvgPrec.plot.hist() # plot histogram 
plt.title('Africa Total Average Precipitation') # add title
#%%
## FINDING WHERE AFRICA'S AVERAGE PRECIPITATION IS BELOW 5MM

Africa_Prec_below_5mm = AfricaAvgPrec.where(AfricaAvgPrec.values <5)
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

avgmap5 = plt.pcolormesh(Africa_Prec_below_5mm.longitude,Africa_Prec_below_5mm.latitude,Africa_Prec_below_5mm,\
                        vmin=0, vmax=5, cmap='PuBuGn')
    
ax.set_title('Africa_Prec_below_5mm')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
ax.add_feature(cartopy.feature.OCEAN, zorder=100, edgecolor='k')
cb = plt.colorbar(avgmap5,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

# %%
