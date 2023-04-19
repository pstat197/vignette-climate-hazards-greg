#%%
import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import seaborn
import cartopy.feature
import matplotlib.pyplot as plt
import pymannkendall as mk
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
#from CoreFuncs import FILL_HOBBINS_DEKAD_GLOBAL
import hvplot.xarray
import holoviews as hv
hv.extension('bokeh', width=80)
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#from CoreFuncs import FILL_HOBBINS_DEKAD_GLOBAL
#from functions_greg import Precip_2_SPI, GET_LLD_PARS, GET_SPEI_FROM_D
#%% correlations between PPT and PET
# read in CHIRPS  
PPT_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'
PPT = xr.open_dataset(PPT_dir+infile)

# read in Hobbins
file_paths = ['C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_01.nc',
              'C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_02.nc',
              'C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_03.nc',
              'C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_04.nc',
              'C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_05.nc',
              'C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_06.nc',
              'C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_07.nc',
              'C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_08.nc',
              'C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_09.nc',
              'C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_10.nc',
              'C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_11.nc',
              'C:/Users/gizra/OneDrive/Documents/netcdf/PETmonthly_12.nc']
#PET_dir = '/home/chc-data-out/people/husak/forCapstone/'
PET = xr.open_mfdataset(file_paths, combine='nested', concat_dim='date').sortby('date')
#xr.open_mfdataset(PET_dir+'*.nc',combine='nested',concat_dim='date')
#this didn't work with the sortby('date') 
#PET = xr.open_mfdataset(PET_dir+'*.nc',combine='nested',concat_dim='date').sortby('date')
#%%
#set spatial subset dimensions and specific month
minlat = -35.; maxlat = 13.
minlon = -20.; maxlon = 52.
#minlat = -35.; maxlat = 40.
#minlon = -20.; maxlon = 52. 

moi = 7 # month of interest 1=Jan, 2=Feb, ... 12=Dec

PPTsub = PPT.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),\
                 time=PPT['time.month'] == moi)
PETsub = PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
                 date=PET['date.month'] == moi).sortby('date')
#%%
PPTsub.precip[0,:,:].plot.pcolormesh()
#%%
PETsub.PET[:,:,0].plot.pcolormesh()

#%% 
tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should have the same numbers, but out of order')

#if the numbers aren't the same, then you might have to remove the final year from
# the PPT because it may have been updated
# tmpPPT = tmpPPT[:-1,:,:]
#%%
subNY,subNX = tmpPET[:,:,0].shape

r_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])
#%%
tic = time.time()
for i in range(nvals):
  r_vals[gvals[0][i],gvals[1][i]] = linregress(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                      tmpPET[gvals[0][i],gvals[1][i],:])[2]
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))    
#%%
# quick map of correlations
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = fig.add_axes([0.05,0.05,0.9,0.9],projection=projection) # set the drawing area
#ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PPTsub.longitude,PPTsub.latitude,r_vals,\
                        vmin=-1, vmax=1, cmap='BrBG') #cmap='tab20'
#PETsub.lons,PETsub.lats
ax.set_title('Map of Correlation Month = {:02d}'.format(moi))  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
#fig.tight_layout()
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

# %%
