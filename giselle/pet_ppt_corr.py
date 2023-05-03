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
#from pmdarima.arima import ARIMA
#from pmdarima.arima import auto_arima
import time
from datetime import date
from scipy.stats import spearmanr, pearsonr, mode, linregress
import matplotlib as mpl
import matplotlib.pyplot as plt
#from CoreFuncs import FILL_HOBBINS_DEKAD_GLOBAL
import hvplot.xarray
import holoviews as hv

from functions_greg import Precip_2_SPI, GET_LLD_PARS, GET_SPEI_FROM_D
hv.extension('bokeh', width=80)
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#from CoreFuncs import FILL_HOBBINS_DEKAD_GLOBAL
#from functions_greg import Precip_2_SPI, GET_LLD_PARS, GET_SPEI_FROM_D
#%% correlations between PPT and PET
# read in CHIRPS  
'''
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
'''
#%% download here and graph it
PPT_dir = '/home/chc-data-out/products/CHIRPS-2.0/global_monthly/netcdf/'
infile = 'chirps-v2.0.monthly.nc'
PPT = xr.open_dataset(PPT_dir+infile)
# get data from https://data.chc.ucsb.edu/products/CHPclim/netcdf/
PET_dir = '/home/chc-data-out/people/husak/forCapstone/'
import glob
fnames = sorted(glob.glob(PET_dir+'*.nc'))
PET = xr.open_mfdataset(PET_dir+'*.nc',combine='nested',concat_dim='date').sortby('date')
#PETmonthly_01.nc  PETmonthly_03.nc  PETmonthly_05.nc  PETmonthly_07.nc  PETmonthly_09.nc  PETmonthly_11.nc
#PETmonthly_02.nc  PETmonthly_04.nc  PETmonthly_06.nc  PETmonthly_08.nc  PETmonthly_10.nc  PETmonthly_12.nc

#%%
#set spatial subset dimensions and specific month
minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.
#minlat = -35.; maxlat = 18.
#minlon = -20.; maxlon = 52. 
#drop_2mon = congo.drop([np.datetime64('1981-01-01'), np.datetime64('1981-02-01')], dim='time')
#DA_DJF = drop_2mon.sel(time=drop_2mon.time.dt.season=="DJF")
moi = 11 # month of interest 1=Jan, 2=Feb, ... 12=Dec
#PET = PET.sel(lons=slice(minlon,maxlon), lats=slice(minlat,maxlat))
#PPT = PPT.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
#%% functions
'''
def is_djfm(month):
    return ((month >= 1) & (month <= 3)) | (month == 12)

seasonal_data = PPT.sel(time=is_djfm(PPT['time.month'])).shift(time=1).resample(time='1Y').mean(dim='date').drop('2023-12-31', dim='date')
'''
#%%

PPTsub = PPT.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),\
                 time=PPT['time.month'] == moi)
                 #time=PPT.time.dt.season=="DJF")
                 #time=PPT['time.month'].isin([8,9,10,11])).shift(time=1).resample(time='1Y').mean(dim='time')#.drop('2023-12-31', dim='time')
#data.where(((data['time.year'] == 2020) & (data['time.month'] == 1)), drop=True)
                 #time=PPT['time.month'] == moi)
#%%
PETsub = PET.sel(lons=slice(minlon,maxlon), lats=slice(minlat,maxlat), \
                 #date=PET['date.month'].isin([8,9,10,11])).shift(date=1).resample(date='1Y').mean(dim='date')#.drop('2023-12-31', dim='date')
#PETsub = PETsub.PET.values.reshape(700,900,42)
#.reshape(2,3,4)#.mean(dim='time')
                 #date=PET.date.dt.season)
                 date=PET['date.month'] == moi).sortby('date')
#%%
#PPTsub.precip[0,:,:].plot.pcolormesh()
#%%
#PETsub.PET[:,:,0].plot.pcolormesh()

#%% 
tmpPPT = PPTsub.precip.values #43,700,900
tmpPET = PETsub.PET.values#.reshape(700, 900, 42)

print(tmpPPT.shape,tmpPET.shape,'these should have the same numbers, but out of order')
#%%
#if the numbers aren't the same, then you might have to remove the final year from
# the PPT because it may have been updated
#tmpPPT = tmpPPT[:-1,:,:]
#tmpPET = tmpPET[:,:,:-1]
#%%

#%%
subNY,subNX = tmpPET[:,:,0].shape

r_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])
#%%
tic = time.time()
for i in range(nvals):
  #for j in range(len((PET["lons"].values))):
    #if tmpPPT[:, gvals[0][i],gvals[1][i]].std() > 1:
  #if std devation of precip (tmpPPT with gvals) = 0 or less than 1 (SKIP)
  # if tmpPPT[:,gvals[0][i],gvals[1][i].std() > 1:#greater than 1: run below
  if tmpPPT[:, gvals[0][i],gvals[1][i]].std() > 1:
  #r_vals[gvals[0][i],gvals[1][i]]
    r_vals[gvals[0][i],gvals[1][i]] = linregress(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                                                 tmpPET[gvals[0][i],gvals[1][i],:])[2]
  #elif tmpPPT[:, gvals[0][i],gvals[1][i]].std() < 1:
  else:
    continue
  #elif tmpPPT[:, gvals[0][i],gvals[1][i]].std() == 0:
    #continue
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))    
#%% colorCET (color map)
# quick map of correlations
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
#ax = fig.add_axes([0.05,0.05,0.9,0.9],projection=projection) # set the drawing area
ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=-1, vmax=1, cmap='tab20b') #cmap='tab20'
#PETsub.lons,PETsub.lats
ax.set_title('Map of Correlation Month = {:02d}'.format(moi))  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
#fig.tight_layout()
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar


# %%
#correlation (SPI and SPEI) compare to PET and PPT  (maps are similar)
#calc residuals (linear relationship; complex b/t precip and PPT)
#use precip to drive est of PPT
#look at diff and our est of PET (per month and wet season)
#PET: dependent var and PPT: independent variable
