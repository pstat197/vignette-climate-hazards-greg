#SPI/SPEI
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

#from CoreFuncs import FILL_HOBBINS_DEKAD_GLOBAL
#%%
from functions_greg import Precip_2_SPI, GET_LLD_PARS, GET_SPEI_FROM_D

#%%
#opening dataset
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
PET_monthly = xr.open_mfdataset(file_paths, combine='nested', concat_dim='date')
PET_monthly = PET_monthly.sortby('date')
#%% download here and graph it
data_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'
# %% 
#chirps = xr.open_dataset(data_dir+infile, )
CHIRPS= xr.open_dataset(data_dir+infile,chunks = "auto")
CHIRPS_monthly = CHIRPS.rename({'time': 'date', 'latitude': 'lats', 'longitude': 'lons'})

#%%
minlat = -23; maxlat = -15
minlon = 24; maxlon = 34
SA_PET =  PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT =  CHIRPS_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT.load()
#%%

import matplotlib as mpl
import matplotlib.pyplot as plt
#from CoreFuncs import FILL_HOBBINS_DEKAD_GLOBAL
import pandas as pd
import time
from datetime import date
from scipy.stats import spearmanr, pearsonr, mode, linregress
import hvplot.xarray
import holoviews as hv
hv.extension('bokeh', width=80)
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')

#%% read in climatology
data_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'

#make a quick visual of the january average rainfall
CHIRPS_monthly.precip[0,:,:].plot.pcolormesh()

#%% make a better map of the rainfall

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

janmap = plt.pcolormesh(CHIRPS_monthly.longitude,CHIRPS_monthly.latitude,CHIRPS_monthly.precip[0,:,:],\
                        vmin=0, vmax=400, cmap='PuBuGn')

ax.set_title('January Average Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(janmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

#plt.savefig('testmap.png',dpi=200)

#%% find the index of the maximum value
maxind = clim.precip.argmax(dim="time")

projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

monmap = plt.pcolormesh(clim.longitude,clim.latitude,maxind,\
                        vmin=-0.5, vmax=11.5, cmap='hsv')

ax.set_title('Month of Maximum Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='white',zorder=100) #color the oceans
cb = plt.colorbar(monmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal',\
                  ticks=[0,1,2,3,4,5,6,7,8,9,10,11],) #add colorbar
cb.ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

## try with discrete colorbar
fig = plt.figure(figsize=(12,4))  #make the window for the graphics
ax = plt.axes([0.05,0.15,0.9,0.8],projection=projection)  # set the drawing area within the window

cmap = plt.cm.get_cmap('gist_rainbow')
clevels = np.arange(-0.5,12.,1.0)
monmap = plt.pcolormesh(clim.longitude,clim.latitude,maxind,\
                        vmin=clevels[0], vmax=11.5, cmap=cmap,\
                        norm=mpl.colors.BoundaryNorm(clevels,ncolors=cmap.N))

ax.set_title('Month of Maximum Rainfall')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
#ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
ax.add_feature(cartopy.feature.OCEAN,color='white',zorder=100) #color the oceans
cb = plt.colorbar(monmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal',\
                  ticks=[0,1,2,3,4,5,6,7,8,9,10,11],) #add colorbar
cb.ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

#%% load in monthly data

data_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'
# %% 
#chirps = xr.open_dataset(data_dir+infile, )
CHIRPS= xr.open_dataset(data_dir+infile,chunks = "auto")
CHIRPS_monthly = CHIRPS.rename({'time': 'date', 'latitude': 'lats', 'longitude': 'lons'})
CHIRPSp = xr.open_dataset(data_dir+infile)

#set spatial subset dimensions
minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.
CHsub = CHIRPSp.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))
CHsub.load()

CHsub.precip[0,:,:].plot.pcolormesh()

#monthly average of the data
climatology = CHsub.groupby('time.month').mean('time')  
#each month's difference from the monthly average
anoms = CHsub.groupby('time.month') - climatology  

AOIts = CHsub.precip.sel(latitude=slice(minlat+5,maxlat-5),\
                         longitude=slice(minlon+5,maxlon-5)).\
                     mean(dim=['latitude','longitude'])

#monthly 3-month totals of rainfall
CHsub3mo = CHsub.rolling(time=3).sum()
x=200
y=200
CHsub.precip[0:12,y,x].values
CHsub3mo.precip[0:12,y,x].values

#%% make monthly Hobbins netcdf data

for m in range(1,13):
  
  tic = time.time()
  petblock,PixInfo2,ProjInfo2 = FILL_HOBBINS_DEKAD_GLOBAL((m-1)*3+1,m*3)
  toc = time.time()
  print('{:10.2f} sec elapsed for PET read-in'.format(toc-tic))
  ssnPET = petblock[800:2800,:,:,:].sum(axis=3)   # trim to match dimensions of rainblock
  print('Size of ssnPET is ',ssnPET.shape)

  subNY,subNX,Nyrs = ssnPET.shape
  ssnPET[ssnPET < 0.0] = np.nan

  lons = np.around(np.arange(\
    PixInfo2[0]+(PixInfo2[1]/2),-1*PixInfo2[0],PixInfo2[1]),decimals=3)
  lats = np.around(np.arange(\
    PixInfo2[3]+(PixInfo2[5]/2),-1*PixInfo2[3],PixInfo2[5]),decimals=3)
    
  strt_d8 = date(1981,m,1)
  d8s = pd.date_range(start=strt_d8, periods=Nyrs,freq='12MS')
  pet_da = xr.DataArray(np.flipud(ssnPET),name='PET',\
                         coords=[np.flipud(lats[800:2800]),lons,d8s],\
                         dims=['lats','lons','date'])
  out_dir = '/home/chc-data-out/people/husak/forCapstone/'
  outfile = out_dir+'PETmonthly_{:02d}.nc'.format(m)
  pet_da.to_netcdf(outfile)
  
#%%
data_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'
# %% 
#chirps = xr.open_dataset(data_dir+infile, )
CHIRPS= xr.open_dataset(data_dir+infile,chunks = "auto")
CHIRPS_monthly = CHIRPS.rename({'time': 'date', 'latitude': 'lats', 'longitude': 'lons'})
#%% correlations between PPT and PET

# read in CHIRPS  

PPT_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
infile = 'chirps-v2.0.monthly (1).nc'
PPT = xr.open_dataset(PPT_dir+infile)

# read in Hobbins
#PET_dir = 'C:/Users/gizra/OneDrive/Documents/netcdf/'
#PET = xr.open_mfdataset(PET_dir+'*.nc',combine='nested',concat_dim='date').sortby('date')

PET_monthly = xr.open_mfdataset(file_paths, combine='nested', concat_dim='date')
PET_monthly = PET_monthly.sortby('date')

PET = PET_monthly

#set spatial subset dimensions and specific month
minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.
moi = 5 # month of interest 1=Jan, 2=Feb, ... 12=Dec

PPTsub = PPT.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),\
                 time=PPT['time.month'] == moi)
PETsub = PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
                 date=PET['date.month'] == moi)
 
tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should have the same numbers, but out of order')

subNY,subNX = tmpPET[:,:,0].shape

tmplats = np.arange(subNY+1)*0.05 + minlat #make array of latitude edges for pcolormesh
tmplons = np.arange(subNX+1)*0.05 + minlon #make array of longitude edges for pcolormesh

r_vals = np.zeros([subNY,subNX]) * np.nan

gvals = np.where(tmpPPT[0,:,:] >= 0.00)
nvals = len(gvals[0])

tic = time.time()
for i in range(nvals):
  r_vals[gvals[0][i],gvals[1][i]] = linregress(tmpPPT[:,gvals[0][i],gvals[1][i]],\
                      tmpPET[gvals[0][i],gvals[1][i],:])[2]
toc = time.time()
print('{:10.2f} sec elapsed for correlation calculation'.format(toc-tic))    

# quick map of correlations
projection = ccrs.PlateCarree()  #set the projection of the map
fig = plt.figure(figsize=(6,6))  #make the window for the graphics
ax = plt.subplot(111,projection=projection)  # set the drawing area within the window

tmpmap = ax.pcolormesh(PETsub.lons,PETsub.lats,r_vals,\
                        vmin=-1, vmax=1, cmap='BrBG')

ax.set_title('Map of Correlation')  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar
#%%
SA_PPT_DJF = SA_PPT.sel(date=SA_PPT['date.month'].isin([1, 2, 12])).resample(date='1Y').mean(dim='date').precip.mean(dim=('lons', 'lats')).to_pandas()
SA_PET_DJF = SA_PET.sel(date=SA_PET['date.month'].isin([1, 2, 12])).resample(date='1Y').mean(dim='date').PET.mean(dim=('lons', 'lats')).to_pandas()
SA_PET_DJF = SA_PET_DJF.iloc[:-1] #PET dataset has one more 
D = (SA_PPT_DJF.array-SA_PET_DJF.array).to_numpy()
#%%
"""
PET_PPT = pd.concat([SA_PPT_DJF, SA_PET_DJF],keys = ['PPT','PET'], axis=1) #concat both dataset
seaborn.lmplot(x='PPT', y='PET', data=PET_PPT, line_kws={'color': 'red'}, ci=None)
corr = np.corrcoef(SA_PPT_DJF, SA_PET_DJF)[0, 1]
rsq = corr**2

# Show the plot
plt.show()



resid_df = PET_PPT['PPT']-PET_PPT['PET']
resid_df.plot()


PET_PPT['indate']= np.where((PET_PPT.index >= '2004-12-31'), 'After 2004', 'Before 2004')
g=seaborn.lmplot(x='PPT', y='PET', data=PET_PPT, ci=None, hue = 'indate', fit_reg=False)
seaborn.regplot(x="PPT", y="PET", data=PET_PPT, scatter=False, ax=g.axes[0, 0],ci = None)
plt.show()


minlat = -3.5; maxlat = -2
minlon = 10; maxlon = 13.5
SA_PET =  PET_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT =  CHIRPS_monthly.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon))
SA_PPT.load()

SA_PPT_DJF = SA_PPT.sel(date=SA_PPT['date.month'].isin([9, 10, 11])).resample(date='1Y').mean(dim='date').precip.mean(dim=('lons', 'lats')).to_pandas()
SA_PET_DJF = SA_PET.sel(date=SA_PET['date.month'].isin([9, 10, 11])).resample(date='1Y').mean(dim='date').PET.mean(dim=('lons', 'lats')).to_pandas()

PET_PPT = pd.concat([SA_PPT_DJF, SA_PET_DJF],keys =['PPT','PET'], axis=1) #concat both dataset
seaborn.lmplot(x='PPT', y='PET', data=PET_PPT, line_kws={'color': 'red'}, ci=None)
corr = np.corrcoef(SA_PPT_DJF, SA_PET_DJF)[0, 1]
rsq = corr**2

# Show the plot
plt.show()



resid_df = PET_PPT['PPT']-PET_PPT['PET']
resid_df.plot()


PET_PPT['indate']= np.where((PET_PPT.index >= '1992-12-31') & (PET_PPT.index <= '2010-12-31'), 'Between 1992 and 2010', 'Outisde 1992 and 2010')
g=seaborn.lmplot(x='PPT', y='PET', data=PET_PPT, ci=None, hue = 'indate', fit_reg=False)
seaborn.regplot(x="PPT", y="PET", data=PET_PPT, scatter=False, ax=g.axes[0, 0],ci = None)
plt.show()
"""

pd.DataFrame(Precip_2_SPI(SA_PPT_DJF.array)).plot()

GET_SPEI_FROM_D(D)
# %%
