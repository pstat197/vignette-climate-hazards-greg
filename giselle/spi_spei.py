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
