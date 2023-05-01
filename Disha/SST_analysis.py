
import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import seaborn
import cartopy.feature
import matplotlib.pyplot as plt
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

count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')

#%%

# read in data 
in_dir = '/Users/disha/downloads/'
infile = in_dir + 'sst.mnmean.nc'

SSTS = xr.open_dataset(infile)
PPT_dir = '/Users/disha/downloads/'
infile = 'chirps-v2.0.monthly.nc'
PPT = xr.open_dataset(PPT_dir+infile)


file_paths = ['/Users/disha/Downloads/PETmonthly_01.nc','/Users/disha/Downloads/PETmonthly_02.nc',
              '/Users/disha/Downloads/PETmonthly_03.nc','/Users/disha/Downloads/PETmonthly_04.nc',
              '/Users/disha/Downloads/PETmonthly_05.nc','/Users/disha/Downloads/PETmonthly_06.nc',
              '/Users/disha/Downloads/PETmonthly_07.nc','/Users/disha/Downloads/PETmonthly_08.nc',
              '/Users/disha/Downloads/PETmonthly_09.nc','/Users/disha/Downloads/PETmonthly_10.nc',
              '/Users/disha/Downloads/PETmonthly_11.nc','/Users/disha/Downloads/PETmonthly_12.nc']
PET = xr.open_mfdataset(file_paths, combine='nested', concat_dim='date').sortby('date')


#%%
minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.
moi =2 # month of interest 1=Jan, 2=Feb, ... 12=Dec

PPTsub = PPT.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),\
                 time=PPT['time.month'] == moi)
PETsub = PET.sel(lats=slice(minlat,maxlat),lons=slice(minlon,maxlon),\
                 date=PET['date.month'] == moi).sortby('date')
 
tmpPPT = PPTsub.precip.values
tmpPET = PETsub.PET.values

print(tmpPPT.shape,tmpPET.shape,'these should have the same numbers, but out of order')

ppt_ts = PPTsub.precip.sel(latitude=slice(-23.,-14.),longitude=slice(25.,35.)).mean(axis=(1,2)).values
pet_ts = PETsub.PET.sel(lats=slice(-23.,-14.),lons=slice(25.,35.)).mean(axis=(0,1)).values

print(ppt_ts.shape,pet_ts.shape,'these should have the same numbers, but out of order')


regvals = linregress(ppt_ts,pet_ts)
pet_est = regvals[1] + (regvals[0] * pet_ts)
resids = pet_ts - pet_est

#calculate a rolling 3-month average, and select the average with the endmonth equal to SST_moi
SST_moi =2 #the final month in the temporal average
#sst3 = SSTS.sst.rolling(time=3).mean().values
sstvals = SSTS.sst.rolling(time=3).mean().sel(time=(SSTS.time['time.month'] == SST_moi)).values

nyrs,NY,NX = sstvals.shape
corr_mat = np.zeros([NY,NX]) * np.nan

tic = time.time()
for x in range(NX):
  for y in range(NY):
    corr_mat[y,x] = linregress(resids[1:],sstvals[:,y,x])[2]
toc = time.time()
print(toc-tic, ' sec elapsed for correlation calculation')
    
resid_SST_corr = xr.DataArray(corr_mat,coords=[SSTS.lat,SSTS.lon],dims=['lats','lons'])
    
projection = ccrs.PlateCarree(central_longitude=180)
ncolors = 9  # number of colors to display in tshe map
colorcuts = [-0.8,-0.7,-0.6,-0.5,-0.35,0.35,0.5,0.6,0.7,0.8]  # thresholds for colors (ncolors + 1) values in array

fig = plt.figure(figsize = (8,4.5))
ax0 = plt.subplot(111,projection=projection)
tmpgr = ax0.contourf(SSTS.lon,SSTS.lat,corr_mat,colorcuts,\
                     levels=colorcuts,extend='both',transform=ccrs.PlateCarree(),\
                     cmap='coolwarm_r')

plt.title('Correlation w/ Residuals',fontsize=12)
ax0.coastlines(); ax0.add_feature(count_bord,edgecolor='black')
plt.tight_layout()

#cbar = fig.colorbar(tmpgr, cax=fig.add_axes([0.2,0.07,0.6,0.03]), orientation='horizontal',shrink=0.75, aspect=40)
cbar = fig.colorbar(tmpgr, cax=fig.add_axes([0.05,0.04,0.9,0.03]),orientation='horizontal')
cbar.ax.tick_params(labelsize=8)
plt.tight_layout()

# %%
# %%

## scattter plot for ppt and pet month 2


fig = plt.figure(figsize = (6,6))
ax2 = fig.add_axes([0.15,0.1,0.80,0.80])
tsscat = ax2.plot(ppt_ts,pet_ts,'o',lw=0)
#tsline = ax2.plot([ppt_ts.min(),ppt_ts.max()],[pet_est.max(),pet_est.min()],'--',\
 #                 label='r^2={:0.6f}'.format(regvals[2]**2))
a, b = np.polyfit(ppt_ts, pet_ts, 1)
tsline = plt.plot(ppt_ts, a*ppt_ts+b)
plt.ylabel('PET (mm)')
plt.xlabel('CHIRPS (mm)')
plt.title('Scatterplot of Precip and PET')
ax2.legend(loc='best')

  
# show the plot
plt.show()