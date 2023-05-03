#pet_intro
#%%

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import percentileofscore
import intake
import holoviews as hv
import cartopy.crs as ccrs
import geoviews as gv
import hvplot.xarray
#%matplotlib inline
import pandas as pd
import pymannkendall as mk
import time
from datetime import date
#from CoreFuncs import FILL_HOBBINS_DEKAD_GLOBAL
from scipy.stats import spearmanr, pearsonr, mode, linregress

hv.extension('bokeh', width=80)
count_bord = cartopy.feature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land','110m',facecolor='none')
#%% download here and graph it
PPT_dir = '/home/chc-data-out/products/CHIRPS-2.0/global_monthly/netcdf/'
infile = 'chirps-v2.0.monthly.nc'
PPT = xr.open_dataset(PPT_dir+infile)

# get data from https://data.chc.ucsb.edu/products/CHPclim/netcdf/
PET_dir = '/home/chc-data-out/people/husak/forCapstone/'
PET = xr.open_mfdataset(PET_dir+'*.nc',combine='nested',concat_dim='date').sortby('date')

in_dir = '/home/sandbox-people/husak/PythonScripts/'
infile = in_dir + 'sst.mnmean.nc'
SSTS = xr.open_dataset(infile)
#%%
minlat = -35.; maxlat = 0.
minlon = 5.; maxlon = 50.
moi = [9,10,11] # month of interest 1=Jan, 2=Feb, ... 12=Dec
#%%

PPTsub = PPT.sel(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),\
                 time=PPT['time.month'].isin([9, 10, 11]))
#%%
PETsub = PET.sel(lons=slice(minlon,maxlon), lats=slice(minlat,maxlat), \
                 date=PET['date.month'].isin([9, 10, 11])).sortby('date')
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
subNY,subNX,nyrs = tmpPET.shape

years = 1981 + np.arange(nyrs)
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
ax.set_title('Map of Correlation Month')# = {:02d})'.format(moi))  #put a title on the map
ax.coastlines(color='gray') #draw the coastlines in gray
ax.add_feature(count_bord,edgecolor='gray') #draw the country boundaries
#ax.add_feature(cartopy.feature.OCEAN,color='skyblue',zorder=100) #color the oceans
#fig.tight_layout()
cb = plt.colorbar(tmpmap,cax=fig.add_axes([0.1,0.09,0.8,0.03]), orientation='horizontal') #add colorbar

#%% GET REGIONAL AVERAGE RAINFALL AND CORRELATE WITH PIXEL-LEVEL SSTS
ppt_ts = PPTsub.precip.sel(latitude=slice(-23.,-14.),longitude=slice(25.,35.)).mean(axis=(1,2)).values
pet_ts = PETsub.PET.sel(lats=slice(-23.,-14.),lons=slice(25.,35.)).mean(axis=(0,1)).values

print(ppt_ts.shape,pet_ts.shape,'these should have the same numbers, but out of order')

#if the numbers aren't the same, then you might have to remove the final year from
# the PPT because it may have been updated
# ppt_ts = ppt_ts[:-1]

ppt_norm = (ppt_ts - ppt_ts.mean()) / ppt_ts.std()
pet_norm = (pet_ts - pet_ts.mean()) / pet_ts.std()

regvals = linregress(ppt_ts,pet_ts)
pet_est = regvals[1] + (regvals[0] * ppt_ts)

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_axes([0.1,0.1,0.85,0.80])
tsplot = ax1.plot(years,pet_ts,alpha=0.5,label='Hobbins PET')
tsplot = ax1.plot(years,ppt_ts,alpha=0.5,label='CHIRPS Precip')
plt.ylabel('Millimeters')
plt.xlabel('Year')
plt.title('Timeseries of OND Precip and PET')
ax1.legend(loc='best')
#plt.savefig('/home/sandbox-people/husak/PythonScripts/TempDelete.png',dpi=200)

fig = plt.figure(figsize = (6,6))
ax2 = fig.add_axes([0.15,0.1,0.80,0.80])
tsscat = ax2.plot(ppt_ts,pet_ts,'o',lw=0)
tsline = ax2.plot([ppt_ts.min(),ppt_ts.max()],[pet_est.max(),pet_est.min()],'--',\
                  label='r^2={:0.6f}'.format(regvals[2]**2))
plt.ylabel('PET (mm)')
plt.xlabel('CHIRPS (mm)')
plt.title('Scatterplot of Precip and PET')
ax2.legend(loc='best')

fig = plt.figure(figsize = (8,6))
ax3 = fig.add_axes([0.1,0.1,0.85,0.80])
tsplot = ax3.plot(years,pet_norm,alpha=0.5,label='Normalized PET')
tsplot = ax3.plot(years,ppt_norm,alpha=0.5,label='Normalized Precip')
ax3.legend(loc='best')

resids = pet_ts - pet_est

fig = plt.figure(figsize = (8,6))
ax1 = fig.add_axes([0.1,0.1,0.85,0.80])
tsplot = ax1.bar(years,resids,alpha=0.5,label='PET Residuals')
plt.ylabel('PET residual (mm)')
plt.xlabel('Year')
plt.title('Timeseries of MAM PET Residuals')
ax1.legend(loc='best')


#%%  put the 4 figures on a single window

fig = plt.figure(figsize = (12,10))
ax1 = plt.subplot(221)
tsplot = ax1.plot(years,pet_ts,alpha=0.5,label='Hobbins PET')
tsplot = ax1.plot(years,ppt_ts,alpha=0.5,label='CHIRPS Precip')
plt.ylabel('Millimeters')
plt.xlabel('Year')
plt.title('Timeseries of OND Precip and PET')
ax1.legend(loc='best')
#plt.savefig('/home/sandbox-people/husak/PythonScripts/TempDelete.png',dpi=200)

ax2 = plt.subplot(222)
tsscat = ax2.plot(ppt_ts,pet_ts,'o',lw=0)
tsline = ax2.plot([ppt_ts.min(),ppt_ts.max()],[pet_est.max(),pet_est.min()],'--',\
                  label='r^2={:0.6f}'.format(regvals[2]**2))
plt.ylabel('PET (mm)')
plt.xlabel('CHIRPS (mm)')
plt.title('Scatterplot of Precip and PET')
ax2.legend(loc='best')

ax3 = plt.subplot(223)
tsplot = ax3.plot(years,pet_norm,alpha=0.5,label='Normalized PET')
tsplot = ax3.plot(years,ppt_norm,alpha=0.5,label='Normalized Precip')
plt.ylabel('Normalized PPT and PET')
ax3.legend(loc='best')

resids = pet_ts - pet_est

ax4 = plt.subplot(224)
tsplot = ax4.bar(years,resids,alpha=0.5,label='PET Residuals')
plt.ylabel('PET residual (mm)')
plt.xlabel('Year')
plt.title('Timeseries of MAM PET Residuals')
ax1.legend(loc='best')

fig.tight_layout()
#%%
#calculate a rolling 3-month average, and select the average with the endmonth equal to SST_moi
SST_moi = 12 #the final month in the temporal average
#sst3 = SSTS.sst.rolling(time=3).mean().values
sstvals = SSTS.sst.rolling(time=3).mean().sel(time=(SSTS.time['time.month'] == SST_moi)).values

nyrs,NY,NX = sstvals.shape
corr_mat = np.zeros([NY,NX]) * np.nan

tic = time.time()
for x in range(NX):
  for y in range(NY):
    corr_mat[y,x] = linregress(resids[1:],sstvals[1:,y,x])[2]
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

#plt.savefig('/home/sandbox-people/husak/PythonScripts/PETresid_SST_corr.png',dpi=200)
# %%
